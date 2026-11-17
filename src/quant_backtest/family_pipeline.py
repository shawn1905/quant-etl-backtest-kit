"""Modular 5-family research pipeline: entry gate funnels, 3 standard simulators, and cross-family combo evaluator.

Design goals:
  - Keep the 5 strategy families cleanly separated: trend, mean_reversion, carry (sealed), relative_value, volatility.
  - Multi-condition entries (e.g. liquidation spike + low price percentile + ladder bids) are composed via StrategyRecipe gates + simulator, NEVER confused with multi-strategy portfolio combos.
  - Variant differences must be proven via prove_variant_diff(); sealed families/tokens fail closed.
  - Pure Python standard library at import time (no pandas/pyarrow).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Mapping, Sequence

from quant_backtest.opt_plateau_clean import compute_cell_stats, eval_option_leg, max_drawdown_r
from quant_backtest import overfit_stats as OS

FAMILIES: tuple[str, ...] = ("trend", "mean_reversion", "carry", "relative_value", "volatility")
SEALED_FAMILIES: frozenset[str] = frozenset(("carry",))
SEALED_TOKENS: tuple[str, ...] = ("mom60", "b1_b5", "b1-b5", "b14_carry", "carry_b14")
SIMULATOR_KINDS: tuple[str, ...] = ("bar_order", "ladder_grid", "option_leg")
GATE_KINDS: tuple[str, ...] = ("trigger", "location", "env_brake")


@dataclass(frozen=True)
class GateSpec:
    """One causal entry condition in a StrategyRecipe funnel."""

    name: str
    kind: str
    fn: Callable[[Mapping[str, Any]], tuple[bool, int]]


@dataclass(frozen=True)
class StrategyRecipe:
    """Declarative single-strategy specification under one of the 5 strategy families."""

    recipe_id: str
    family: str
    simulator_kind: str
    gates: tuple[GateSpec, ...]
    simulator_params: Mapping[str, Any] = field(default_factory=dict)
    cooldown_bars: int = 0
    description: str = ""


def validate_recipe(recipe: StrategyRecipe) -> None:
    """Reject unknown families, sealed directions/families, bad simulator kinds, or missing trigger gates."""
    if recipe.family not in FAMILIES:
        raise ValueError(f"Unknown strategy family: {recipe.family!r}")
    if recipe.family in SEALED_FAMILIES:
        raise ValueError(f"Sealed family or rule: family {recipe.family!r} is sealed")
    low_id = recipe.recipe_id.lower()
    for tok in SEALED_TOKENS:
        if tok in low_id:
            raise ValueError(f"Sealed family or rule: token {tok!r} in recipe_id {recipe.recipe_id!r}")
    if recipe.simulator_kind not in SIMULATOR_KINDS:
        raise ValueError(f"Unknown simulator_kind: {recipe.simulator_kind!r}")
    if not recipe.gates:
        raise ValueError("StrategyRecipe must contain at least one GateSpec")
    has_trigger = False
    for g in recipe.gates:
        if g.kind not in GATE_KINDS:
            raise ValueError(f"Unknown gate kind {g.kind!r} on gate {g.name!r}")
        if g.kind == "trigger":
            has_trigger = True
    if not has_trigger:
        raise ValueError("StrategyRecipe requires at least one 'trigger' gate")


def run_entry_funnel(
    candidates: Sequence[Mapping[str, Any]],
    gates: Sequence[GateSpec],
    cooldown_bars: int = 0,
) -> dict[str, Any]:
    """Run candidates through ordered AND gates and track per-gate attrition + cooldown."""
    ordered = sorted(candidates, key=lambda c: (str(c["date"]), str(c["asset"])))
    funnel: collections.Counter[str] = collections.Counter()
    admitted: list[dict[str, Any]] = []
    next_allowed_idx: dict[str, int] = collections.defaultdict(int)
    asset_seq_idx: dict[str, int] = collections.defaultdict(int)

    for cand in ordered:
        asset = str(cand["asset"])
        if "day_idx" in cand and cand["day_idx"] is not None:
            cur_idx = int(cand["day_idx"])
        else:
            cur_idx = asset_seq_idx[asset]
            asset_seq_idx[asset] += 1

        funnel["total_candidates"] += 1
        direction = 0
        blocked_reason: str | None = None

        for g in gates:
            passed, g_dir = g.fn(cand)
            if not passed:
                blocked_reason = f"blocked:{g.name}"
                break
            if g.kind == "trigger":
                if g_dir not in (1, -1):
                    blocked_reason = f"blocked:{g.name}"
                    break
                if direction == 0:
                    direction = int(g_dir)
                elif int(g_dir) != direction:
                    blocked_reason = "blocked:direction_conflict"
                    break
            else:
                if g_dir in (1, -1):
                    if direction == 0:
                        direction = int(g_dir)
                    elif int(g_dir) != direction:
                        blocked_reason = "blocked:direction_conflict"
                        break

        if blocked_reason is not None:
            funnel[blocked_reason] += 1
            continue
        if direction not in (1, -1):
            funnel["blocked:no_direction"] += 1
            continue

        if cooldown_bars > 0 and cur_idx < next_allowed_idx[asset]:
            funnel["blocked:cooldown"] += 1
            continue

        funnel["admitted"] += 1
        item = dict(cand)
        item["direction"] = direction
        admitted.append(item)
        if cooldown_bars > 0:
            next_allowed_idx[asset] = cur_idx + cooldown_bars + 1

    return {"admitted": admitted, "funnel": dict(funnel)}


def prove_variant_diff(base_run: Mapping[str, Any], variant_run: Mapping[str, Any]) -> dict[str, Any]:
    """Fail if a declared variant produced identical funnel and trade stream to its baseline."""
    f_changed = base_run.get("funnel") != variant_run.get("funnel")
    b_sig = [
        (t["date"], t["asset"], t["direction"], round(float(t["net_1x"]), 10))
        for t in base_run.get("trades", [])
    ]
    v_sig = [
        (t["date"], t["asset"], t["direction"], round(float(t["net_1x"]), 10))
        for t in variant_run.get("trades", [])
    ]
    t_changed = b_sig != v_sig
    if not f_changed and not t_changed:
        raise ValueError("Variant change had zero effect on funnel or trades")
    return {
        "funnel_changed": f_changed,
        "trades_changed": t_changed,
        "base_admitted": base_run.get("funnel", {}).get("admitted", 0),
        "variant_admitted": variant_run.get("funnel", {}).get("admitted", 0),
        "base_trades": len(b_sig),
        "variant_trades": len(v_sig),
    }


def simulate_bar_order(
    admitted: Sequence[Mapping[str, Any]],
    bar_lookup: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None],
    params: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Simulate single-leg entry/exit trades with 1x & 2x cost/slippage and optional risk-unit R normalization."""
    p = dict(params or {})
    normalize_by_risk = bool(p.get("normalize_by_risk", False))
    default_fee = float(p.get("fee_pct", 0.0))
    default_slip = float(p.get("slip_pct", 0.0))

    trades: list[dict[str, Any]] = []
    unvalued: collections.Counter[str] = collections.Counter()

    for cand in admitted:
        res = bar_lookup(cand, p)
        if not res:
            unvalued["missing_bar"] += 1
            continue
        if res.get("unvalued_reason"):
            unvalued[str(res["unvalued_reason"])] += 1
            continue
        entry = float(res["entry"])
        exit_px = float(res["exit"])
        if entry <= 0.0 or exit_px <= 0.0 or not math.isfinite(entry + exit_px):
            unvalued["bad_price"] += 1
            continue

        direction = int(cand["direction"])
        gross = direction * (exit_px - entry) / entry
        fee_ret = float(res.get("fee_ret", default_fee))
        slip_ret = float(res.get("slip_ret", default_slip))
        net1 = gross - fee_ret - slip_ret
        net2 = gross - 2.0 * fee_ret - 2.0 * slip_ret if res.get("double_both") else gross - fee_ret - 2.0 * slip_ret

        risk_unit = res.get("risk_unit")
        if normalize_by_risk:
            if risk_unit is None or float(risk_unit) <= 0.0 or not math.isfinite(float(risk_unit)):
                unvalued["bad_risk_unit"] += 1
                continue
            denom = float(risk_unit)
            val_gross, val_1x, val_2x = gross / denom, net1 / denom, net2 / denom
        else:
            val_gross, val_1x, val_2x = gross, net1, net2

        trades.append(
            {
                "date": str(cand["date"]),
                "asset": str(cand["asset"]),
                "direction": direction,
                "gross": val_gross,
                "net_1x": val_1x,
                "net_2x": val_2x,
                "meta": {"entry": entry, "exit": exit_px, "risk_unit": risk_unit},
            }
        )

    return trades, dict(unvalued)


def _simulate_ladder_session(
    ohlc: Sequence[tuple[float, float, float, float]],
    direction: int,
    prior_range: float,
    params: Mapping[str, Any],
    cost_mult: float = 1.0,
) -> tuple[float, int, float] | None:
    """Simulate resting limit-order ladder (buy dips for +1, sell spikes for -1) over OHLC bars."""
    if not ohlc:
        return None
    p0 = float(ohlc[0][0])
    if p0 <= 0.0 or not math.isfinite(p0) or not math.isfinite(prior_range) or prior_range < 0.0:
        return None

    levels = int(params.get("levels", 6))
    range_mult = float(params.get("range_multiplier", 0.30))
    min_step = float(params.get("min_step", 0.003))
    unit_notional = float(params.get("unit_notional", 1.0))
    maker_fee = float(params.get("maker_fee", 0.0002)) * cost_mult
    taker_fee = float(params.get("taker_fee", 0.0005)) * cost_mult
    spread = float(params.get("spread", 0.0005)) * cost_mult
    eps = float(params.get("penetration", 0.0005))
    tp_steps = params.get("take_profit_steps")
    tp_steps_i = int(tp_steps) if tp_steps is not None else None

    if levels < 1 or unit_notional <= 0.0:
        return None
    step = max(range_mult * prior_range, min_step)

    # Precompute entry levels: for long (+1) below P0, for short (-1) above P0
    entry_lvls = [
        p0 * (1.0 - direction * (k + 1) * step) for k in range(levels)
    ]
    if any(lvl <= 0.0 for lvl in entry_lvls):
        return None

    next_fill_k = 0
    open_tranches: list[float] = []  # entry price per active unit tranche
    realized_cash = 0.0
    filled_count = 0

    for o, h, l, c in ohlc:
        # Conservative path heuristic matching imported/liq/dgrid.py
        legs = (("d", l), ("u", h)) if c >= o else (("u", h), ("d", l))
        for side, px_ext in legs:
            want_entry_side = "d" if direction > 0 else "u"
            if side == want_entry_side:
                while next_fill_k < levels:
                    lvl = entry_lvls[next_fill_k]
                    triggered = (px_ext <= lvl * (1.0 - eps)) if direction > 0 else (px_ext >= lvl * (1.0 + eps))
                    if not triggered:
                        break
                    qty = unit_notional / lvl
                    if direction > 0:
                        realized_cash -= qty * lvl * (1.0 + maker_fee)
                    else:
                        realized_cash += qty * lvl * (1.0 - maker_fee)
                    open_tranches.append(lvl)
                    next_fill_k += 1
                    filled_count += 1
            elif tp_steps_i is not None and open_tranches:
                # Take profit highest/latest tranches first when target rebound touched
                remaining: list[float] = []
                for e_lvl in open_tranches:
                    tp_lvl = e_lvl * (1.0 + direction * tp_steps_i * step)
                    hit_tp = (px_ext >= tp_lvl * (1.0 + eps)) if direction > 0 else (px_ext <= tp_lvl * (1.0 - eps))
                    if hit_tp:
                        qty = unit_notional / e_lvl
                        if direction > 0:
                            realized_cash += qty * tp_lvl * (1.0 - maker_fee)
                        else:
                            realized_cash -= qty * tp_lvl * (1.0 + maker_fee)
                    else:
                        remaining.append(e_lvl)
                open_tranches = remaining

    c_last = float(ohlc[-1][3])
    if c_last <= 0.0 or not math.isfinite(c_last):
        return None
    for e_lvl in open_tranches:
        qty = unit_notional / e_lvl
        if direction > 0:
            realized_cash += qty * c_last * (1.0 - spread) * (1.0 - taker_fee)
        else:
            realized_cash -= qty * c_last * (1.0 + spread) * (1.0 + taker_fee)

    if filled_count == 0:
        return 0.0, 0, step

    max_budget = levels * unit_notional
    return realized_cash / max_budget, filled_count, step


def simulate_ladder_grid(
    admitted: Sequence[Mapping[str, Any]],
    session_lookup: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None],
    params: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Simulate ladder limit-order entries (e.g. post-liquidation low-dip accumulation)."""
    p = dict(params or {})
    normalize_by_risk = bool(p.get("normalize_by_risk", False))
    trades: list[dict[str, Any]] = []
    unvalued: collections.Counter[str] = collections.Counter()

    for cand in admitted:
        sess = session_lookup(cand, p)
        if not sess:
            unvalued["missing_session"] += 1
            continue
        if sess.get("unvalued_reason"):
            unvalued[str(sess["unvalued_reason"])] += 1
            continue
        ohlc = sess.get("ohlc") or []
        prior_range = sess.get("prior_range")
        if prior_range is None:
            unvalued["missing_prior_range"] += 1
            continue
        sess_p = {
            **p,
            **{
                k: sess[k]
                for k in ("maker_fee", "taker_fee", "spread", "min_step", "penetration")
                if k in sess and sess[k] is not None
            },
        }
        direction = int(cand["direction"])
        res1 = _simulate_ladder_session(ohlc, direction, float(prior_range), sess_p, cost_mult=1.0)
        res2 = _simulate_ladder_session(ohlc, direction, float(prior_range), sess_p, cost_mult=2.0)
        res0 = _simulate_ladder_session(ohlc, direction, float(prior_range), sess_p, cost_mult=0.0)
        if res1 is None or res2 is None or res0 is None:
            unvalued["bad_session_prices"] += 1
            continue
        ret1, filled1, step = res1
        ret2, _, _ = res2
        gross, _, _ = res0
        if filled1 == 0:
            unvalued["no_fills"] += 1
            continue

        risk_unit = sess.get("risk_unit")
        if normalize_by_risk:
            try:
                denom = float(risk_unit)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                unvalued["bad_risk_unit"] += 1
                continue
            if not math.isfinite(denom) or denom <= 0.0:
                unvalued["bad_risk_unit"] += 1
                continue
            val_gross, val_1x, val_2x = gross / denom, ret1 / denom, ret2 / denom
        else:
            val_gross, val_1x, val_2x = gross, ret1, ret2

        trades.append(
            {
                "date": str(cand["date"]),
                "asset": str(cand["asset"]),
                "direction": direction,
                "gross": val_gross,
                "net_1x": val_1x,
                "net_2x": val_2x,
                "meta": {"filled_levels": filled1, "step": step, "risk_unit": risk_unit},
            }
        )

    return trades, dict(unvalued)


def simulate_option_leg(
    admitted: Sequence[Mapping[str, Any]],
    opt_lookup: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None],
    params: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Simulate single-leg option trades using clean fill-vol-gated option evaluator."""
    p = dict(params or {})
    require_fill_vol = bool(p.get("require_fill_vol", True))
    trades: list[dict[str, Any]] = []
    unvalued: collections.Counter[str] = collections.Counter()

    for cand in admitted:
        ctx = opt_lookup(cand, p)
        if not ctx:
            unvalued["missing_option_ctx"] += 1
            continue
        if ctx.get("unvalued_reason"):
            unvalued[str(ctx["unvalued_reason"])] += 1
            continue
        res = eval_option_leg(
            ctx["cand"],
            ctx["dates"],
            int(ctx["sig_idx"]),
            int(ctx["h"]),
            ctx["px_map"],
            require_fill_vol=require_fill_vol,
            require_full_horizon=bool(ctx.get("require_full_horizon", False)),
            max_exit_date=ctx.get("max_exit_date"),
        )
        if res is None:
            unvalued["missing_entry_or_exit"] += 1
            continue
        pnl1_rmb, pnl2_rmb, risk1_rmb, r1, held_bars, _ = res
        r2 = pnl2_rmb / risk1_rmb
        trades.append(
            {
                "date": str(cand["date"]),
                "asset": str(cand["asset"]),
                "direction": int(cand["direction"]),
                "gross": r1,
                "net_1x": r1,
                "net_2x": r2,
                "meta": {"pnl1_rmb": pnl1_rmb, "risk1_rmb": risk1_rmb, "held_bars": held_bars},
            }
        )

    return trades, dict(unvalued)


def run_recipe(
    recipe: StrategyRecipe,
    candidates: Sequence[Mapping[str, Any]],
    sim_lookup: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None],
    *,
    nperm: int = 2000,
    seed: int = 20261117,
    min_events: int = 30,
) -> dict[str, Any]:
    """Validate recipe, execute funnel + designated simulator, and return standardized run evaluation."""
    validate_recipe(recipe)
    funnel_res = run_entry_funnel(candidates, recipe.gates, cooldown_bars=recipe.cooldown_bars)
    if recipe.simulator_kind == "bar_order":
        trades, unvalued = simulate_bar_order(funnel_res["admitted"], sim_lookup, recipe.simulator_params)
    elif recipe.simulator_kind == "ladder_grid":
        trades, unvalued = simulate_ladder_grid(funnel_res["admitted"], sim_lookup, recipe.simulator_params)
    else:
        trades, unvalued = simulate_option_leg(funnel_res["admitted"], sim_lookup, recipe.simulator_params)

    return evaluate_recipe_run(
        recipe,
        funnel_res,
        trades,
        unvalued,
        nperm=nperm,
        seed=seed,
        min_events=min_events,
    )


def evaluate_recipe_run(
    recipe: StrategyRecipe,
    funnel_res: Mapping[str, Any],
    trades: Sequence[Mapping[str, Any]],
    sim_unvalued: Mapping[str, int],
    *,
    nperm: int = 2000,
    seed: int = 20261117,
    min_events: int = 30,
) -> dict[str, Any]:
    """Produce standardized single-strategy metrics, monthly blocks, and daily return series."""
    ev = [(t["date"], t["asset"], t["direction"], float(t["net_1x"])) for t in trades]
    ev2 = [(t["date"], t["asset"], t["direction"], float(t["net_2x"])) for t in trades]
    stats = compute_cell_stats(ev, ev2, nperm=nperm, seed=seed, min_events=min_events)

    monthly_blocks: dict[str, list[Any]] = collections.defaultdict(lambda: [0.0, 0])
    daily_vals: dict[str, list[float]] = collections.defaultdict(list)
    for t in trades:
        ym = str(t["date"])[:6]
        monthly_blocks[ym][0] += float(t["net_1x"])
        monthly_blocks[ym][1] += 1
        daily_vals[str(t["date"])].append(float(t["net_1x"]))

    blocks_tuple = {ym: (float(v[0]), int(v[1])) for ym, v in sorted(monthly_blocks.items())}
    daily_returns = {d: sum(vs) / len(vs) for d, vs in sorted(daily_vals.items())}
    monthly_totals = {ym: float(v[0]) for ym, v in sorted(monthly_blocks.items())}

    basic = {
        "n": len(trades),
        "mean_gross": round(sum(float(t["gross"]) for t in trades) / len(trades), 6) if trades else None,
        "mean_net_1x": round(sum(float(t["net_1x"]) for t in trades) / len(trades), 6) if trades else None,
        "mean_net_2x": round(sum(float(t["net_2x"]) for t in trades) / len(trades), 6) if trades else None,
        "win_rate": round(sum(1 for t in trades if float(t["net_1x"]) > 0.0) / len(trades), 4) if trades else None,
        "max_dd": round(max_drawdown_r(ev), 4) if ev else 0.0,
    }

    return {
        "recipe_id": recipe.recipe_id,
        "family": recipe.family,
        "simulator_kind": recipe.simulator_kind,
        "description": recipe.description,
        "funnel": dict(funnel_res["funnel"]),
        "sim_unvalued": dict(sim_unvalued),
        "basic": basic,
        "stats": stats,
        "trades": list(trades),
        "monthly_blocks": blocks_tuple,
        "monthly_totals": monthly_totals,
        "daily_returns": daily_returns,
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def evaluate_portfolio_combo(
    combo_id: str,
    runs: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | None = None,
    *,
    calendar_dates: Sequence[str] | None = None,
    n_boot: int = 2000,
    seed: int = 20261117,
) -> dict[str, Any]:
    """Combine 2+ single-strategy runs across orthogonal legs/families and test incremental improvement."""
    if len(runs) < 2:
        raise ValueError("Portfolio combo requires at least 2 strategy runs")
    k = len(runs)
    if weights is None:
        ws = [1.0 / k] * k
    else:
        if len(weights) != k or any(w <= 0.0 for w in weights):
            raise ValueError("weights must match runs length with strictly positive values")
        s_w = sum(weights)
        ws = [w / s_w for w in weights]

    names = [str(r["recipe_id"]) for r in runs]
    all_days = sorted(calendar_dates) if calendar_dates is not None else sorted({d for r in runs for d in r["daily_returns"]})
    all_months = sorted({d[:6] for d in all_days} | {ym for r in runs for ym in r["monthly_totals"]})

    # 1) Correlation matrices on union daily returns and union monthly totals (0 when flat)
    daily_corr: dict[str, dict[str, float | None]] = {n: {} for n in names}
    monthly_corr: dict[str, dict[str, float | None]] = {n: {} for n in names}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            d_xi = [float(runs[i]["daily_returns"].get(d, 0.0)) for d in all_days]
            d_xj = [float(runs[j]["daily_returns"].get(d, 0.0)) for d in all_days]
            daily_corr[ni][nj] = round(v, 4) if (v := _pearson(d_xi, d_xj)) is not None else None
            m_xi = [float(runs[i]["monthly_totals"].get(m, 0.0)) for m in all_months]
            m_xj = [float(runs[j]["monthly_totals"].get(m, 0.0)) for m in all_months]
            monthly_corr[ni][nj] = round(v, 4) if (v := _pearson(m_xi, m_xj)) is not None else None

    # 2) Calendar-book weighted portfolio daily series & MaxDD comparison
    def _series_metrics(vals: Sequence[float]) -> dict[str, Any]:
        cur = peak = mdd = 0.0
        for x in vals:
            cur += x
            if cur > peak:
                peak = cur
            mdd = max(mdd, peak - cur)
        n = len(vals)
        mean_v = sum(vals) / n if n else 0.0
        sd_v = math.sqrt(sum((x - mean_v) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        sharpe = (mean_v / sd_v * math.sqrt(252.0)) if sd_v > 0.0 else None
        return {"days": n, "total": round(cur, 4), "mean_day": round(mean_v, 6), "sharpe_ann": round(sharpe, 3) if sharpe is not None else None, "max_dd": round(mdd, 4)}

    leg_day_series = [[float(r["daily_returns"].get(d, 0.0)) for d in all_days] for r in runs]
    combo_day_series = [sum(ws[j] * leg_day_series[j][di] for j in range(k)) for di in range(len(all_days))]
    leg_book_metrics = {names[j]: _series_metrics(leg_day_series[j]) for j in range(k)}
    combo_book_metrics = _series_metrics(combo_day_series)

    # Monthly calendar-book consistency
    combo_month_vals = {
        m: sum(ws[j] * float(runs[j]["monthly_totals"].get(m, 0.0)) for j in range(k))
        for m in all_months
    }
    by_year: dict[str, float] = collections.defaultdict(float)
    for m, val in combo_month_vals.items():
        by_year[m[:4]] += val
    pos_years = sum(1 for v in by_year.values() if v > 0.0)

    # 3) Pooled event-level stream & block_bootstrap_diff vs each leg
    pooled_events = []
    for j, r in enumerate(runs):
        scale = ws[j] * k
        for t in r["trades"]:
            pooled_events.append((t["date"], f"{names[j]}:{t['asset']}", t["direction"], float(t["net_1x"]) * scale))
    pooled_events.sort(key=lambda x: (x[0], x[1]))

    combo_blocks: dict[str, tuple[float, int]] = {}
    tmp_blk: dict[str, list[Any]] = collections.defaultdict(lambda: [0.0, 0])
    for dt_s, _, _, val in pooled_events:
        ym = dt_s[:6]
        tmp_blk[ym][0] += val
        tmp_blk[ym][1] += 1
    combo_blocks = {ym: (float(v[0]), int(v[1])) for ym, v in sorted(tmp_blk.items())}

    vs_legs: dict[str, dict[str, Any]] = {}
    for j, r in enumerate(runs):
        pt, lo, hi = OS.block_bootstrap_diff(combo_blocks, r["monthly_blocks"], n=n_boot, seed=seed + j)
        leg_mdd = leg_book_metrics[names[j]]["max_dd"]
        dd_cut = (1.0 - combo_book_metrics["max_dd"] / leg_mdd) if leg_mdd > 0.0 else None
        vs_legs[names[j]] = {
            "diff_mean": round(pt, 5) if pt is not None else None,
            "diff_ci95_lo": round(lo, 5) if lo is not None else None,
            "diff_ci95_hi": round(hi, 5) if hi is not None else None,
            "book_dd_cut": round(dd_cut, 4) if dd_cut is not None else None,
        }

    pooled_stats = compute_cell_stats(pooled_events, pooled_events, nperm=1000, seed=seed, min_events=1)
    return {
        "combo_id": combo_id,
        "members": [{"recipe_id": r["recipe_id"], "family": r["family"], "weight": ws[j]} for j, r in enumerate(runs)],
        "daily_corr": daily_corr,
        "monthly_corr": monthly_corr,
        "calendar_book_combo": combo_book_metrics,
        "calendar_book_legs": leg_book_metrics,
        "combo_by_year_sum": {y: round(v, 4) for y, v in sorted(by_year.items())},
        "combo_pos_years": f"{pos_years}/{len(by_year)}",
        "pooled_event_stats": {
            "n": len(pooled_events),
            "mean": round(pooled_stats["mean"], 4) if pooled_stats else None,
            "lo": round(pooled_stats["lo"], 4) if pooled_stats else None,
            "p": round(pooled_stats["p"], 4) if pooled_stats else None,
            "max_dd": round(max_drawdown_r(pooled_events), 2),
        },
        "vs_legs": vs_legs,
    }
