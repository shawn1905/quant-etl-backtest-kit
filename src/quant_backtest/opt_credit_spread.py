"""Capped vertical option credit spread backtest helpers (volatility family).

Builds on opt_plateau_clean (roll-free 1d_contracts features + causal pos60 +
strict vol > 0 and dte/maturity checks) to evaluate 2-leg capped credit spreads:
  - Bull Put Credit Spread: sell near-OTM Put, buy further-OTM protective Put
  - Bear Call Credit Spread: sell near-OTM Call, buy further-OTM protective Call

Pure Python standard library at import time (no pandas/pyarrow).
"""

from __future__ import annotations

import collections
import math
import statistics as st_mod
from typing import Any, Mapping, Sequence

from quant_backtest.opt_plateau_clean import (
    FEE_PER_LOT_SIDE_RMB,
    compute_cell_stats,
    is_tw_blocked,
    max_drawdown_r,
)
from quant_backtest import overfit_stats as OS


def causal_r20(r_same: Sequence[float | None], i: int) -> float | None:
    """Return 20-day same-contract cumulative log return at index i, or None if gapped."""
    if i < 20:
        return None
    win = r_same[i - 19 : i + 1]
    if any(x is None for x in win):
        return None
    return float(sum(win))  # type: ignore[arg-type]


def causal_atm_iv(
    cands: Sequence[Mapping[str, Any]],
    underlying_code: str,
    lo_abs_d: float = 0.35,
    hi_abs_d: float = 0.65,
) -> float | None:
    """Median IV of liquid near-ATM options matching underlying_code on day t."""
    ivs = [
        float(c["iv"])
        for c in cands
        if c.get("underlying") == underlying_code
        and lo_abs_d <= abs(float(c["delta"])) <= hi_abs_d
        and float(c["iv"]) > 0.0
    ]
    if not ivs:
        return None
    return float(st_mod.median(ivs))


def pick_credit_spread_pair(
    cands: Sequence[Mapping[str, Any]],
    cp: str,
    underlying_code: str,
    short_lo_d: float = 0.22,
    short_hi_d: float = 0.42,
    short_target_d: float = 0.30,
    long_lo_d: float = 0.05,
    long_hi_d: float = 0.20,
    long_target_d: float = 0.12,
    min_credit_ticks: float = 6.0,
    min_width_ticks: float = 2.0,
    min_leg_vol: float = 50.0,
    min_leg_oi: float = 200.0,
) -> tuple[Mapping[str, Any], Mapping[str, Any], float, float] | None:
    """Select (short_cand, long_cand, credit_t, width_px) on day t close.

    Bull Put spread (cp='P'): protective strike_long < strike_short.
    Bear Call spread (cp='C'): protective strike_long > strike_short.
    """
    eligible = [
        c
        for c in cands
        if c.get("cp") == cp
        and c.get("underlying") == underlying_code
        and float(c.get("vol", 0.0)) >= min_leg_vol
        and float(c.get("oi", 0.0)) >= min_leg_oi
        and float(c.get("strike", 0.0)) > 0.0
        and float(c.get("close", 0.0)) > 0.0
        and float(c.get("tick", 0.0)) > 0.0
        and float(c.get("mult", 0.0)) > 0.0
    ]
    short_pool = [
        c for c in eligible if short_lo_d <= abs(float(c["delta"])) <= short_hi_d
    ]
    if not short_pool:
        return None

    short_pool.sort(
        key=lambda c: (
            abs(abs(float(c["delta"])) - short_target_d),
            -float(c["vol"]),
            str(c["code"]),
        )
    )
    for sc in short_pool:
        tick = float(sc["tick"])
        mult = float(sc["mult"])
        mat = sc["mat"]
        k_short = float(sc["strike"])
        c_short = float(sc["close"])

        prot_pool = []
        for lc in eligible:
            if lc["code"] == sc["code"] or lc["mat"] != mat:
                continue
            if float(lc["tick"]) != tick or float(lc["mult"]) != mult:
                continue
            if not (long_lo_d <= abs(float(lc["delta"])) <= long_hi_d):
                continue
            k_long = float(lc["strike"])
            if cp == "P" and k_long >= k_short:
                continue
            if cp == "C" and k_long <= k_short:
                continue
            width_px = abs(k_short - k_long)
            credit_t = c_short - float(lc["close"])
            if credit_t < min_credit_ticks * tick:
                continue
            if width_px < max(min_width_ticks * tick, credit_t + tick):
                continue
            prot_pool.append(lc)

        if prot_pool:
            prot_pool.sort(
                key=lambda c: (
                    abs(abs(float(c["delta"])) - long_target_d),
                    -float(c["vol"]),
                    str(c["code"]),
                )
            )
            lc = prot_pool[0]
            width_px = abs(k_short - float(lc["strike"]))
            credit_t = c_short - float(lc["close"])
            return sc, lc, credit_t, width_px
    return None


def eval_credit_spread(
    short_cand: Mapping[str, Any],
    long_cand: Mapping[str, Any],
    width_px: float,
    dates: Sequence[str],
    i: int,
    h: int,
    px_map: Mapping[str, Mapping[str, tuple[float, float, float]]],
    require_fill_vol: bool = True,
    max_exit_date: str | None = None,
) -> dict[str, Any] | None:
    """Evaluate entering a 2-leg credit spread at t+1 open and exiting at min(t+1+h, mat_m2) close.

    Cost conventions:
      - 1x: 1 tick adverse slip per leg per side (4 ticks total) + 4 * 5 RMB fee
      - 6t: 1.5 tick adverse slip per leg per side (6 ticks total) + 4 * 5 RMB fee
      - 2x: 2 ticks adverse slip per leg per side (8 ticks total) + 8 * 5 RMB fee
    Risk unit 1R = capped spread max loss under 1x entry credit + 1x round-trip fee.
    """
    if i + 1 >= len(dates):
        return None
    entry_d = dates[i + 1]
    mat_m2 = min(str(short_cand["mat_m2"]), str(long_cand["mat_m2"]))
    if entry_d > mat_m2 or (max_exit_date is not None and entry_d > max_exit_date):
        return None

    s_px = px_map.get(str(short_cand["code"]), {})
    l_px = px_map.get(str(long_cand["code"]), {})
    eb_s = s_px.get(entry_d)
    eb_l = l_px.get(entry_d)
    if eb_s is None or eb_l is None:
        return None
    op_s, _, vol_s_in = eb_s
    op_l, _, vol_l_in = eb_l
    if op_s <= 0.0 or op_l <= 0.0:
        return None
    if require_fill_vol and (vol_s_in <= 0.0 or vol_l_in <= 0.0):
        return None

    target_idx = i + 1 + h
    early_capped = target_idx >= len(dates) or (
        max_exit_date is not None and dates[min(target_idx, len(dates) - 1)] > max_exit_date
    )
    max_idx = min(target_idx, len(dates) - 1)
    cap_exit_d = mat_m2 if max_exit_date is None else min(mat_m2, max_exit_date)

    exit_idx: int | None = None
    for idx in range(max_idx, i + 1, -1):
        d_ex = dates[idx]
        if d_ex > cap_exit_d:
            continue
        bs = s_px.get(d_ex)
        bl = l_px.get(d_ex)
        if bs is None or bl is None:
            continue
        if bs[1] < 0.0 or bl[1] < 0.0:
            continue
        if require_fill_vol and (bs[2] <= 0.0 or bl[2] <= 0.0):
            continue
        exit_idx = idx
        break

    if exit_idx is None:
        return None

    cl_s = s_px[dates[exit_idx]][1]
    cl_l = l_px[dates[exit_idx]][1]
    tick = float(short_cand["tick"])
    mult = float(short_cand["mult"])
    fee_px = FEE_PER_LOT_SIDE_RMB / mult

    def _net_px(slip_ticks: float, fee_mult: float) -> tuple[float, float, float]:
        ent_cred = max(0.0, op_s - slip_ticks * tick) - (op_l + slip_ticks * tick)
        ext_deb = (cl_s + slip_ticks * tick) - max(0.0, cl_l - slip_ticks * tick)
        pnl = ent_cred - ext_deb - 4.0 * fee_mult * fee_px
        return ent_cred, ext_deb, pnl

    ent1, _, pnl1_px = _net_px(1.0, 1.0)
    if ent1 <= 0.0 or ent1 >= width_px:
        return None
    _, _, pnl6t_px = _net_px(1.5, 1.0)
    _, _, pnl2_px = _net_px(2.0, 2.0)

    risk1_px = (width_px - ent1) + 4.0 * fee_px
    if risk1_px <= 0.0:
        return None

    return {
        "entry_date": entry_d,
        "exit_date": dates[exit_idx],
        "held_bars": exit_idx - (i + 1),
        "early_capped": early_capped,
        "r1": pnl1_px / risk1_px,
        "r_6t": pnl6t_px / risk1_px,
        "r2": pnl2_px / risk1_px,
        "pnl1_rmb": pnl1_px * mult,
        "pnl6t_rmb": pnl6t_px * mult,
        "pnl2_rmb": pnl2_px * mult,
        "risk1_rmb": risk1_px * mult,
    }


def run_credit_spread_config(
    cfg_id: str,
    fut_feats: Mapping[str, Mapping[str, Any]],
    pos60_map: Mapping[str, float | None],
    day_cands: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    px_by_code: Mapping[str, Mapping[str, tuple[float, float, float]]],
    lo: str,
    hi: str,
    *,
    h: int = 10,
    iv_rv_min: float = 1.20,
    min_credit_ticks: float = 6.0,
    min_leg_vol: float = 50.0,
    min_leg_oi: float = 200.0,
    use_mkt_align: bool = False,
    put_only: bool = False,
    require_f1_nonneg: bool = False,
    require_notw: bool = False,
    max_exit_date: str | None = None,
) -> dict[str, Any]:
    """Run one pre-registered credit spread recipe over [lo, hi] without overlapping holds per product."""
    events_1x: list[tuple[str, str, int, float]] = []
    events_6t: list[tuple[str, str, int, float]] = []
    events_2x: list[tuple[str, str, int, float]] = []
    trades: list[dict[str, Any]] = []
    diag: collections.Counter[str] = collections.Counter()
    cp_counts: collections.Counter[str] = collections.Counter()

    for prod, pf in sorted(fut_feats.items()):
        dates = pf["dates"]
        r_same = pf["r_same"]
        rv = pf["rv"]
        f1 = pf["f1_same"]
        dom = pf["dom"]
        i = 21
        nxt = 0
        while i < len(dates) - 2:
            dt_s = dates[i]
            if i < nxt or not (lo <= dt_s <= hi):
                i += 1
                continue

            diag["scan_bars"] += 1
            rv_v = rv[i]
            r20_v = causal_r20(r_same, i)
            if rv_v is None or rv_v <= 1e-4 or r20_v is None:
                diag["no_fut_state"] += 1
                i += 1
                continue

            if put_only:
                if r20_v <= 0.0:
                    diag["bear_skipped_put_only"] += 1
                    i += 1
                    continue
                if require_f1_nonneg and f1[i] < 0:
                    diag["f1_neg_skipped"] += 1
                    i += 1
                    continue
                cp = "P"
            else:
                cp = "P" if r20_v >= 0.0 else "C"

            if require_notw:
                blocked, reason = is_tw_blocked(pos60_map.get(dt_s))
                if blocked:
                    diag[reason or "tw_blocked"] += 1
                    i += 1
                    continue

            if use_mkt_align:
                p60 = pos60_map.get(dt_s)
                if p60 is None:
                    diag["no_mkt_state"] += 1
                    i += 1
                    continue
                if cp == "P" and p60 < 0.25:
                    diag["mkt_crash_block_put"] += 1
                    i += 1
                    continue
                if cp == "C" and p60 > 0.75:
                    diag["mkt_meltup_block_call"] += 1
                    i += 1
                    continue

            cands = day_cands.get(dt_s, {}).get(prod, ())
            dom_code = dom[i]
            atm_iv = causal_atm_iv(cands, dom_code)
            if atm_iv is None:
                diag["no_atm_iv"] += 1
                i += 1
                continue
            if (atm_iv / rv_v) < iv_rv_min:
                diag["ivrv_below_thresh"] += 1
                i += 1
                continue

            picked = pick_credit_spread_pair(
                cands,
                cp=cp,
                underlying_code=dom_code,
                min_credit_ticks=min_credit_ticks,
                min_leg_vol=min_leg_vol,
                min_leg_oi=min_leg_oi,
            )
            if picked is None:
                diag["no_valid_spread_pair"] += 1
                i += 1
                continue

            sc, lc, credit_t, width_px = picked
            res = eval_credit_spread(
                sc,
                lc,
                width_px,
                dates,
                i,
                h,
                px_by_code,
                require_fill_vol=True,
                max_exit_date=max_exit_date,
            )
            if res is None:
                diag["unfilled_or_missing_exit"] += 1
                i += 1
                continue

            diag["admitted"] += 1
            cp_counts[cp] += 1
            events_1x.append((dt_s, prod, 1, res["r1"]))
            events_6t.append((dt_s, prod, 1, res["r_6t"]))
            events_2x.append((dt_s, prod, 1, res["r2"]))
            trades.append(
                {
                    "cfg": cfg_id,
                    "signal_date": dt_s,
                    "entry_date": res["entry_date"],
                    "exit_date": res["exit_date"],
                    "product": prod,
                    "cp": cp,
                    "iv_rv": round(atm_iv / rv_v, 4),
                    "credit_ticks": round(credit_t / float(sc["tick"]), 2),
                    "held_bars": res["held_bars"],
                    "r1": res["r1"],
                    "r_6t": res["r_6t"],
                    "r2": res["r2"],
                }
            )
            nxt = i + 1 + res["held_bars"]
            i += 1

    return {
        "cfg": cfg_id,
        "events_1x": events_1x,
        "events_6t": events_6t,
        "events_2x": events_2x,
        "trades": trades,
        "diag": dict(diag),
        "cp_counts": dict(cp_counts),
    }


def summarize_credit_spread_suite(
    runs: Sequence[Mapping[str, Any]],
    nperm: int = 2000,
    seed_base: int = 20261117,
) -> list[dict[str, Any]]:
    """Apply compute_cell_stats + BH-FDR + 6-gate audit uniformly across N pre-registered configs."""
    stats_list: list[dict[str, Any] | None] = []
    pvals_valid: list[float] = []
    valid_indices: list[int] = []
    for idx, r in enumerate(runs):
        ev1 = r["events_1x"]
        ev2 = r["events_2x"]
        st1 = compute_cell_stats(ev1, ev2, nperm=nperm, seed=seed_base + idx, min_events=1)
        stats_list.append(st1)
        if st1 is not None and len(ev1) >= 30:
            valid_indices.append(idx)
            pvals_valid.append(float(st1["p"]))

    bh_map: dict[int, bool] = {}
    if pvals_valid:
        kept = OS.bh_fdr(pvals_valid, q=0.05)
        for idx_k, k_flag in zip(valid_indices, kept):
            bh_map[idx_k] = bool(k_flag)

    out: list[dict[str, Any]] = []
    for idx, r in enumerate(runs):
        cid = r["cfg"]
        ev1 = r["events_1x"]
        ev6t = r["events_6t"]
        st1 = stats_list[idx]

        mean_1x = st1["mean"] if st1 is not None else None
        mean_6t = (sum(e[3] for e in ev6t) / len(ev6t)) if ev6t else None
        mean_2x = st1["mean_2xcost"] if st1 is not None else None
        boot_lo = st1["lo"] if st1 is not None else None
        drop_best = st1["drop_best_month"] if st1 is not None else None
        n = len(ev1)
        blocks = st1["blocks"] if st1 is not None else 0
        bh_pass = bh_map.get(idx, False)

        gates = {
            "n_ge_30": n >= 30,
            "mean_1x_gt_0": mean_1x is not None and mean_1x > 0.0,
            "mean_2x_gt_0": mean_2x is not None and mean_2x > 0.0,
            "boot_lo_gt_0": boot_lo is not None and boot_lo > 0.0,
            "bh_pass": bh_pass,
            "drop_best_gt_0": drop_best is not None and drop_best > 0.0,
        }
        pass_cnt = sum(1 for v in gates.values() if v)
        out.append(
            {
                "cfg": cid,
                "n": n,
                "blocks": blocks,
                "win_rate": st1["win_rate"] if st1 is not None else 0.0,
                "mean_1x": mean_1x,
                "mean_6t": mean_6t,
                "mean_2x": mean_2x,
                "boot_lo": boot_lo,
                "perm_p": float(st1["p"]) if st1 is not None else 1.0,
                "bh_pass": bh_pass,
                "drop_best_month": drop_best,
                "max_dd_r": st1["max_dd_r"] if st1 is not None else 0.0,
                "years": st1["years"] if st1 is not None else {},
                "pos_years": st1["pos_years"] if st1 is not None else 0,
                "gates": gates,
                "pass_cnt": pass_cnt,
                "pass_all": pass_cnt == 6,
                "cp_counts": r["cp_counts"],
                "diag": r["diag"],
            }
        )
    return out


def run_to_portfolio_leg(
    run_res: Mapping[str, Any],
    family: str = "volatility",
) -> dict[str, Any]:
    """Convert run_credit_spread_config output into evaluate_portfolio_combo leg format."""
    daily_vals: dict[str, list[float]] = collections.defaultdict(list)
    monthly_totals: dict[str, float] = collections.defaultdict(float)
    monthly_blocks_tmp: dict[str, list[Any]] = collections.defaultdict(lambda: [0.0, 0])
    trades_out: list[dict[str, Any]] = []

    for tr in run_res["trades"]:
        d = tr["signal_date"]
        r1 = float(tr["r1"])
        daily_vals[d].append(r1)
        ym = d[:6]
        monthly_totals[ym] += r1
        monthly_blocks_tmp[ym][0] += r1
        monthly_blocks_tmp[ym][1] += 1
        trades_out.append(
            {
                "date": d,
                "asset": tr["product"],
                "direction": 1,
                "net_1x": r1,
                "net_2x": float(tr["r2"]),
            }
        )

    return {
        "recipe_id": run_res["cfg"],
        "family": family,
        "daily_returns": {d: sum(vs) / len(vs) for d, vs in sorted(daily_vals.items())},
        "monthly_totals": dict(sorted(monthly_totals.items())),
        "monthly_blocks": {ym: (float(v[0]), int(v[1])) for ym, v in sorted(monthly_blocks_tmp.items())},
        "trades": trades_out,
    }
