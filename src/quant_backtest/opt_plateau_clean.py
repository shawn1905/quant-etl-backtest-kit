"""Clean same-contract & roll-free COMBO_OPT_PLATEAU verification engine.

Fixes two structural biases in the legacy option plateau scripts
(combo_opt_plateau_scan.py & opt_single_leg_scan.py):
  1. Computes F1_20 breakouts, RV_20 realized volatility, and the NO_TW equal-weight
     market pos60 state strictly from 1d_contracts same-contract returns / physical
     contract bars (never across unadjusted 1d_fut roll jumps).
  2. Verifies non-zero traded volume (vol > 0) on both option entry and exit bars
     and supports optional futures-underlying contract alignment diagnostics.

Pure Python standard library at import time (no pandas/pyarrow dependency).
"""

from __future__ import annotations

import collections
import datetime as dt
import math
import random
import re
import statistics as st_mod
from typing import Any, Mapping, Sequence

from quant_backtest.contracts import dominant_by_day, split_code
from quant_backtest import overfit_stats as OS

H_GRID: tuple[int, ...] = (10, 15, 20, 30)
CAP_GRID: tuple[tuple[str, float | None], ...] = (
    ("CAP085", 0.85),
    ("CAP095", 0.95),
    ("CAP105", 1.05),
    ("NOLIM", None),
)
FEE_PER_LOT_SIDE_RMB = 5.0
MAX_CAL_GAP_DAYS = 20
_TRAILING_DIGITS = re.compile(r"\d+$")


def cap_tag(iv_cap: float | None) -> str:
    return "NOLIM" if iv_cap is None else f"CAP{int(round(iv_cap * 100)):03d}"


def cell_id(h: int, iv_cap: float | None) -> str:
    return f"OPT_PLATEAU|h{h}_{cap_tag(iv_cap)}"


def prod_from_underlying(underlying: str) -> str | None:
    """Extract lowercase product code from futures underlying symbol."""
    if not underlying:
        return None
    _, prod = split_code(str(underlying).split("@")[-1])
    if prod:
        return prod
    tail = str(underlying).split("@")[-1].split(".")[-1]
    cleaned = _TRAILING_DIGITS.sub("", tail).lower()
    return cleaned if cleaned and cleaned.isalpha() else None


def _pos_float(val: Any) -> float | None:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if (math.isfinite(x) and x > 0.0) else None


def _nonneg_float(val: Any) -> float | None:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if (math.isfinite(x) and x >= 0.0) else None


def _gap_days(d1: str, d2: str) -> int:
    return (dt.datetime.strptime(d2, "%Y%m%d") - dt.datetime.strptime(d1, "%Y%m%d")).days


def build_clean_fut_features(
    rows_by_product: Mapping[str, Sequence[Mapping[str, Any]]],
    min_len: int = 30,
) -> dict[str, dict[str, Any]]:
    """Build roll-free futures features from 1d_contracts physical contract rows.

    Returns per-product dict with:
      - dates: sorted trading dates where a valid dominant bar exists
      - dom: dominant contract code per day
      - r_same: same-contract daily log return log(c_t / c_{t-1}), or None if unavailable
      - rv: 20d annualized realized vol from r_same (strictly causal <= t)
      - f1_same: 20d breakout (+1/-1/0) strictly on day t dominant contract's own 21d bars
      - f1_chain: 20d breakout (+1/-1/0) on same-contract-return-adjusted chain
    """
    sqrt252 = math.sqrt(252.0)
    feats: dict[str, dict[str, Any]] = {}

    for prod, raw_rows in sorted(rows_by_product.items()):
        valid_rows: list[dict[str, Any]] = []
        by_cd: dict[tuple[str, str], dict[str, float]] = {}
        for r in raw_rows:
            d = str(r.get("date") or "").replace("-", "")[:8]
            code = str(r.get("code") or "")
            op = _pos_float(r.get("open"))
            hi = _pos_float(r.get("high"))
            lo = _pos_float(r.get("low"))
            cl = _pos_float(r.get("close"))
            if not d or not code or op is None or hi is None or lo is None or cl is None:
                continue
            oi = _nonneg_float(r.get("oi"))
            vol = _nonneg_float(r.get("vol")) or 0.0
            rec = {"date": d, "code": code, "open": op, "high": hi, "low": lo, "close": cl, "oi": oi, "vol": vol}
            valid_rows.append(rec)
            by_cd[(code, d)] = {"open": op, "high": hi, "low": lo, "close": cl, "vol": vol}

        dom_map = dominant_by_day(valid_rows)
        dates = sorted(dom_map)
        n = len(dates)
        if n < min_len:
            continue

        dom_codes = [dom_map[d] for d in dates]
        r_same: list[float | None] = [None] * n
        seg = [0] * n
        ch_c = [0.0] * n
        ch_h = [0.0] * n
        ch_l = [0.0] * n

        first_bar = by_cd[(dom_codes[0], dates[0])]
        ch_c[0] = first_bar["close"]
        ch_h[0] = first_bar["high"]
        ch_l[0] = first_bar["low"]

        for k in range(1, n):
            d_prev, d_cur = dates[k - 1], dates[k]
            c_prev, c_cur = dom_codes[k - 1], dom_codes[k]
            bar_cur = by_cd[(c_cur, d_cur)]

            rs: float | None = None
            if _gap_days(d_prev, d_cur) <= MAX_CAL_GAP_DAYS:
                a_prev, b_prev = by_cd.get((c_prev, d_prev)), by_cd.get((c_prev, d_cur))
                if a_prev is not None and b_prev is not None:
                    rs = math.log(b_prev["close"] / a_prev["close"])
                else:
                    a_cur, b_cur = by_cd.get((c_cur, d_prev)), by_cd.get((c_cur, d_cur))
                    if a_cur is not None and b_cur is not None:
                        rs = math.log(b_cur["close"] / a_cur["close"])
            r_same[k] = rs

            if rs is None:
                seg[k] = seg[k - 1] + 1
                ch_c[k] = bar_cur["close"]
                ch_h[k] = bar_cur["high"]
                ch_l[k] = bar_cur["low"]
            else:
                seg[k] = seg[k - 1]
                ch_c[k] = ch_c[k - 1] * math.exp(rs)
                ch_h[k] = ch_c[k] * (bar_cur["high"] / bar_cur["close"])
                ch_l[k] = ch_c[k] * (bar_cur["low"] / bar_cur["close"])

        rv: list[float | None] = [None] * n
        f1_same = [0] * n
        f1_chain = [0] * n

        for i in range(20, n):
            win_r = r_same[i - 19 : i + 1]
            if all(x is not None for x in win_r):
                rv[i] = st_mod.pstdev(win_r) * sqrt252  # type: ignore[arg-type]

            if seg[i] == seg[i - 20]:
                hi_ch = max(ch_h[i - 20 : i])
                lo_ch = min(ch_l[i - 20 : i])
                if ch_c[i] > hi_ch:
                    f1_chain[i] = 1
                elif ch_c[i] < lo_ch:
                    f1_chain[i] = -1

            c_i = dom_codes[i]
            bars_c = [by_cd.get((c_i, dates[k])) for k in range(i - 20, i + 1)]
            if all(b is not None for b in bars_c) and _gap_days(dates[i - 20], dates[i]) <= 60:
                # 21 consecutive trading days (~30 calendar days, allow long holidays)
                hi_sc = max(b["high"] for b in bars_c[:-1])  # type: ignore[index]
                lo_sc = min(b["low"] for b in bars_c[:-1])  # type: ignore[index]
                cl_i = bars_c[-1]["close"]  # type: ignore[index]
                if cl_i > hi_sc:
                    f1_same[i] = 1
                elif cl_i < lo_sc:
                    f1_same[i] = -1

        feats[prod] = {
            "dates": dates,
            "dom": dom_codes,
            "r_same": r_same,
            "rv": rv,
            "f1_same": f1_same,
            "f1_chain": f1_chain,
        }
    return feats


def build_clean_pos60_map(
    feats_by_prod: Mapping[str, Mapping[str, Any]],
    window: int = 60,
    min_levels: int = 22,
    min_products_per_day: int = 3,
) -> dict[str, float | None]:
    """Compute causal equal-weight commodity index pos60[date] from same-contract returns."""
    by_date_rets: dict[str, list[float]] = collections.defaultdict(list)
    all_dates: set[str] = set()
    for pf in feats_by_prod.values():
        for d, rs in zip(pf["dates"], pf["r_same"]):
            all_dates.add(d)
            if rs is not None:
                by_date_rets[d].append(math.expm1(rs))

    sorted_dates = sorted(all_dates)
    r_mkt: dict[str, float] = {
        d: sum(vals) / len(vals)
        for d, vals in by_date_rets.items()
        if len(vals) >= min_products_per_day
    }

    pos60: dict[str, float | None] = {}
    for k, d in enumerate(sorted_dates):
        if d not in r_mkt:
            pos60[d] = None
            continue
        ds = sorted_dates[max(0, k - window) : k + 1]
        lv = [1.0]
        valid_steps = 0
        for d_step in ds[1:]:
            mr = r_mkt.get(d_step)
            if mr is not None:
                valid_steps += 1
                lv.append(lv[-1] * (1.0 + mr))
            else:
                lv.append(lv[-1])
        if len(lv) < min_levels or valid_steps < (min_levels - 1):
            pos60[d] = None
            continue
        hi_lv, lo_lv = max(lv), min(lv)
        pos60[d] = (lv[-1] - lo_lv) / (hi_lv - lo_lv) if hi_lv > lo_lv else 0.5
    return pos60


def is_tw_blocked(pos60_val: float | None) -> tuple[bool, str | None]:
    """Return (blocked, reason) matching NO_TW (0.20 <= pos60 <= 0.80 blocked)."""
    if pos60_val is None:
        return True, "no_state"
    if 0.20 <= pos60_val <= 0.80:
        return True, "tw_blocked"
    return False, None


def pick_call_candidate(
    cands: Sequence[Mapping[str, Any]],
    cp: str = "C",
    lo_abs_d: float = 0.30,
    hi_abs_d: float = 0.60,
    target_abs_d: float = 0.45,
    underlying_code: str | None = None,
) -> Mapping[str, Any] | None:
    """Select the liquid option contract closest to target |delta|, tie-breaking by vol."""
    filtered = [
        c
        for c in cands
        if c["cp"] == cp
        and lo_abs_d <= abs(c["delta"]) <= hi_abs_d
        and (underlying_code is None or c.get("underlying") == underlying_code)
    ]
    if not filtered:
        return None
    return min(filtered, key=lambda c: (abs(abs(c["delta"]) - target_abs_d), -c["vol"], str(c["code"])))


def eval_option_leg(
    cand: Mapping[str, Any],
    dates: Sequence[str],
    i: int,
    h: int,
    px_map: Mapping[str, tuple[float, float, float]],
    require_fill_vol: bool = True,
    require_full_horizon: bool = False,
    max_exit_date: str | None = None,
) -> tuple[float, float, float, float, int, bool] | None:
    """Evaluate buying option leg at t+1 open and exiting at min(t+1+h, mat_m2) close.

    px_map maps date -> (open, close, vol).
    Returns (pnl1_rmb, pnl2_rmb, risk1_rmb, r1, held_bars, early_capped) or None.
    """
    if i + 1 >= len(dates):
        return None
    entry_d = dates[i + 1]
    if entry_d > cand["mat_m2"] or (max_exit_date is not None and entry_d > max_exit_date):
        return None
    eb = px_map.get(entry_d)
    if not eb:
        return None
    open_entry, _, vol_entry = eb
    if open_entry <= 0 or (require_fill_vol and vol_entry <= 0):
        return None

    target_idx = i + 1 + h
    early_capped = target_idx >= len(dates) or (
        max_exit_date is not None and dates[min(target_idx, len(dates) - 1)] > max_exit_date
    )
    if require_full_horizon and early_capped:
        return None
    max_idx = min(target_idx, len(dates) - 1)

    cap_exit_d = cand["mat_m2"] if max_exit_date is None else min(cand["mat_m2"], max_exit_date)
    exit_idx = None
    for idx in range(max_idx, i + 1, -1):
        bar = px_map.get(dates[idx])
        if dates[idx] <= cap_exit_d and bar is not None and bar[1] >= 0:
            if not require_fill_vol or bar[2] > 0:
                exit_idx = idx
                break
    if exit_idx is None:
        return None

    close_exit = px_map[dates[exit_idx]][1]
    tick = float(cand["tick"])
    mult = float(cand["mult"])
    fee_px = FEE_PER_LOT_SIDE_RMB / mult

    entry1 = open_entry + tick
    exit1 = max(0.0, close_exit - tick)
    risk1_px = entry1 + fee_px
    if risk1_px <= 0:
        return None
    pnl1_px = exit1 - entry1 - 2.0 * fee_px

    entry2 = open_entry + 2.0 * tick
    exit2 = max(0.0, close_exit - 2.0 * tick)
    pnl2_px = exit2 - entry2 - 4.0 * fee_px

    risk1_rmb = risk1_px * mult
    pnl1_rmb = pnl1_px * mult
    pnl2_rmb = pnl2_px * mult
    return pnl1_rmb, pnl2_rmb, risk1_rmb, pnl1_rmb / risk1_rmb, (exit_idx - (i + 1)), early_capped


def run_plateau_cell(
    h: int,
    iv_cap: float | None,
    fut_feats: Mapping[str, Mapping[str, Any]],
    pos60_map: Mapping[str, float | None],
    day_cands: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    px_by_code: Mapping[str, Mapping[str, tuple[float, float, float]]],
    lo: str,
    hi: str,
    f1_key: str = "f1_same",
    require_fill_vol: bool = True,
    match_underlying: bool = False,
    require_full_horizon: bool = False,
    max_exit_date: str | None = None,
) -> tuple[list[tuple[str, str, int, float]], list[tuple[str, str, int, float]], dict[str, Any]]:
    """Run one (h, iv_cap) cell under causal signal generation & non-overlapping hold."""
    ev: list[tuple[str, str, int, float]] = []
    ev2: list[tuple[str, str, int, float]] = []
    diag: collections.Counter[str] = collections.Counter()
    hold_days_list: list[int] = []

    for prod, pf in sorted(fut_feats.items()):
        dates = pf["dates"]
        f1 = pf[f1_key]
        rv = pf["rv"]
        dom = pf.get("dom")
        i = 21
        nxt = 0
        while i < len(dates) - 2:
            dt_s = dates[i]
            if i < nxt or not (lo <= dt_s <= hi):
                i += 1
                continue
            if f1[i] <= 0:
                i += 1
                continue
            blocked, reason = is_tw_blocked(pos60_map.get(dt_s))
            if blocked:
                diag[reason or "tw_blocked"] += 1
                i += 1
                continue
            cands = day_cands.get(dt_s, {}).get(prod)
            if not cands:
                diag["no_opt_cands"] += 1
                i += 1
                continue
            und_target = dom[i] if (match_underlying and dom is not None) else None
            cand = pick_call_candidate(cands, "C", 0.30, 0.60, 0.45, underlying_code=und_target)
            if cand is None:
                diag["no_contract"] += 1
                i += 1
                continue
            if iv_cap is not None and (rv[i] is None or rv[i] <= 0 or cand["iv"] > iv_cap * rv[i]):
                diag["iv_filtered"] += 1
                i += 1
                continue
            res = eval_option_leg(
                cand,
                dates,
                i,
                h,
                px_by_code.get(cand["code"], {}),
                require_fill_vol=require_fill_vol,
                require_full_horizon=require_full_horizon,
                max_exit_date=max_exit_date,
            )
            if res is None:
                diag["missing_entry_or_exit"] += 1
                i += 1
                continue
            _, p2, rk, r1, held_bars, early_capped = res
            ev.append((dt_s, prod, 1, r1))
            ev2.append((dt_s, prod, 1, p2 / rk))
            hold_days_list.append(held_bars)
            if dom is not None and cand.get("underlying") == dom[i]:
                diag["underlying_matched_dom"] += 1
            else:
                diag["underlying_diff_dom"] += 1
            if early_capped:
                diag["early_capped"] += 1
            if r1 >= 2.0:
                diag["tail_ge_2r"] += 1
            nxt = i + h + 1
            i += 1

    out_diag: dict[str, Any] = dict(diag)
    out_diag["mean_hold_bars"] = round(sum(hold_days_list) / len(hold_days_list), 2) if hold_days_list else None
    return ev, ev2, out_diag


def max_drawdown_r(ev: Sequence[tuple[str, str, int, float]]) -> float:
    if not ev:
        return 0.0
    cur = peak = mdd = 0.0
    for _, _, _, r in sorted(ev, key=lambda x: (x[0], x[1])):
        cur += r
        if cur > peak:
            peak = cur
        dd = peak - cur
        if dd > mdd:
            mdd = dd
    return mdd


def compute_cell_stats(
    ev: Sequence[tuple[str, str, int, float]],
    ev2: Sequence[tuple[str, str, int, float]],
    nperm: int = 2000,
    seed: int = 20261116,
    n_boot: int = 1000,
    min_events: int = 30,
) -> dict[str, Any] | None:
    """Compute monthly block-bootstrap CI, permutation p-value, drop-best-month, ex-top5, 2x cost."""
    n = len(ev)
    if n < min_events:
        return None
    rnd = random.Random(seed)
    rs = [e[3] for e in ev]
    mean_r = sum(rs) / n

    blk: dict[str, list[Any]] = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        b = blk[e[0][:6]]
        b[0] += e[3]
        b[1] += 1
    keys = sorted(blk)

    bs = []
    for _ in range(n_boot):
        pick = [keys[rnd.randrange(len(keys))] for _ in keys]
        sn = sum(blk[k][1] for k in pick)
        if sn > 0:
            bs.append(sum(blk[k][0] for k in pick) / sn)
    bs.sort()
    lo = bs[int(0.025 * len(bs))]

    null_samples = [
        sum(blk[k][0] * (1.0 if rnd.random() < 0.5 else -1.0) for k in keys) / n
        for _ in range(nperm)
    ]
    p_val = OS.perm_pvalue(mean_r, null_samples)

    yr_map: dict[str, list[float]] = collections.defaultdict(list)
    for e in ev:
        yr_map[e[0][:4]].append(e[3])
    ymean = {y: sum(v) / len(v) for y, v in sorted(yr_map.items())}
    pos_years = sum(1 for v in ymean.values() if v > 0)

    best_m = max(blk, key=lambda k: blk[k][0])
    drop_best_month = (sum(rs) - blk[best_m][0]) / (n - blk[best_m][1]) if n > blk[best_m][1] else None
    ex_top5 = OS.mean_ex_top(rs, 5)
    mean_2xcost = sum(e[3] for e in ev2) / len(ev2) if ev2 else None

    return {
        "n": n,
        "blocks": len(keys),
        "win_rate": round(sum(1 for x in rs if x > 0) / n, 4),
        "mean": mean_r,
        "lo": lo,
        "p": p_val,
        "years": ymean,
        "pos_years": pos_years,
        "drop_best_month": drop_best_month,
        "mean_ex_top5": ex_top5,
        "mean_2xcost": mean_2xcost,
        "max_dd_r": round(max_drawdown_r(ev), 2),
    }


def evaluate_plateau_grid(
    fut_feats: Mapping[str, Mapping[str, Any]],
    pos60_map: Mapping[str, float | None],
    day_cands: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    px_by_code: Mapping[str, Mapping[str, tuple[float, float, float]]],
    lo: str,
    hi: str,
    *,
    f1_key: str = "f1_same",
    require_fill_vol: bool = True,
    match_underlying: bool = False,
    require_full_horizon: bool = False,
    max_exit_date: str | None = None,
    nperm: int = 2000,
    seed: int = 20261116,
    min_events: int = 30,
) -> dict[str, Any]:
    """Run all 4x4 (h x iv_cap) grid cells and apply BH-FDR + 2D plateau evaluation."""
    grid_stats: dict[tuple[int, int], dict[str, Any] | None] = {}
    cell_meta: dict[tuple[int, int], dict[str, Any]] = {}
    diags: dict[str, dict[str, Any]] = {}
    raw_summaries: dict[str, dict[str, Any]] = {}

    for hi_i, h in enumerate(H_GRID):
        for ci_i, (_, cap) in enumerate(CAP_GRID):
            cid = cell_id(h, cap)
            ev, ev2, diag = run_plateau_cell(
                h,
                cap,
                fut_feats,
                pos60_map,
                day_cands,
                px_by_code,
                lo,
                hi,
                f1_key=f1_key,
                require_fill_vol=require_fill_vol,
                match_underlying=match_underlying,
                require_full_horizon=require_full_horizon,
                max_exit_date=max_exit_date,
            )
            diags[cid] = diag
            st = compute_cell_stats(ev, ev2, nperm=nperm, seed=seed, min_events=min_events)
            grid_stats[(hi_i, ci_i)] = st
            cell_meta[(hi_i, ci_i)] = {"cid": cid, "h": h, "iv_cap": cap, "raw_n": len(ev)}
            raw_summaries[cid] = {
                "n": len(ev),
                "mean": round(sum(x[3] for x in ev) / len(ev), 4) if ev else None,
                "win_rate": round(sum(1 for x in ev if x[3] > 0) / len(ev), 4) if ev else None,
                "tail_ge_2r": diag.get("tail_ge_2r", 0),
                "mean_hold_bars": diag.get("mean_hold_bars"),
            }

    valid_keys = [(hi_i, ci_i) for hi_i in range(len(H_GRID)) for ci_i in range(len(CAP_GRID)) if grid_stats[(hi_i, ci_i)] is not None]
    plist = [grid_stats[k]["p"] for k in valid_keys] + [1.0] * (len(H_GRID) * len(CAP_GRID) - len(valid_keys))  # type: ignore[index]
    bh_map = dict(zip(valid_keys, OS.bh_fdr(plist, q=0.05)[: len(valid_keys)]))
    plat_map = OS.eval_plateau_2d(grid_stats, nh=len(H_GRID), nc=len(CAP_GRID))

    rows: list[dict[str, Any]] = []
    for k in valid_keys:
        st = grid_stats[k]
        assert st is not None
        meta = cell_meta[k]
        pl = plat_map[k]
        yrs_cnt = len(st["years"])
        bh_ok = bool(bh_map.get(k, False))
        pass_single = bool(
            st["blocks"] >= 24
            and st["n"] >= 60
            and st["mean"] > 0
            and st["lo"] > 0
            and bh_ok
            and yrs_cnt >= 2
            and st["pos_years"] >= 0.6 * yrs_cnt
            and (st["drop_best_month"] or 0.0) > 0
            and (st["mean_2xcost"] or 0.0) > 0
        )
        pass_all = bool(pass_single and pl["ok"])
        rows.append(
            {
                "cfg": meta["cid"],
                "h": meta["h"],
                "iv_cap": meta["iv_cap"],
                "bh_pass": bh_ok,
                "plateau": pl["ok"],
                "plateau_nb": f"{pl['good_nb']}/{pl['valid_nb']}",
                "pos_nb": f"{pl['pos_nb']}/{pl['valid_nb']}",
                "pass_single": pass_single,
                "pass_all": pass_all,
                **st,
            }
        )

    rows_sorted = sorted(rows, key=lambda x: x["mean"], reverse=True)
    finalists = [r["cfg"] for r in rows_sorted if r["pass_all"]]
    single_pass = [r["cfg"] for r in rows_sorted if r["pass_single"]]
    return {
        "lo": lo,
        "hi": hi,
        "f1_key": f1_key,
        "require_fill_vol": require_fill_vol,
        "match_underlying": match_underlying,
        "max_exit_date": max_exit_date,
        "n_cells": len(H_GRID) * len(CAP_GRID),
        "valid_cells": len(valid_keys),
        "pos_cells": sum(1 for r in rows if r["mean"] > 0),
        "lo_pos_cells": sum(1 for r in rows if r["lo"] > 0),
        "single_pass": single_pass,
        "finalists": finalists,
        "rows": rows_sorted,
        "raw_summaries": raw_summaries,
        "diags": diags,
    }


def build_legacy_1d_fut_and_pos60(
    bars_by_product: Mapping[str, Sequence[tuple[str, float, float, float, float]]],
    min_len: int = 300,
) -> tuple[dict[str, dict[str, Any]], dict[str, float | None]]:
    """Legacy unadjusted 1d_fut features & pos60 baseline for comparison against clean_chain.

    bars_by_product maps prod -> sorted sequence of (date, open, high, low, close).
    """
    sqrt252 = math.sqrt(252.0)
    feats: dict[str, dict[str, Any]] = {}
    closes_by_pd: dict[str, dict[str, float]] = {}
    all_dates: set[str] = set()

    for prod, seq in sorted(bars_by_product.items()):
        s = sorted(seq, key=lambda x: x[0])
        if len(s) < min_len:
            continue
        n = len(s)
        dates = [x[0] for x in s]
        highs = [x[2] for x in s]
        lows = [x[3] for x in s]
        closes = [x[4] for x in s]
        rets: list[float | None] = [None] + [
            math.log(closes[k] / closes[k - 1]) if closes[k] > 0 and closes[k - 1] > 0 else 0.0
            for k in range(1, n)
        ]
        f1 = [0] * n
        rv: list[float | None] = [None] * n
        for i in range(20, n):
            hi20 = max(highs[i - 20 : i])
            lo20 = min(lows[i - 20 : i])
            if closes[i] > hi20:
                f1[i] = 1
            elif closes[i] < lo20:
                f1[i] = -1
            win_r = rets[i - 19 : i + 1]
            if all(x is not None for x in win_r):
                rv[i] = st_mod.pstdev(win_r) * sqrt252  # type: ignore[arg-type]
        feats[prod] = {"dates": dates, "f1_legacy": f1, "rv": rv}
        closes_by_pd[prod] = {d: c for d, c in zip(dates, closes)}
        all_dates.update(dates)

    sorted_dates = sorted(all_dates)
    prods = sorted(closes_by_pd)
    adj_ret = [0.0] * len(sorted_dates)
    for k in range(1, len(sorted_dates)):
        da, db = sorted_dates[k - 1], sorted_dates[k]
        rs = [
            closes_by_pd[p][db] / closes_by_pd[p][da] - 1.0
            for p in prods
            if da in closes_by_pd[p] and db in closes_by_pd[p] and closes_by_pd[p][da] > 0
        ]
        adj_ret[k] = sum(rs) / len(rs) if rs else 0.0

    pos60: dict[str, float | None] = {}
    for k, d in enumerate(sorted_dates):
        start_k = max(0, k - 60)
        if k - start_k + 1 < 22:
            pos60[d] = None
            continue
        lv = [1.0]
        for j in range(start_k + 1, k + 1):
            lv.append(lv[-1] * (1.0 + adj_ret[j]))
        hi_lv, lo_lv = max(lv), min(lv)
        pos60[d] = (lv[-1] - lo_lv) / (hi_lv - lo_lv) if hi_lv > lo_lv else 0.5
    return feats, pos60

