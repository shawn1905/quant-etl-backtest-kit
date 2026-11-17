"""Minimal offline synthetic demo: multi-gate StrategyRecipe + ladder grid + combo check."""

from __future__ import annotations

import datetime as dt
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from quant_backtest import family_pipeline as FP  # noqa: E402


def _trading_days(n: int = 160, start: str = "20250102") -> list[str]:
    d = dt.datetime.strptime(start, "%Y%m%d")
    out: list[str] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


def main() -> None:
    days = _trading_days(160)
    # Synthetic multi-asset bar lookup (100% fake toy series)
    bars: dict[str, dict[str, dict[str, float]]] = {"cu": {}, "al": {}}
    cands: list[dict] = []
    for idx, d in enumerate(days):
        for prod, bias in (("cu", 1.0008), ("al", 1.0006)):
            px = 1000.0 * (bias ** idx) * (1.015 if idx % 11 == 0 else 0.997 if idx % 5 == 0 else 1.0)
            bars[prod][d] = {"open": px, "high": px * 1.020, "low": px * 0.975, "close": px}
            cands.append(
                {
                    "date": d,
                    "day_idx": idx,
                    "asset": prod,
                    "mom_sig": 1 if (idx >= 5 and idx % 9 == 0) else 0,
                    "mr_drop": -0.025 if (idx >= 5 and idx % 7 == 0) else 0.0,
                    "env_ok": True,
                }
            )

    def lookup_bar(cand: dict, params: dict) -> dict:
        i = cand["day_idx"]
        h = int(params.get("h", 5))
        if i + 1 + h >= len(days):
            return {"unvalued_reason": "beyond_end"}
        b_in = bars[cand["asset"]][days[i + 1]]
        b_ex = bars[cand["asset"]][days[i + 1 + h]]
        return {"entry": b_in["open"], "exit": b_ex["close"], "fee_ret": 0.0002, "slip_ret": 0.0002, "risk_unit": 0.015}

    def lookup_ladder(cand: dict, params: dict) -> dict:
        i = cand["day_idx"]
        h = int(params.get("h", 5))
        if i + 1 + h >= len(days):
            return {"unvalued_reason": "beyond_end"}
        win = [bars[cand["asset"]][days[k]] for k in range(i + 1, i + 1 + h)]
        return {
            "ref_price": bars[cand["asset"]][cand["date"]]["close"],
            "bars": win,
            "levels": int(params["levels"]),
            "range_multiplier": float(params["range_multiplier"]),
            "min_step": float(params["min_step"]),
            "take_profit_steps": int(params["take_profit_steps"]),
            "fee_ret": 0.0002,
            "slip_ret": 0.0002,
            "risk_unit": 0.015,
            "normalize_by_risk": True,
        }

    r_trend = FP.StrategyRecipe(
        recipe_id="demo_trend_bar",
        family="trend",
        simulator_kind="bar_order",
        gates=(
            FP.GateSpec("mom_trigger", "trigger", lambda c: (c["mom_sig"] != 0, int(c["mom_sig"]))),
            FP.GateSpec("env_ok", "env_brake", lambda c: (bool(c["env_ok"]), 0)),
        ),
        simulator_params={"h": 5, "normalize_by_risk": True},
        cooldown_bars=5,
    )
    r_mr = FP.StrategyRecipe(
        recipe_id="demo_mr_ladder",
        family="mean_reversion",
        simulator_kind="ladder_grid",
        gates=(
            FP.GateSpec("drop_trigger", "trigger", lambda c: (c["mr_drop"] <= -0.02, 1)),
            FP.GateSpec("env_ok", "env_brake", lambda c: (bool(c["env_ok"]), 0)),
        ),
        simulator_params={"h": 5, "levels": 4, "range_multiplier": 0.3, "min_step": 0.005, "take_profit_steps": 1},
        cooldown_bars=5,
    )

    res_t = FP.run_recipe(r_trend, cands, lookup_bar, nperm=100, min_events=5)
    res_mr = FP.run_recipe(r_mr, cands, lookup_ladder, nperm=100, min_events=5)
    combo = FP.evaluate_portfolio_combo("DemoCombo", [res_t, res_mr], calendar_dates=days, n_boot=100)
    print("Trend basic:", res_t["basic"])
    print("MR ladder basic:", res_mr["basic"])
    print("Combo book metrics:", combo["calendar_book_combo"])


if __name__ == "__main__":
    main()
