"""Synthetic unit tests for opt_credit_spread (no NAS/network access)."""

from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quant_backtest import opt_credit_spread as OCS  # noqa: E402
from quant_backtest import opt_plateau_clean as OPC  # noqa: E402
from quant_backtest import family_pipeline as FP  # noqa: E402


def _make_trading_dates(n: int = 80, start: str = "20240102") -> list[str]:
    d = dt.datetime.strptime(start, "%Y%m%d")
    out: list[str] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


class TestOptCreditSpread(unittest.TestCase):
    def test_pick_spread_pair_strike_order_and_liquidity(self) -> None:
        cands = [
            # Put candidates (short K=4000 delta=-0.30, prot K=3800 delta=-0.12)
            {
                "code": "P4000", "cp": "P", "underlying": "CU2405", "strike": 4000.0,
                "close": 120.0, "iv": 0.25, "delta": -0.30, "dte": 40, "mat": "20240415",
                "mat_m2": "20240413", "tick": 2.0, "mult": 5.0, "vol": 200.0, "oi": 500.0,
            },
            {
                "code": "P3800", "cp": "P", "underlying": "CU2405", "strike": 3800.0,
                "close": 40.0, "iv": 0.24, "delta": -0.12, "dte": 40, "mat": "20240415",
                "mat_m2": "20240413", "tick": 2.0, "mult": 5.0, "vol": 150.0, "oi": 400.0,
            },
            # Higher-strike put cannot serve as protective wing for a Put credit spread
            {
                "code": "P4200", "cp": "P", "underlying": "CU2405", "strike": 4200.0,
                "close": 10.0, "iv": 0.22, "delta": -0.10, "dte": 40, "mat": "20240415",
                "mat_m2": "20240413", "tick": 2.0, "mult": 5.0, "vol": 300.0, "oi": 600.0,
            },
        ]
        picked = OCS.pick_credit_spread_pair(cands, cp="P", underlying_code="CU2405")
        self.assertIsNotNone(picked)
        sc, lc, cred, width = picked  # type: ignore[misc]
        self.assertEqual(sc["code"], "P4000")
        self.assertEqual(lc["code"], "P3800")
        self.assertEqual(cred, 80.0)
        self.assertEqual(width, 200.0)

        # Call spread requires strike_long > strike_short
        call_bad_order = [
            {
                "code": "C4000", "cp": "C", "underlying": "CU2405", "strike": 4000.0,
                "close": 120.0, "iv": 0.25, "delta": 0.30, "dte": 40, "mat": "20240415",
                "mat_m2": "20240413", "tick": 2.0, "mult": 5.0, "vol": 200.0, "oi": 500.0,
            },
            {
                "code": "C3900", "cp": "C", "underlying": "CU2405", "strike": 3900.0,
                "close": 40.0, "iv": 0.24, "delta": 0.12, "dte": 40, "mat": "20240415",
                "mat_m2": "20240413", "tick": 2.0, "mult": 5.0, "vol": 200.0, "oi": 500.0,
            },
        ]
        self.assertIsNone(OCS.pick_credit_spread_pair(call_bad_order, cp="C", underlying_code="CU2405"))

    def test_eval_credit_spread_cost_tiers_vol0_and_bad_quote(self) -> None:
        dates = _make_trading_dates(30, "20240102")
        sc = {"code": "P4000", "tick": 1.0, "mult": 10.0, "mat_m2": "20240401"}
        lc = {"code": "P3800", "tick": 1.0, "mult": 10.0, "mat_m2": "20240401"}
        px_map = {
            "P4000": {
                dates[1]: (50.0, 48.0, 100.0),
                dates[6]: (20.0, 10.0, 80.0),
            },
            "P3800": {
                dates[1]: (15.0, 14.0, 90.0),
                dates[6]: (5.0, 2.0, 60.0),
            },
        }
        res = OCS.eval_credit_spread(sc, lc, 200.0, dates, i=0, h=5, px_map=px_map)
        self.assertIsNotNone(res)
        assert res is not None
        # 1x: ent = (50-1)-(15+1)=33; ext = (10+1)-(2-1)=10; fee = 4*(5/10)=2.0; pnl1 = 33-10-2=21; risk1 = (200-33)+2=169
        self.assertAlmostEqual(res["pnl1_rmb"], 210.0, places=6)
        self.assertAlmostEqual(res["risk1_rmb"], 1690.0, places=6)
        self.assertAlmostEqual(res["r1"], 21.0 / 169.0, places=6)
        # 6t: 1.5 tick -> ent = 48.5-16.5=32; ext = 11.5-0.5=11; fee=2 -> pnl6t=19
        self.assertAlmostEqual(res["r_6t"], 19.0 / 169.0, places=6)
        # 2x: 2 tick -> ent = 48-17=31; ext = 12-0=12; fee=4 -> pnl2=15
        self.assertAlmostEqual(res["r2"], 15.0 / 169.0, places=6)
        self.assertTrue(res["r1"] > res["r_6t"] > res["r2"])

        # Zero-volume entry must fail
        px_zero_vol = {
            "P4000": {dates[1]: (50.0, 48.0, 0.0), dates[6]: (20.0, 10.0, 80.0)},
            "P3800": px_map["P3800"],
        }
        self.assertIsNone(OCS.eval_credit_spread(sc, lc, 200.0, dates, i=0, h=5, px_map=px_zero_vol))

        # Bad/stale quote where 1x credit >= spread width must fail safely
        px_bad_width = {
            "P4000": {dates[1]: (250.0, 240.0, 100.0), dates[6]: (10.0, 10.0, 80.0)},
            "P3800": px_map["P3800"],
        }
        self.assertIsNone(OCS.eval_credit_spread(sc, lc, 200.0, dates, i=0, h=5, px_map=px_bad_width))

    def test_truncation_invariance_and_variant_differentiation(self) -> None:
        dates = _make_trading_dates(120, "20240102")
        # Build two synthetic products so pos60 and suite summary work cleanly
        rows_by_prod = {"cu": [], "al": [], "rb": []}
        px_cu = 1000.0
        for idx, d in enumerate(dates):
            # Realistic oscillating uptrend in cu (non-trivial RV20), sideways in al, rb
            px_cu *= 1.008 if idx % 2 == 0 else 0.996
            px_al = 1000.0 * (1.001 ** idx if idx % 3 != 0 else 0.998)
            px_rb = 1000.0 * (1.0005 ** idx)
            for prod, px in (("cu", px_cu), ("al", px_al), ("rb", px_rb)):
                rows_by_prod[prod].append(
                    {"date": d, "code": f"{prod.upper()}2406", "open": px, "high": px * 1.003, "low": px * 0.997, "close": px, "oi": 10000.0, "vol": 5000.0}
                )
        fut_feats = OPC.build_clean_fut_features(rows_by_prod, min_len=30)
        pos60 = OPC.build_clean_pos60_map(fut_feats, min_levels=10, min_products_per_day=2)

        day_cands: dict[str, dict[str, list[dict]]] = {}
        px_by_code: dict[str, dict[str, tuple[float, float, float]]] = {"P_S": {}, "P_L": {}, "P_ATM": {}}
        for idx, d in enumerate(dates):
            mat_dt = dt.datetime.strptime(dates[-1], "%Y%m%d") + dt.timedelta(days=25)
            mat_s = mat_dt.strftime("%Y%m%d")
            mat_m2 = (mat_dt - dt.timedelta(days=2)).strftime("%Y%m%d")
            # Short leg decays smoothly from 80 -> 30, long wing from 25 -> 10
            p_s = max(20.0, 80.0 - 0.4 * idx)
            p_l = max(5.0, 25.0 - 0.1 * idx)
            px_by_code["P_S"][d] = (p_s, p_s, 150.0)
            px_by_code["P_L"][d] = (p_l, p_l, 120.0)
            px_by_code["P_ATM"][d] = (100.0, 100.0, 200.0)
            day_cands[d] = {
                "cu": [
                    {
                        "code": "P_ATM", "cp": "C", "underlying": "CU2406", "strike": 1050.0,
                        "close": 100.0, "iv": 0.40, "delta": 0.48, "dte": 40, "mat": mat_s,
                        "mat_m2": mat_m2, "tick": 1.0, "mult": 10.0, "vol": 200.0, "oi": 500.0,
                    },
                    {
                        "code": "P_S", "cp": "P", "underlying": "CU2406", "strike": 1000.0,
                        "close": p_s, "iv": 0.42, "delta": -0.30, "dte": 40, "mat": mat_s,
                        "mat_m2": mat_m2, "tick": 1.0, "mult": 10.0, "vol": 150.0, "oi": 400.0,
                    },
                    {
                        "code": "P_L", "cp": "P", "underlying": "CU2406", "strike": 800.0,
                        "close": p_l, "iv": 0.39, "delta": -0.12, "dte": 40, "mat": mat_s,
                        "mat_m2": mat_m2, "tick": 1.0, "mult": 10.0, "vol": 120.0, "oi": 350.0,
                    },
                ]
            }

        # Truncation invariance on early signal dates
        cutoff = dates[60]
        full_cs1 = OCS.run_credit_spread_config(
            "CS1", fut_feats, pos60, day_cands, px_by_code, dates[22], cutoff, h=5, max_exit_date=cutoff
        )
        trunc_feats = OPC.build_clean_fut_features(
            {p: [r for r in rs if r["date"] <= cutoff] for p, rs in rows_by_prod.items()},
            min_len=30,
        )
        trunc_pos60 = OPC.build_clean_pos60_map(trunc_feats, min_levels=10, min_products_per_day=2)
        trunc_cs1 = OCS.run_credit_spread_config(
            "CS1", trunc_feats, trunc_pos60, {d: v for d, v in day_cands.items() if d <= cutoff},
            px_by_code, dates[22], cutoff, h=5, max_exit_date=cutoff
        )
        self.assertGreater(len(full_cs1["trades"]), 0)
        self.assertEqual(full_cs1["trades"], trunc_cs1["trades"])

        # Strict variant (higher iv_rv_min / credit_ticks) produces distinct funnel/trades
        strict_cs3 = OCS.run_credit_spread_config(
            "CS3", fut_feats, pos60, day_cands, px_by_code, dates[22], dates[-1],
            h=5, iv_rv_min=99.0, min_credit_ticks=8.0,
        )
        self.assertEqual(len(strict_cs3["trades"]), 0)
        self.assertGreater(strict_cs3["diag"]["ivrv_below_thresh"], 0)

        # Portfolio leg conversion plugs cleanly into evaluate_portfolio_combo
        leg1 = OCS.run_to_portfolio_leg(
            OCS.run_credit_spread_config("CS1", fut_feats, pos60, day_cands, px_by_code, dates[22], dates[-1], h=5)
        )
        leg2 = OCS.run_to_portfolio_leg(
            OCS.run_credit_spread_config("CS2", fut_feats, pos60, day_cands, px_by_code, dates[22], dates[-1], h=5, use_mkt_align=True)
        )
        combo = FP.evaluate_portfolio_combo("TestCombo", [leg1, leg2], calendar_dates=dates[22:], n_boot=50)
        self.assertIn("calendar_book_combo", combo)

        summary = OCS.summarize_credit_spread_suite(
            [
                OCS.run_credit_spread_config("CS1", fut_feats, pos60, day_cands, px_by_code, dates[22], dates[-1], h=5),
                strict_cs3,
            ],
            nperm=50,
        )
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[1]["n"], 0)


if __name__ == "__main__":
    unittest.main()
