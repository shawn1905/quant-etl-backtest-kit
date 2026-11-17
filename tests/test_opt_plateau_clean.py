import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant_backtest import opt_plateau_clean as OPC


def _make_roll_jump_rows():
    """Construct 65 trading days where contract A rolls to higher-priced B on day 30.

    Each individual contract stays strictly flat (daily close=100 for A, 120 for B;
    intraday high=101/121, low=99/119). A naive unadjusted 1d_fut series would show
    a +20% jump on day 30, inflating RV_20 and triggering a fake F1_20=+1 breakout.
    """
    dates = [f"202301{d:02d}" if d <= 31 else (f"202302{d - 31:02d}" if d <= 59 else f"202303{d - 59:02d}") for d in range(1, 66)]
    rows_a_prod = []
    # Also construct two smooth market products so pos60 has >= 3 products
    rows_m1 = []
    rows_m2 = []
    for idx, d in enumerate(dates):
        # Product 'xx': A is dominant for idx < 30, B is dominant for idx >= 30
        oi_a = 10000 if idx < 30 else 3000
        oi_b = 4000 if idx < 30 else 12000
        rows_a_prod.append(
            {"date": d, "code": "SHFE.xx2305", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "oi": oi_a, "vol": 5000}
        )
        rows_a_prod.append(
            {"date": d, "code": "SHFE.xx2309", "open": 120.0, "high": 121.0, "low": 119.0, "close": 120.0, "oi": oi_b, "vol": 5000}
        )
        # Trending helper products for pos60
        px1 = 100.0 + idx * 0.5
        px2 = 200.0 + idx * 1.0
        rows_m1.append(
            {"date": d, "code": "DCE.m2305", "open": px1, "high": px1 + 1, "low": px1 - 1, "close": px1, "oi": 9000, "vol": 5000}
        )
        rows_m2.append(
            {"date": d, "code": "DCE.y2305", "open": px2, "high": px2 + 1, "low": px2 - 1, "close": px2, "oi": 9000, "vol": 5000}
        )
    return dates, {"xx": rows_a_prod, "m1": rows_m1, "m2": rows_m2}


class TestOptPlateauClean(unittest.TestCase):
    def test_roll_jump_does_not_create_fake_breakout_or_rv_spike(self):
        _, rows_by_prod = _make_roll_jump_rows()
        feats = OPC.build_clean_fut_features(rows_by_prod, min_len=30)
        xx = feats["xx"]
        # Same-contract log returns are strictly 0 everywhere (including roll day 30!)
        self.assertTrue(all(abs(r) < 1e-12 for r in xx["r_same"][1:]))
        self.assertEqual(xx["rv"][30], 0.0)
        self.assertEqual(xx["f1_same"][30], 0)
        self.assertEqual(xx["f1_chain"][30], 0)

        pos60 = OPC.build_clean_pos60_map(feats)
        # m1/m2 trend steadily upward; pos60 on late days approaches 1.0 (not TW blocked)
        self.assertIsNotNone(pos60[xx["dates"][-1]])
        self.assertGreater(pos60[xx["dates"][-1]], 0.80)
        blocked, reason = OPC.is_tw_blocked(pos60[xx["dates"][-1]])
        self.assertFalse(blocked)
        self.assertIsNone(reason)

        # Contrast: legacy unadjusted 1d_fut series sees a fake +20% breakout & inflated RV on day 30
        unadj_xx = [
            (
                d,
                100.0 if i < 30 else 120.0,
                101.0 if i < 30 else 121.0,
                99.0 if i < 30 else 119.0,
                100.0 if i < 30 else 120.0,
            )
            for i, d in enumerate(xx["dates"])
        ]
        leg_feats, _ = OPC.build_legacy_1d_fut_and_pos60({"xx": unadj_xx}, min_len=30)
        self.assertEqual(leg_feats["xx"]["f1_legacy"][30], 1)
        self.assertGreater(leg_feats["xx"]["rv"][30], 0.5)

    def test_filters_invalid_zero_close_bars(self):
        dates = [f"202609{d:02d}" for d in range(1, 31)]
        rows = []
        for idx, d in enumerate(dates):
            # Untraded contract with higher OI but close=0 must NOT win dominant_by_day
            rows.append({"date": d, "code": "CZCE.JR609", "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "oi": 50000, "vol": 0})
            c = 3000.0 + idx * 10.0
            rows.append({"date": d, "code": "CZCE.JR701", "open": c, "high": c + 2.0, "low": c - 2.0, "close": c, "oi": 1000, "vol": 200})
        feats = OPC.build_clean_fut_features({"jr": rows}, min_len=25)
        self.assertEqual(feats["jr"]["dom"][-1], "CZCE.JR701")
        self.assertGreater(feats["jr"]["f1_same"][-1], 0)

    def test_eval_option_leg_arithmetic_and_vol_gate(self):
        dates = [f"202301{d:02d}" for d in range(1, 25)]
        cand = {
            "code": "SHFE.rb2305C4000",
            "cp": "C",
            "iv": 0.20,
            "delta": 0.45,
            "dte": 40,
            "mat_m2": "20230210",
            "mult": 10.0,
            "tick": 1.0,
            "vol": 100.0,
        }
        px_map = {
            dates[1]: (10.0, 11.0, 50.0),  # entry open=10
            dates[11]: (25.0, 30.0, 80.0),  # exit close=30 at h=10
        }
        res = OPC.eval_option_leg(cand, dates, 0, 10, px_map, require_fill_vol=True)
        self.assertIsNotNone(res)
        pnl1_rmb, pnl2_rmb, risk1_rmb, r1, held, early = res
        # entry1=11, exit1=29, fee_px=0.5 -> pnl1_px=17, risk1_px=11.5 -> r1=17/11.5
        self.assertAlmostEqual(risk1_rmb, 115.0)
        self.assertAlmostEqual(pnl1_rmb, 170.0)
        self.assertAlmostEqual(r1, 17.0 / 11.5)
        self.assertEqual(held, 10)
        self.assertFalse(early)

        # Entry vol == 0 rejected when require_fill_vol=True, accepted when False
        px_zero_vol = {
            dates[1]: (10.0, 11.0, 0.0),
            dates[11]: (25.0, 30.0, 80.0),
        }
        self.assertIsNone(OPC.eval_option_leg(cand, dates, 0, 10, px_zero_vol, require_fill_vol=True))
        self.assertIsNotNone(OPC.eval_option_leg(cand, dates, 0, 10, px_zero_vol, require_fill_vol=False))

    def test_evaluate_plateau_grid_runs_end_to_end_synthetic(self):
        dates = []
        y, m, d = 2020, 1, 1
        while len(dates) < 120:
            dates.append(f"{y:04d}{m:02d}{d:02d}")
            d += 1
            if d > 28:
                d = 1
                m += 1
                if m > 12:
                    m = 1
                    y += 1

        rows_by_prod = {"aa": [], "bb": [], "cc": []}
        day_cands = {}
        px_by_code = {"SHFE.aa2009C100": {}}
        for idx, dt_s in enumerate(dates):
            px_a = 100.0 + idx * 1.5
            px_b = 200.0 + idx * 1.2
            px_c = 150.0 + idx * 1.0
            rows_by_prod["aa"].append({"date": dt_s, "code": "SHFE.aa2009", "open": px_a, "high": px_a + 0.2, "low": px_a - 0.2, "close": px_a, "oi": 5000, "vol": 1000})
            rows_by_prod["bb"].append({"date": dt_s, "code": "SHFE.bb2009", "open": px_b, "high": px_b + 0.2, "low": px_b - 0.2, "close": px_b, "oi": 5000, "vol": 1000})
            rows_by_prod["cc"].append({"date": dt_s, "code": "SHFE.cc2009", "open": px_c, "high": px_c + 0.2, "low": px_c - 0.2, "close": px_c, "oi": 5000, "vol": 1000})
            px_by_code["SHFE.aa2009C100"][dt_s] = (10.0 + idx * 0.4, 10.5 + idx * 0.4, 100.0)
            day_cands.setdefault(dt_s, {})["aa"] = [
                {
                    "code": "SHFE.aa2009C100",
                    "cp": "C",
                    "iv": 0.02,
                    "delta": 0.45,
                    "dte": 45,
                    "mat_m2": dates[-1],
                    "mult": 10.0,
                    "tick": 0.5,
                    "vol": 200.0,
                    "underlying": "SHFE.aa2009",
                }
            ]

        feats = OPC.build_clean_fut_features(rows_by_prod, min_len=30)
        pos60 = OPC.build_clean_pos60_map(feats, min_products_per_day=3)
        res = OPC.evaluate_plateau_grid(
            feats,
            pos60,
            day_cands,
            px_by_code,
            dates[25],
            dates[-35],
            min_events=1,
        )
        self.assertEqual(res["n_cells"], 16)
        self.assertGreater(res["valid_cells"], 0)


if __name__ == "__main__":
    unittest.main()
