import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant_backtest import family_pipeline as FP


def _make_dates(n: int = 40) -> list[str]:
    out = []
    y, m, d = 2024, 1, 1
    while len(out) < n:
        out.append(f"{y:04d}{m:02d}{d:02d}")
        d += 1
        if d > 28:
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    return out


class TestFamilyPipeline(unittest.TestCase):
    def test_post_liq_low_ladder_end_to_end(self):
        # Simulate "爆仓后 + 低位 + 波动率门控 + 分批挂单低吸" as a single StrategyRecipe under mean_reversion
        dates = _make_dates(30)
        candidates = []
        sessions = {}
        for i, dt_s in enumerate(dates[:20]):
            liq_z = 3.0 if i < 12 else 1.0          # first 12 trigger liq spike
            pct_20 = 0.15 if i < 8 else 0.60         # only first 8 are in low 20% percentile
            vol_ok = True if i < 6 else False        # only first 6 pass env brake
            candidates.append(
                {
                    "date": dt_s,
                    "day_idx": i,
                    "asset": "BTCUSDT",
                    "liq_z": liq_z,
                    "pct_20": pct_20,
                    "vol_ok": vol_ok,
                }
            )
            # OHLC session starts at 100.0, dips to 96.5 (filling levels 1..3), then bounces to 100.5
            sessions[dt_s] = {
                "prior_range": 0.03,
                "ohlc": [
                    (100.0, 100.2, 96.5, 97.5),  # down bar? wait: c=97.5 < o=100 -> up then down, fills on down
                    (97.5, 100.8, 97.4, 100.5),  # bounce up hits TP & closes higher!
                ],
            }

        recipe = FP.StrategyRecipe(
            recipe_id="mr_liq_low_ladder_v1",
            family="mean_reversion",
            simulator_kind="ladder_grid",
            gates=(
                FP.GateSpec("liq_spike", "trigger", lambda c: (c["liq_z"] >= 2.5, 1)),
                FP.GateSpec("low_price", "location", lambda c: (c["pct_20"] <= 0.20, 0)),
                FP.GateSpec("vol_regime", "env_brake", lambda c: (bool(c["vol_ok"]), 0)),
            ),
            simulator_params={
                "levels": 6,
                "range_multiplier": 0.30,
                "min_step": 0.005,
                "unit_notional": 1.0,
                "maker_fee": 0.0002,
                "taker_fee": 0.0005,
                "spread": 0.0005,
                "penetration": 0.0005,
                "take_profit_steps": 1,
            },
            cooldown_bars=0,
            description="Post-liquidation low-percentile 6-level passive buy ladder",
        )

        res = FP.run_recipe(
            recipe,
            candidates,
            lambda cand, _p: sessions[cand["date"]],
            nperm=100,
            min_events=1,
        )
        self.assertEqual(res["family"], "mean_reversion")
        self.assertEqual(res["funnel"]["total_candidates"], 20)
        self.assertEqual(res["funnel"]["blocked:liq_spike"], 8)
        self.assertEqual(res["funnel"]["blocked:low_price"], 4)
        self.assertEqual(res["funnel"]["blocked:vol_regime"], 2)
        self.assertEqual(res["funnel"]["admitted"], 6)
        self.assertEqual(res["basic"]["n"], 6)
        self.assertGreater(res["basic"]["mean_net_1x"], 0.0)
        self.assertGreater(res["basic"]["mean_net_1x"], res["basic"]["mean_net_2x"])
        self.assertGreaterEqual(res["trades"][0]["meta"]["filled_levels"], 3)

    def test_variant_diff_counterexample_and_valid_filter(self):
        candidates = [
            {"date": d, "day_idx": i, "asset": "SHFE.rb", "trig": True, "low_pos": i % 2 == 0}
            for i, d in enumerate(_make_dates(12))
        ]
        bar_fn = lambda _c, _p: {"entry": 100.0, "exit": 102.0, "fee_ret": 0.0002, "slip_ret": 0.0004}

        base_recipe = FP.StrategyRecipe(
            recipe_id="trend_base_v1",
            family="trend",
            simulator_kind="bar_order",
            gates=(FP.GateSpec("sig", "trigger", lambda c: (c["trig"], 1)),),
        )
        noop_recipe = FP.StrategyRecipe(
            recipe_id="trend_noop_v1",
            family="trend",
            simulator_kind="bar_order",
            gates=(
                FP.GateSpec("sig", "trigger", lambda c: (c["trig"], 1)),
                FP.GateSpec("noop", "env_brake", lambda _c: (True, 0)),
            ),
        )
        filtered_recipe = FP.StrategyRecipe(
            recipe_id="trend_lowpos_v1",
            family="trend",
            simulator_kind="bar_order",
            gates=(
                FP.GateSpec("sig", "trigger", lambda c: (c["trig"], 1)),
                FP.GateSpec("low_pos", "location", lambda c: (c["low_pos"], 0)),
            ),
        )

        r_base = FP.run_recipe(base_recipe, candidates, bar_fn, min_events=1, nperm=50)
        r_noop = FP.run_recipe(noop_recipe, candidates, bar_fn, min_events=1, nperm=50)
        r_filt = FP.run_recipe(filtered_recipe, candidates, bar_fn, min_events=1, nperm=50)

        with self.assertRaises(ValueError):
            FP.prove_variant_diff(r_base, r_noop)

        diff = FP.prove_variant_diff(r_base, r_filt)
        self.assertTrue(diff["funnel_changed"])
        self.assertTrue(diff["trades_changed"])
        self.assertEqual(diff["base_admitted"], 12)
        self.assertEqual(diff["variant_admitted"], 6)

    def test_sealed_family_and_tokens_rejected(self):
        trig = (FP.GateSpec("t", "trigger", lambda _c: (True, 1)),)
        with self.assertRaises(ValueError):
            FP.validate_recipe(FP.StrategyRecipe("xs_carry_retry", "carry", "bar_order", trig))
        with self.assertRaises(ValueError):
            FP.validate_recipe(FP.StrategyRecipe("my_mom60_variant", "trend", "bar_order", trig))
        with self.assertRaises(ValueError):
            FP.validate_recipe(FP.StrategyRecipe("b1_b5_retry", "trend", "bar_order", trig))

    def test_cross_family_portfolio_combo(self):
        dates = _make_dates(60)
        # Construct 3 anti-correlated orthogonal legs across trend / mean_reversion / volatility
        r_trend = FP.run_recipe(
            FP.StrategyRecipe("leg_trend", "trend", "bar_order", (FP.GateSpec("t", "trigger", lambda c: (c["on"], 1)),)),
            [{"date": d, "day_idx": i, "asset": "rb", "on": True} for i, d in enumerate(dates)],
            lambda c, _p: {"entry": 100.0, "exit": 103.0 if c["day_idx"] % 2 == 0 else 99.0, "fee_ret": 0.0, "slip_ret": 0.0},
            min_events=1,
            nperm=50,
        )
        r_mr = FP.run_recipe(
            FP.StrategyRecipe(
                "leg_mr",
                "mean_reversion",
                "ladder_grid",
                (FP.GateSpec("m", "trigger", lambda c: (c["on"], 1)),),
                simulator_params={"levels": 2, "min_step": 0.01, "penetration": 0.0, "maker_fee": 0.0, "taker_fee": 0.0, "spread": 0.0},
            ),
            [{"date": d, "day_idx": i, "asset": "sc", "on": True} for i, d in enumerate(dates)],
            lambda c, _p: {
                "prior_range": 0.01,
                "ohlc": [(100.0, 102.0, 98.0, 102.0 if c["day_idx"] % 2 == 1 else 99.0)],
            },
            min_events=1,
            nperm=50,
        )
        opt_px = {}
        for i, d in enumerate(dates):
            # option pops on alternating cycles
            opt_px[d] = (10.0, 14.0 if i >= 5 and i % 3 == 0 else 10.5, 50.0)
        opt_cand = {"code": "OPT.aaC", "cp": "C", "iv": 0.2, "delta": 0.45, "dte": 40, "mat_m2": dates[-1], "mult": 10.0, "tick": 0.1, "vol": 100.0}
        r_vol = FP.run_recipe(
            FP.StrategyRecipe("leg_vol", "volatility", "option_leg", (FP.GateSpec("v", "trigger", lambda c: (c["day_idx"] < 50, 1)),), cooldown_bars=2),
            [{"date": d, "day_idx": i, "asset": "aa"} for i, d in enumerate(dates)],
            lambda c, _p: {"cand": opt_cand, "dates": dates, "sig_idx": c["day_idx"], "h": 3, "px_map": opt_px},
            min_events=1,
            nperm=50,
        )

        combo = FP.evaluate_portfolio_combo("combo_3fam", [r_trend, r_mr, r_vol], calendar_dates=dates, n_boot=200)
        self.assertEqual(len(combo["members"]), 3)
        self.assertIn("leg_trend", combo["daily_corr"])
        self.assertIn("leg_mr", combo["vs_legs"])
        self.assertGreater(combo["calendar_book_combo"]["days"], 0)
        self.assertLess(combo["daily_corr"]["leg_trend"]["leg_mr"], 0.2)

    def test_ladder_grid_risk_normalization_and_session_fee_override(self):
        admitted = [
            {"date": "20240102", "asset": "rb", "direction": 1},
            {"date": "20240103", "asset": "rb", "direction": 1},
            {"date": "20240104", "asset": "rb", "direction": 1},
        ]
        sess_base = {
            "prior_range": 0.02,
            "ohlc": [(100.0, 100.1, 97.5, 98.5), (98.5, 101.0, 98.4, 100.8)],
            "maker_fee": 0.0001,
            "taker_fee": 0.0001,
            "spread": 0.0002,
            "min_step": 0.005,
        }

        def lookup_norm(cand, _p):
            if cand["date"] == "20240102":
                return {**sess_base, "risk_unit": 0.02}
            if cand["date"] == "20240103":
                return {**sess_base, "risk_unit": 0.0}
            return {**sess_base}  # missing risk_unit

        params_raw = {"levels": 4, "range_multiplier": 0.35, "take_profit_steps": 2, "penetration": 0.0005}
        trades_raw, unv_raw = FP.simulate_ladder_grid([admitted[0]], lambda _c, _p: sess_base, params_raw)
        trades_r, unv_r = FP.simulate_ladder_grid(
            admitted, lookup_norm, {**params_raw, "normalize_by_risk": True}
        )
        self.assertEqual(unv_raw, {})
        self.assertEqual(len(trades_raw), 1)
        self.assertEqual(unv_r, {"bad_risk_unit": 2})
        self.assertEqual(len(trades_r), 1)
        self.assertAlmostEqual(trades_r[0]["net_1x"], trades_raw[0]["net_1x"] / 0.02, places=10)
        self.assertAlmostEqual(trades_r[0]["gross"], trades_raw[0]["gross"] / 0.02, places=10)
        self.assertAlmostEqual(trades_r[0]["net_2x"], trades_raw[0]["net_2x"] / 0.02, places=10)
        self.assertEqual(trades_r[0]["meta"]["risk_unit"], 0.02)
        # Prove session-level fee override actually lowers fee drag vs default high fee
        sess_hi = {**sess_base, "maker_fee": 0.005, "taker_fee": 0.005, "spread": 0.005}
        trades_hi, _ = FP.simulate_ladder_grid([admitted[0]], lambda _c, _p: sess_hi, params_raw)
        self.assertGreater(trades_raw[0]["net_1x"], trades_hi[0]["net_1x"])


if __name__ == "__main__":
    unittest.main()
