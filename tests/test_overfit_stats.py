import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant_backtest import overfit_stats as OS


class TestOverfitStats(unittest.TestCase):
    def test_config_hash_stable_and_order_independent(self):
        self.assertEqual(OS.config_hash({"a": 1, "b": 2}), OS.config_hash({"b": 2, "a": 1}))
        self.assertNotEqual(OS.config_hash({"a": 1}), OS.config_hash({"a": 2}))
        self.assertEqual(len(OS.config_hash({"k": "参数"})), 12)

    def test_register_trials_dedup_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "nested", "trials.jsonl")
            self.assertEqual(OS.register_trials(p, [{"x": 1}, {"x": 2}], "b1", now="T0"), (2, 2))
            self.assertEqual(OS.register_trials(p, [{"x": 2}, {"x": 3}], "b2", now="T1"), (1, 3))
            with open(p, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 3)

    def test_mean_ex_top(self):
        self.assertIsNone(OS.mean_ex_top([1.0, 2.0, 3.0], 5))
        self.assertIsNone(OS.mean_ex_top([1.0, 2.0, 3.0], 3))
        self.assertEqual(OS.mean_ex_top([10.0, 1.0, 1.0, 1.0], 1), 1.0)

    def test_bh_fdr(self):
        self.assertEqual(OS.bh_fdr([], 0.05), [])
        self.assertEqual(OS.bh_fdr([0.001, 0.2, 0.9], 0.05), [True, False, False])
        self.assertEqual(OS.bh_fdr([0.5, 0.6], 0.05), [False, False])

    def test_perm_pvalue(self):
        self.assertEqual(OS.perm_pvalue(5.0, [1.0, 2.0, 3.0]), 0.25)
        self.assertEqual(OS.perm_pvalue(0.0, [1.0, 2.0, 3.0]), 1.0)

    def test_block_bootstrap_diff(self):
        a = {m: (1.0, 2) for m in range(20)}
        b = {m: (0.0, 2) for m in range(20)}
        pt, lo, hi = OS.block_bootstrap_diff(a, b, n=300, seed=7)
        self.assertEqual(pt, 0.5)
        self.assertGreater(lo, 0.0)
        self.assertGreaterEqual(hi, pt)
        self.assertEqual(OS.block_bootstrap_diff({}, {}, n=10), (None, None, None))

    def test_eval_plateau_2d_true_spike_and_missing(self):
        good = {(h, c): {"mean": 0.40, "lo": 0.05} for h in range(2) for c in range(2)}
        p_good = OS.eval_plateau_2d(good, nh=2, nc=2)
        self.assertTrue(p_good[(0, 0)]["ok"])
        self.assertEqual(p_good[(0, 0)]["good_nb"], 2)

        spike = {
            (0, 0): {"mean": 0.10, "lo": -0.1},
            (0, 1): {"mean": 0.10, "lo": -0.1},
            (1, 0): {"mean": 0.10, "lo": -0.1},
            (1, 1): {"mean": 0.80, "lo": 0.2},
        }
        p_spike = OS.eval_plateau_2d(spike, nh=2, nc=2)
        self.assertFalse(p_spike[(1, 1)]["ok"])
        self.assertEqual(p_spike[(1, 1)]["good_nb"], 0)
        self.assertEqual(p_spike[(1, 1)]["pos_nb"], 2)

        isolated = {(0, 0): {"mean": 0.50, "lo": 0.1}}
        p_iso = OS.eval_plateau_2d(isolated, nh=2, nc=2)
        self.assertIsNone(p_iso[(0, 0)]["ok"])
        self.assertIsNone(p_iso[(0, 1)]["ok"])


if __name__ == "__main__":
    unittest.main()
