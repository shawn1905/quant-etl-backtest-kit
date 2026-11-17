import datetime
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ETL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "etl"))
if ETL_DIR not in sys.path:
    sys.path.insert(0, ETL_DIR)

try:
    import pandas as pd
    import update_1dopt_openctp as U
    import update_contracts_openctp as C
    import check_integrity as CI
except ImportError:
    pd = None  # type: ignore[assignment]
    U = C = CI = None  # type: ignore[assignment]


@unittest.skipIf(pd is None, "ETL tests require optional pandas/pyarrow dependencies")
class PriceFallback(unittest.TestCase):
    def test_traded_last_price(self):
        self.assertEqual(U.close_price(dict(Volume=21, ClosePrice=0., SettlementPrice=0., LastPrice=10., AskPrice1=10.)), (10., 'LastPrice'))

    def test_priority_and_inactive(self):
        self.assertEqual(U.close_price(dict(Volume=1, ClosePrice=20, SettlementPrice=30, LastPrice=40)), (20., 'ClosePrice'))
        self.assertEqual(U.close_price(dict(Volume=1, ClosePrice=0, SettlementPrice=30, LastPrice=40)), (30., 'SettlementPrice'))
        self.assertEqual(U.close_price(dict(Volume=0, ClosePrice=20, SettlementPrice=30, LastPrice=40)), (30., 'SettlementPrice'))
        self.assertEqual(U.close_price(dict(Volume=0, ClosePrice=0, SettlementPrice=0, LastPrice=40)), (0., None))

    def test_invalid_values_not_prices(self):
        self.assertEqual(U.close_price(dict(Volume=1, ClosePrice=math.nan, SettlementPrice=-1, LastPrice=10)), (10., 'LastPrice'))
        self.assertEqual(U.close_price(dict(Volume=1, ClosePrice=math.inf, SettlementPrice=0, AskPrice1=10, LowerLimitPrice=10)), (0., None))

    def test_no_valid_price_fails_before_replacing(self):
        with tempfile.TemporaryDirectory() as out:
            p = out + '/2026/20261014.parquet'
            os.makedirs(out + '/2026')
            Path(p).write_bytes(b'original')
            r = dict(InstrumentID='lc2611-C-198000', ExchangeID='GFEX', Volume=3, ClosePrice=0, SettlementPrice=0, LastPrice=0, OpenInterest=10)
            with patch.object(U, 'MIN_ROWS', 1), patch.object(U, 'OUT', out),                  patch.object(U, 'fetch_prices', return_value=dict(rsp_code=0, data=[r])),                  patch.object(U.sys, 'argv', ['etl', '20261014']):
                with self.assertRaises(ValueError):
                    U.main()
            self.assertEqual(Path(p).read_bytes(), b'original')

    def test_api_failure_is_failure(self):
        with patch.object(U, 'fetch_prices', return_value=dict(rsp_code=1, rsp_message='bad')),              patch.object(U, '_can_use_quote_snapshot', return_value=False),              patch.object(U.sys, 'argv', ['etl', '20261014']):
            with self.assertRaises(RuntimeError):
                U.main()

    def test_small_payload_rejected_without_bypass(self):
        r = dict(InstrumentID='lc2611-C-198000', ExchangeID='GFEX', Volume=3, ClosePrice=10, SettlementPrice=10, OpenInterest=5)
        msg = U._openctp_problem(dict(rsp_code=0, data=[r]))
        self.assertIn('行数过少', msg)

    def test_null_oi_triggers_tqsdk_fallback_and_ohl_fill(self):
        bad_openctp = dict(rsp_code=0, data=[
            dict(InstrumentID=f'lc2611-C-{k}', ExchangeID='GFEX', Volume=1, ClosePrice=10, SettlementPrice=10, OpenInterest=None)
            for k in range(5)
        ])
        tq_data = dict(rsp_code=0, data=[
            dict(InstrumentID='lc2611-C-1000', ExchangeID='GFEX', PreSettlementPrice=9.0, PreClosePrice=9.0,
                 OpenPrice=None, HighestPrice=None, LowestPrice=None, ClosePrice=12.0, SettlementPrice=12.0,
                 LastPrice=12.0, Volume=5.0, Turnover=100.0, OpenInterest=88.0),
        ])
        with tempfile.TemporaryDirectory() as out:
            with patch.object(U, 'MIN_ROWS', 1), patch.object(U, 'OUT', out),                  patch.object(U, 'fetch_prices', return_value=bad_openctp),                  patch.object(U, '_can_use_quote_snapshot', return_value=True),                  patch.object(U, 'fetch_tqsdk_prices', return_value=tq_data),                  patch.object(U.sys, 'argv', ['etl', '20261116']):
                U.main()
            df = pd.read_parquet(f'{out}/2026/20261116.parquet')
            self.assertEqual(str(df['oi'].dtype), 'Int64')
            self.assertEqual(int(df['oi'].iloc[0]), 88)
            self.assertEqual(float(df['open'].iloc[0]), 12.0)


@unittest.skipIf(pd is None, "ETL tests require optional pandas/pyarrow dependencies")
class ContractsAndIntegrityGuards(unittest.TestCase):
    def test_pick_close_unifies_zero_volume(self):
        self.assertEqual(C._pick_close(0, 0.0, 2500.0, 2490.0), 2500.0)
        self.assertEqual(C._pick_close(0, 0.0, 0.0, 2490.0), 2490.0)
        self.assertEqual(C._pick_close(10, 2510.0, 2500.0, 2490.0), 2510.0)

    def test_normalize_ohl_fills_traded_missing(self):
        df = pd.DataFrame({
            'code': ['DCE.a2701', 'CZCE.JR701'],
            'open': [float('nan'), 0.0],
            'high': [float('nan'), 0.0],
            'low': [float('nan'), 0.0],
            'close': [4000.0, 3000.0],
            'vol': [10.0, 0.0],
        })
        out = C._normalize_ohl(df.copy())
        self.assertEqual(float(out.loc[0, 'open']), 4000.0)
        self.assertTrue(pd.isna(out.loc[1, 'open']))

    def test_can_use_quote_snapshot_blocks_night_session(self):
        class FakeDT(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 16, 20, 50, 0)
        with patch.object(C.datetime, 'datetime', FakeDT), patch.object(U.datetime, 'datetime', FakeDT):
            self.assertFalse(C._can_use_quote_snapshot('2026-11-16'))
            self.assertFalse(U._can_use_quote_snapshot('2026-11-16'))

    def test_check_integrity_catches_nan_ohl_on_traded(self):
        df_c = pd.DataFrame({
            'code': [f'SHFE.cu270{i}' for i in range(60)],
            'trade_time': ['2026-11-16 00:00:00'] * 60,
            'open': [float('nan')] + [100.0] * 59,
            'high': [float('nan')] + [101.0] * 59,
            'low': [float('nan')] + [99.0] * 59,
            'close': [100.0] * 60,
            'vol': [10.0] * 60,
            'amount': [1000.0] * 60,
            'oi': [50.0] * 60,
            'date': ['20261116'] * 60,
        })
        df_f = pd.DataFrame({
            'code': [f'KQ.m@SHFE.cu{i}' for i in range(45)],
            'trade_time': ['2026-11-16 00:00:00'] * 45,
            'open': [100.0] * 45,
            'high': [101.0] * 45,
            'low': [99.0] * 45,
            'close': [100.0] * 45,
            'vol': [10.0] * 45,
            'amount': [1000.0] * 45,
            'oi': [50.0] * 45,
            'date': ['20261116'] * 45,
            'contract': [f'SHFE.cu270{i}' for i in range(45)],
        })
        df_o = pd.DataFrame({
            'code': [f'CU2701C{10000 + i}.SHF' for i in range(550)],
            'date': ['20261116'] * 550,
            'exchange': ['SHFE'] * 550,
            'open': [10.0] * 550, 'high': [11.0] * 550, 'low': [9.0] * 550, 'close': [10.0] * 550,
            'vol': [5.0] * 550, 'amount': [500.0] * 550, 'oi': [20] * 550,
        })
        df_g = pd.DataFrame({
            'code': ['CU2701C10000.SHF'], 'trade_time': ['2026-11-16 00:00:00'],
            'iv': [0.2], 'delta': [0.5], 'gamma': [0.01], 'vega': [1.0], 'theta': [-1.0], 'S': [100.0],
        })
        with tempfile.TemporaryDirectory() as r:
            for sub, d in [('futures/1d_contracts', df_c), ('futures/1d_fut', df_f),
                           ('options/1d_opt', df_o), ('greeks/1d_greeks_v2', df_g)]:
                os.makedirs(f'{r}/{sub}/2026', exist_ok=True)
                d.to_parquet(f'{r}/{sub}/2026/20261116.parquet', index=False)
            ok, rep = CI.run_checks('20261116', root=r, require_enrichment=False)
            self.assertFalse(ok)
            self.assertIn('sanity.1d_contracts', rep['failed'])


if __name__ == '__main__':
    unittest.main()
