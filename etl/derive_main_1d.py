#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derive_main_1d.py — 由 futures/1d_contracts 推导主连日线 futures/1d_fut (不再单独下载主连)

规则 (与天勤 KQ.m 一致, 无未来函数):
  - D 日主力 = D-1 日收盘持仓量最大的合约;
  - 只向远月切换, 不回滚 (最大持仓合约比当前主力更近月时保持原主力);
  - 当前主力 D 日已不存在 (到期/摘牌) 时, 取 D-1 持仓最大的在市合约, 仍无则取 D 日持仓最大.
  - 输出列与旧 1d_fut 相同 (code=KQ.m@EXCH.prod, trade_time=YYYY-MM-DD 00:00:00 北京交易日),
    另加物理合约列 contract (如 SHFE.rb2610).

无状态: 每次从目标区间前 WARMUP 个交易日起顺推, 结果只由 1d_contracts 决定.
交易日历只用 options/1d_opt 的工作日文件 (1d_contracts 混有周末文件).

用法:
  python3 derive_main_1d.py --date 20260924                        # 写 futures/1d_fut/2026/20260924.parquet
  python3 derive_main_1d.py --start 20260812 --end 20260924 --out DIR   # 区间写到指定目录 (DIR/YYYY/...)
"""

import argparse
import datetime
import glob
import os
import re
import sys

import pandas as pd

ROOT_DIR = os.environ.get("QUANT_DATA_ROOT", "./data")
CONTRACTS_DIR = os.path.join(ROOT_DIR, "futures", "1d_contracts")
FUT_DIR = os.path.join(ROOT_DIR, "futures", "1d_fut")
CAL_DIR = os.path.join(ROOT_DIR, "options", "1d_opt")
WARMUP = 60

CODE_RE = re.compile(r"^([A-Z]+)\.([A-Za-z]+)(\d{3,4})$")
COLS = ["code", "trade_time", "open", "high", "low", "close", "vol", "amount", "oi", "date", "contract"]


def trading_calendar():
    """交易日 = 期权日线或期货合约日线存在的工作日.
    (2026-09-28 修: 原先只看 1d_opt, 但 ETL 里期权日线在本步之后才拉, 当天总被判为"不在交易日历中" → 整条流水线终止;
     1d_contracts 是 ETL 第一步的产物, 拉到数据本身即说明当天开盘.)"""
    days = []
    for p in glob.glob(os.path.join(CAL_DIR, "*", "*.parquet")) + glob.glob(os.path.join(CONTRACTS_DIR, "*", "*.parquet")):
        d = os.path.basename(p)[:8]
        if d.isdigit() and datetime.datetime.strptime(d, "%Y%m%d").weekday() < 5:
            days.append(d)
    return sorted(set(days))


def expiry_key(month_digits: str, day: str) -> int:
    """合约月份 -> YYYYMM 整数; 郑商所 3 位月份按所在十年补全."""
    if len(month_digits) == 4:
        return 200000 + int(month_digits)
    year = int(day[:4])
    y = (year // 10) * 10 + int(month_digits[0])
    if y < year - 1:
        y += 10
    return y * 100 + int(month_digits[1:])


def load_day(day: str) -> pd.DataFrame:
    p = os.path.join(CONTRACTS_DIR, day[:4], f"{day}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    m = df["code"].astype(str).str.extract(CODE_RE)
    df = df[m[0].notna()].copy()
    m = m[m[0].notna()]
    df["exch"], df["prod"], df["mon"] = m[0], m[1], m[2]
    df["ekey"] = [expiry_key(x, day) for x in df["mon"]]
    df["main_code"] = "KQ.m@" + df["exch"] + "." + df["prod"]
    return df


def derive(days_out, calendar):
    first = calendar.index(days_out[0])
    run_days = calendar[max(0, first - WARMUP): calendar.index(days_out[-1]) + 1]
    want = set(days_out)
    mains = {}          # main_code -> (contract, ekey)
    prev = None         # D-1 数据
    out = {}
    for day in run_days:
        cur = load_day(day)
        if cur is None:
            print(f"  ⚠️ {day} 缺 1d_contracts 文件, 跳过 (主力状态沿用)")
            continue
        cur_by = cur.set_index("code")
        rows = []
        for mc, g in cur.groupby("main_code", sort=True):
            live = set(g["code"])
            pick = None
            if prev is not None:
                pg = prev[(prev["main_code"] == mc) & (prev["code"].isin(live)) & (prev["oi"] > 0)]
                if len(pg):
                    top = pg.sort_values(["oi", "ekey"], ascending=[False, False]).iloc[0]
                    pick = (top["code"], top["ekey"])
            old = mains.get(mc)
            if old is not None and old[0] in live and (pick is None or pick[1] <= old[1]):
                pick = old
            if pick is None:
                top = g.sort_values(["oi", "ekey"], ascending=[False, False]).iloc[0]
                pick = (top["code"], top["ekey"])
            mains[mc] = pick
            if day in want:
                r = cur_by.loc[pick[0]]
                rows.append({
                    "code": mc,
                    "trade_time": f"{day[:4]}-{day[4:6]}-{day[6:]} 00:00:00",
                    "open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "vol": float(r["vol"]), "amount": float(r["amount"]), "oi": float(r["oi"]),
                    "date": day, "contract": pick[0],
                })
        if day in want:
            out[day] = pd.DataFrame(rows, columns=COLS)
        prev = cur
    return out


def write_atomic(df, out_root, day):
    d = os.path.join(out_root, day[:4])
    os.makedirs(d, exist_ok=True)
    final = os.path.join(d, f"{day}.parquet")
    tmp = final + f".tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    return final


def main():
    global CONTRACTS_DIR
    ap = argparse.ArgumentParser(description="由 1d_contracts 推导主连日线")
    ap.add_argument("--date", default="")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--out", default=FUT_DIR, help="输出根目录 (默认 futures/1d_fut)")
    ap.add_argument("--contracts-dir", default=CONTRACTS_DIR, help="1d_contracts 根目录 (默认正式目录)")
    args = ap.parse_args()
    CONTRACTS_DIR = args.contracts_dir

    cal = trading_calendar()
    if args.date:
        days = [args.date]
    else:
        days = [d for d in cal if args.start <= d <= (args.end or "99999999")]
    days = [d for d in days if d in cal]
    if not days:
        print(f"❌ 目标日期不在交易日历中: {args.date or (args.start, args.end)}")
        sys.exit(2)

    out = derive(days, cal)
    missing = [d for d in days if d not in out or out[d].empty]
    for d, df in sorted(out.items()):
        if df.empty:
            continue
        p = write_atomic(df, args.out, d)
        print(f"  ✅ {d}: {len(df)} 个品种 -> {p}")
    if missing:
        print(f"❌ 以下交易日无法推导 (缺 1d_contracts): {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
