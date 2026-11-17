#!/usr/bin/env python3
"""check_integrity.py — 交易日 T 的数据完整性检查 (ETL 最后一步, 不通过则不发布 manifest)

检查项 (DATA_LAYER_HANDOVER §4.6):
  coverage   : 各层当日文件存在且非空 (1d_contracts / 1d_fut / 1d_opt / greeks / 1m 期权与期货 / 5m)
  time_axis  : 日线 trade_time 为 'YYYY-MM-DD 00:00:00' (无 16:00 UTC 痕迹); 1m 时刻落在交易时段; (code, trade_time) 无重复
  cross      : 1d_fut 每行的 (contract, close, oi) 能在 1d_contracts 找到; 期货 1m 日盘最后一根 close = 日线 close (≥95%);
               1d_opt 代码为库内格式 (大写品种 + 交易所后缀)
  sanity     : 有成交合约 open/high/low/close > 0 且非空、high >= low; oi/vol 非空率与活跃合约下限; greeks 非空率

用法: python3 check_integrity.py --date T [--root DIR] [--core-only]
"""
import argparse, json, os, re, sys
import numpy as np, pandas as pd

ROOT = os.environ.get("QUANT_DATA_ROOT", "./data")
OPT_CODE = re.compile(r"^[A-Z]+\d{3,4}(-?[A-Z]{2})?-?[CP]-?[\d.]+\.(SHF|DCE|ZCE|GFE|INE|CFX)$|^\d{8}\.(SH|SZ)$")


def rd(root, rel, d):
    p = f"{root}/{rel}/{d[:4]}/{d}.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else None


def run_checks(d, root=ROOT, require_enrichment=True):
    R, fails = {}, []

    def check(name, ok, detail):
        R[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    core_layers = {"1d_contracts": "futures/1d_contracts", "1d_fut": "futures/1d_fut",
                   "1d_opt": "options/1d_opt", "greeks": "greeks/1d_greeks_v2"}
    enrich_layers = {"opt_1m": "options/1m_opt", "fut_1m": "futures/1m_contracts",
                     "opt_5m": "options/5m_opt", "fut_5m": "futures/5m_contracts"}
    layers = {**core_layers, **(enrich_layers if require_enrichment else {})}
    data = {k: rd(root, v, d) for k, v in layers.items()}
    for k, df in data.items():
        check(f"coverage.{k}", df is not None and len(df) > 0, 0 if df is None else len(df))

    c, f, o, gr = data["1d_contracts"], data["1d_fut"], data["1d_opt"], data["greeks"]
    tt = f"{d[:4]}-{d[4:6]}-{d[6:]} 00:00:00"
    for name, df in (("1d_contracts", c), ("1d_fut", f)):
        if df is not None:
            bad = int((df["trade_time"].astype(str) != tt).sum())
            check(f"time_axis.{name}_trade_time", bad == 0, f"{bad} 行不是 {tt}")
            dup = int(df.duplicated("code").sum())
            check(f"time_axis.{name}_dup", dup == 0, dup)
    if require_enrichment:
        for name in ("opt_1m", "fut_1m"):
            df = data[name]
            if df is not None:
                hh = df["trade_time"].str[11:13].astype(int)
                mm = df["trade_time"].str[14:16].astype(int)
                t = hh * 60 + mm
                in_session = ((t >= 9 * 60) & (t <= 15 * 60 + 15)) | (t >= 21 * 60) | (t <= 2 * 60 + 30)
                check(f"time_axis.{name}_session", float(in_session.mean()) > 0.999, f"时段外 {int((~in_session).sum())} 行")
                dup = int(df.duplicated(["code", "trade_time"]).sum())
                check(f"time_axis.{name}_dup", dup == 0, dup)

    if f is not None and c is not None and "contract" in f:
        known = f[f["contract"].fillna("").astype(str) != ""]
        j = known.merge(c[["code", "close", "oi"]].rename(columns={"code": "contract"}), on="contract", how="left", suffixes=("", "_c"))
        miss = int(j["close_c"].isna().sum())
        mism = int((j["close_c"].notna() & ((j["close"] - j["close_c"]).abs() > 1e-6)).sum())
        check("cross.1d_fut_in_contracts", miss == 0 and mism == 0,
              f"找不到 {miss}, 价格不一致 {mism}, 合约未知 {len(f) - len(known)} (旧段反查不到, 不计失败)")
    if require_enrichment:
        fm = data["fut_1m"]
        if fm is not None and c is not None:
            day = fm[(fm["trade_time"].str[:10] == tt[:10]) & (fm["trade_time"].str[11:16] <= "15:15")]
            last = day.sort_values("trade_time").groupby("code").tail(1)
            last = last[last["trade_time"].str[11:16] >= "14:59"]
            j = last.merge(c[["code", "close", "vol"]], on="code", suffixes=("_1m", "_1d"))
            j = j[j["vol_1d"] > 0]
            rate = float(((j["close_1m"] - j["close_1d"]).abs() < 1e-6).mean()) if len(j) else 0.0
            check("cross.fut_1m_close_vs_1d", rate >= 0.95, f"{len(j)} 个有成交合约, 一致 {rate:.1%}")
    if o is not None:
        bad = o[~o["code"].astype(str).str.match(OPT_CODE)]
        check("cross.1d_opt_code_format", len(bad) == 0, f"{len(bad)} 个非库内格式, 例 {bad['code'].head(3).tolist()}")

    for name, df, min_tr in (("1d_contracts", c, 50), ("1d_fut", f, 40), ("1d_opt", o, 500)):
        if df is None:
            continue
        v = pd.to_numeric(df["vol"], errors="coerce")
        oi = pd.to_numeric(df["oi"], errors="coerce")
        op = pd.to_numeric(df["open"], errors="coerce")
        hi = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        cl = pd.to_numeric(df["close"], errors="coerce")
        tr = v > 0
        bad_px = int((tr & ~(cl > 0)).sum())
        bad_ohl = int((tr & (~(op > 0) | ~(hi > 0) | ~(lo > 0) | (hi < lo))).sum())
        oi_rate = float(oi.notna().mean()) if len(df) else 0.0
        vol_rate = float(v.notna().mean()) if len(df) else 0.0
        tr_cnt = int(tr.sum())
        all_cl_pos = True if name == "1d_opt" else bool((cl > 0).all())
        ok = (bad_px == 0 and bad_ohl == 0 and oi_rate >= 0.8 and vol_rate >= 0.8 and tr_cnt >= min_tr and all_cl_pos)
        check(f"sanity.{name}", ok,
              f"bad_px={bad_px}, bad_ohl={bad_ohl}, oi={oi_rate:.1%}, vol={vol_rate:.1%}, traded={tr_cnt}, cl_pos={all_cl_pos}")

    if gr is not None:
        iv_ok = float(pd.to_numeric(gr["iv"], errors="coerce").notna().mean())
        delta_ok = float(pd.to_numeric(gr["delta"], errors="coerce").notna().mean())
        s_ok = float(pd.to_numeric(gr["S"], errors="coerce").notna().mean())
        check("sanity.greeks", iv_ok >= 0.5 and delta_ok >= 0.6 and s_ok >= 0.99,
              f"iv={iv_ok:.1%}, delta={delta_ok:.1%}, S={s_ok:.1%}")

    return len(fails) == 0, {"date": d, "root": root, "failed": fails, "checks": R}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--core-only", action="store_true")
    a = ap.parse_args()
    ok, rep = run_checks(a.date, a.root, require_enrichment=not a.core_only)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
