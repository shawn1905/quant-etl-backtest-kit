#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_seat.py — NAS 期货席位持仓排名采集 (独立落盘 seat/, 断点续传)

数据来源: tqsdk api.query_symbol_ranking (LONG/SHORT 两榜, 免费)
数据流:
  1. 读最新期货主连日线 (futures/1d_fut/<year>/<latest>.parquet) 得到主连品种
  2. dominant_contract(): 主连 -> 当前主力可交易合约 (未下市 OI 最大)
  3. 主力合约 query_symbol_ranking(LONG) + (SHORT)
  4. 逐品种追加落盘 .tmp, 全部完成后合并 seat/<year>/<YYYYMMDD>.parquet (断点续传: 已落盘合约跳过)

用法:
  python3 collect_seat.py                 # 当前最新交易日, 断点续采
  python3 collect_seat.py --date 20260907  # 指定交易日(读 1d_contracts)
  python3 collect_seat.py --dry-run       # 只打印候选不落盘
  python3 collect_seat.py --limit N       # 只处理前 N 个品种(分批跑)
  python3 collect_seat.py --start IDX     # 从第 IDX 个品种继续(分批跑)
"""
import argparse, datetime as dt, os, sys, time
import pandas as pd

ROOT = os.environ.get("QUANT_DATA_ROOT", "./data")
SEAT_DIR = os.path.join(ROOT, "seat")
FUT_DIR = os.path.join(ROOT, "futures", "1d_contracts")

OUT_COLS = ["date", "symbol", "broker",
            "long_oi", "long_change", "long_ranking",
            "short_oi", "short_change", "short_ranking"]


def load_creds():
    p = os.path.expanduser("~/.tq_creds")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip().replace("export ", ""), v.strip().strip("'").strip('"'))
    if not os.environ.get("TQ_USERNAME") or not os.environ.get("TQ_PASSWORD"):
        raise RuntimeError("Missing TQ_USERNAME / TQ_PASSWORD in environment or ~/.tq_creds")


def main_symbol(exchange, product):
    product = product.upper() if exchange in ("CZCE", "CFFEX") else product.lower()
    return f"KQ.m@{exchange}.{product}"


def latest_daily_file(date_s=None):
    if not os.path.isdir(FUT_DIR):
        return None
    if date_s:
        y = date_s[:4]
        p = os.path.join(FUT_DIR, y, f"{date_s}.parquet")
        return p if os.path.exists(p) else None
    for y in sorted(os.listdir(FUT_DIR))[::-1]:
        yp = os.path.join(FUT_DIR, y)
        if not os.path.isdir(yp):
            continue
        fs = sorted(f for f in os.listdir(yp) if f.endswith(".parquet"))
        if fs:
            return os.path.join(yp, fs[-1])
    return None


def dominant_contract(api, sym, quotes_cache):
    code = sym.split("@")[-1]
    if "." not in code:
        return None
    ex = code.split(".")[0]
    prod = "".join(ch for ch in code.split(".")[1] if not ch.isdigit())
    try:
        contracts = api.query_quotes(ins_class="FUTURE", product_id=prod,
                                     exchange_id=ex, expired=False)
        contracts = [c for c in contracts if "@" not in c]
    except Exception:
        return None
    for c in contracts:
        if c not in quotes_cache:
            try:
                quotes_cache[c] = api.get_quote(c)
                api.wait_update(deadline=time.time() + 1)
            except Exception:
                quotes_cache[c] = None
    try:
        api.wait_update(deadline=time.time() + 4)
    except Exception:
        pass
    best, best_oi = None, -1
    for c in contracts:
        q = quotes_cache.get(c)
        if q is not None and getattr(q, "open_interest", None) is not None:
            if q.open_interest > best_oi:
                best_oi, best = q.open_interest, c
    return best


def fetch_seat(api, contract, rtype, date_s):
    print(f"  请求 {contract} {rtype} {date_s}", flush=True)
    try:
        frame = api.query_symbol_ranking(
            contract, ranking_type=rtype, days=1,
            start_dt=dt.datetime.strptime(date_s, "%Y%m%d").date())
        if len(frame):
            if "datetime" not in frame:
                raise ValueError("排名响应缺少源日期")
            dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.strftime("%Y%m%d")
            if dates.isna().any() or not dates.eq(date_s).all():
                raise ValueError("排名源日期不等于目标交易日")
        return frame
    except Exception as ex:
        print(f"  失败 {contract} {rtype}: {type(ex).__name__} status={getattr(ex, 'status', None)} {str(ex)[:100]}", flush=True)
        return None


def merge(contract, df_l, df_s):
    def prep(df):
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["broker"])
        g = df.copy()
        g.columns = [str(c).lower() for c in g.columns]
        return g
    l = prep(df_l)
    s = prep(df_s)
    if "broker" not in l.columns:
        l["broker"] = None
    if "broker" not in s.columns:
        s["broker"] = None
    for c, tgt in [("long_oi", "loi"), ("long_change", "lch"), ("long_ranking", "lrk")]:
        if c not in l.columns:
            l[c] = None
    for c, tgt in [("short_oi", "soi"), ("short_change", "sch"), ("short_ranking", "srk")]:
        if c not in s.columns:
            s[c] = None
    l = l.rename(columns={"long_oi": "loi", "long_change": "lch", "long_ranking": "lrk"})
    s = s.rename(columns={"short_oi": "soi", "short_change": "sch", "short_ranking": "srk"})
    brokers = []
    seen = set()
    for b in list(l["broker"]) + list(s["broker"]):
        if b is not None and b not in seen:
            seen.add(b)
            brokers.append(b)
    rows = []
    for b in brokers:
        rl = None
        for _, r in l.iterrows():
            if r["broker"] == b:
                rl = r
                break
        rs = None
        for _, r in s.iterrows():
            if r["broker"] == b:
                rs = r
                break
        rows.append({
            "broker": b,
            "long_oi": _g(rl, "loi"), "long_change": _g(rl, "lch"), "long_ranking": _g(rl, "lrk"),
            "short_oi": _g(rs, "soi"), "short_change": _g(rs, "sch"), "short_ranking": _g(rs, "srk"),
        })
    return rows


def _g(r, k):
    if r is None:
        return None
    try:
        return r[k]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default="", help="目标交易日 YYYYMMDD，默认最新 1d_contracts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    load_creds()
    from tqsdk import TqApi, TqAuth
    api = TqApi(auth=TqAuth(os.environ["TQ_USERNAME"], os.environ["TQ_PASSWORD"]),
                disable_print=True)
    try:
        time.sleep(1)

        src = latest_daily_file(args.date.strip() or None)
        if not src:
            print("找不到期货日线"); sys.exit(1)
        d = pd.read_parquet(src)
        # 从具体合约提取品种 (全合约 1d_contracts)
        import re as _re
        def _prod_of(code):
            parts = str(code).split(".")
            if len(parts) < 2:
                return None
            return "".join(ch for ch in parts[1] if ch.isalpha()).lower()
        prods = {}
        for code in d["code"]:
            p = _prod_of(code)
            if p:
                prods.setdefault(p, code)
        # doms: 品种 -> 主连风格 sym (dominant_contract 会重新查主力)
        doms = []
        for p, sample in sorted(prods.items()):
            ex = str(sample).split(".")[0]
            doms.append(main_symbol(ex, p))
        if args.start:
            doms = doms[args.start:]
        if args.limit:
            doms = doms[:args.limit]
        date_s = str(d["date"].iloc[0]) if "date" in d else os.path.basename(src)[:8]
        year = date_s[:4]
        print(f"来源: {src}\n目标交易日 {date_s}, 待处理 {len(doms)} 品种 (start={args.start})")

        tmp_path = os.path.join(SEAT_DIR, f".collect_{date_s}.tmp")
        os.makedirs(os.path.join(SEAT_DIR, year), exist_ok=True)

        # 断点: 已采品种 = 已落盘最终 parquet + .tmp 两处合并
        done = set()
        final_path = os.path.join(SEAT_DIR, year, f"{date_s}.parquet")
        if os.path.exists(final_path):
            prev = pd.read_parquet(final_path)
            done |= set(prev["symbol"].astype(str))
        if os.path.exists(tmp_path):
            prev = pd.read_parquet(tmp_path)
            done |= set(prev["symbol"].astype(str))
        print(f"已采 {len(done)} 品种, 续传跳过")

        qc = {}
        dom_cache = {}   # sym -> contract
        # 已采 contract 全部标记为已处理(便于 merge 时不去重丢)
        rows_buf = []   # 本次新采
        n_ok = 0
        failed = []
        t0 = time.time()
        # 主力合约优先取 futures/1d_fut 当日 contract 列 (与主连推导同口径, 免去逐合约订阅行情, 原每日超时的主因)
        mains = {}
        fut_f = os.path.join(ROOT, "futures", "1d_fut", year, f"{date_s}.parquet")
        if os.path.exists(fut_f):
            _m = pd.read_parquet(fut_f, columns=["code", "contract"])
            mains = {k: v for k, v in zip(_m["code"], _m["contract"].fillna("")) if v}
        print(f"1d_fut 主力合约 {len(mains)} 个")
        for i, sym in enumerate(doms):
            contract = mains.get(sym) or dominant_contract(api, sym, qc)
            if not contract:
                print(f"  [{i+1}] {sym} 无主力, 跳过")
                continue
            if contract in done:
                print(f"  [{i+1}] {sym}->{contract} 已过, 跳过")
                continue
            df_l = fetch_seat(api, contract, "LONG", date_s)
            df_s = fetch_seat(api, contract, "SHORT", date_s)
            if df_l is None or df_s is None:
                failed.append(contract)
                print(f"  [{i+1}] {sym}->{contract} 排名查询失败，不标记完成")
                continue
            rows = merge(contract, df_l, df_s)
            for r in rows:
                r["symbol"] = contract
                r["date"] = date_s
                rows_buf.append(r)
            done.add(contract)
            n_ok += 1
            print(f"  [{i+1}] {sym}->{contract} {len(rows)}席位 (累计{len(rows_buf)}行)")
            # 增量写 .tmp (断点)
            tmpdf = pd.DataFrame(rows_buf) if not os.path.exists(tmp_path) else pd.concat(
                [pd.read_parquet(tmp_path), pd.DataFrame(rows_buf)])
            tmpdf.to_parquet(tmp_path, index=False)
            rows_buf = []
        print(f"\n本次采集 {n_ok} 品种, 耗 {int(time.time()-t0)}s")

        if args.dry_run:
            print("dry-run 不合并落盘")
            return
        if not done and not rows_buf:
            print("无数据"); sys.exit(1)

        # 合并到最终 parquet (.tmp 本次新增 + 已存在 final 旧数据合并去重)
        frames = []
        final_path = os.path.join(SEAT_DIR, year, f"{date_s}.parquet")
        if os.path.exists(tmp_path):
            frames.append(pd.read_parquet(tmp_path))
        if os.path.exists(final_path):
            frames.append(pd.read_parquet(final_path))
        if frames:
            full = pd.concat(frames, ignore_index=True)
            # 保序去重(同 symbol 同 date)
            full = full.drop_duplicates(subset=["symbol", "broker", "date"], keep="last")
            full.to_parquet(final_path, index=False)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"\n✅ 合并落盘 {final_path}  {len(full)}行 {full['symbol'].nunique()}品种")
            print(full.head(3).to_string())
        if failed:
            raise RuntimeError(f"{len(failed)}个合约排名请求失败，已保存成功部分，需续采")

    finally:
        api.close()


if __name__ == "__main__":
    main()