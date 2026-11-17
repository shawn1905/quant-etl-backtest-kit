"""1d_contracts 增量: openctp futures 为主, 缺字段/异常时自动切换 TqSdk 兜底 (全市场)."""
import datetime
import json
import math
import os
import re
import sys
import time
import urllib.request
import pandas as pd

ROOT = os.environ.get("QUANT_DATA_ROOT", "./data")
OUT = f"{ROOT}/futures/1d_contracts"
COLS = ["code", "trade_time", "open", "high", "low", "close", "vol", "amount", "oi", "date"]
EXCHANGES = ["SHFE", "DCE", "CZCE", "GFEX", "INE", "CFFEX"]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())


def _is_num(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _pos_or_nan(v):
    return float(v) if _is_num(v) and float(v) > 0 else float("nan")


def _num_or_nan(v):
    return float(v) if _is_num(v) else float("nan")


def _pick_close(vol, close_val, settle_val, pre_settle_val=None, last_val=None):
    """统一口径: 有成交按 close->settle->last->pre_settle; 零成交按 settle->close->pre_settle, 杜绝 0.0."""
    v = float(vol) if _is_num(vol) else 0.0
    order = (close_val, settle_val, last_val, pre_settle_val) if v > 0 else (settle_val, close_val, pre_settle_val)
    for cand in order:
        if _is_num(cand) and float(cand) > 0:
            return float(cand)
    return float("nan")


def _contract_month(sym):
    body = sym.split(".", 1)[-1]
    m = re.match(r"^[A-Za-z]+(\d{3,4})", body)
    if not m:
        return None
    d = m.group(1)
    return int("2" + d) if len(d) == 3 else int(d)


def _can_use_quote_snapshot(td):
    now = datetime.datetime.now()
    return td == now.strftime("%Y-%m-%d") and now.strftime("%H%M") < "2045"


def _load_tq_creds():
    p = os.path.expanduser("~/.tq_creds")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip().replace("export ", ""), v.strip().strip("\"'"))


def _normalize_ohl(df):
    traded = df["vol"].fillna(0) > 0
    for col in ("open", "high", "low"):
        df[col] = df[col].where(df[col] > 0)
        df.loc[traded & df[col].isna(), col] = df.loc[traded & df[col].isna(), "close"]
    return df


def _validate_df(df):
    if df.empty or len(df) < 500:
        raise RuntimeError(f"合约数过少 ({len(df)})")
    oi_ok = float(df["oi"].notna().mean())
    vol_ok = float(df["vol"].notna().mean())
    traded_n = int((df["vol"].fillna(0) > 0).sum())
    close_ok = bool((df["close"] > 0).all())
    traded = df[df["vol"].fillna(0) > 0]
    ohl_ok = bool(traded[["open", "high", "low"]].notna().all().all()) and bool((traded["high"] >= traded["low"]).all())
    if oi_ok < 0.8 or vol_ok < 0.8 or traded_n < 50 or not close_ok or not ohl_ok:
        raise RuntimeError(
            f"关键字段不完整: rows={len(df)} oi={oi_ok:.1%} vol={vol_ok:.1%} traded={traded_n} close_pos={close_ok} ohl={ohl_ok}"
        )


def _fetch_openctp(day, td):
    d = fetch(f"http://dict.openctp.cn/prices?types=futures&tradingday={td}")
    if d.get("rsp_code") != 0 or not d.get("data"):
        raise RuntimeError(f"OpenCTP 接口异常: {d.get('rsp_message')}")
    rows = []
    for x in d["data"]:
        ex = x["ExchangeID"]
        code = f"{ex}.{x['InstrumentID']}"
        vol = _num_or_nan(x.get("Volume"))
        c = _pick_close(vol, x.get("ClosePrice"), x.get("SettlementPrice"),
                        x.get("PreSettlementPrice"), x.get("LastPrice"))
        rows.append({
            "code": code,
            "trade_time": f"{td} 00:00:00",
            "open": _pos_or_nan(x.get("OpenPrice")),
            "high": _pos_or_nan(x.get("HighestPrice")),
            "low": _pos_or_nan(x.get("LowestPrice")),
            "close": c,
            "vol": vol,
            "amount": _num_or_nan(x.get("Turnover")),
            "oi": _num_or_nan(x.get("OpenInterest")),
            "date": day,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code")
    for c in ["open", "high", "low", "close", "vol", "amount", "oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = _normalize_ohl(df)
    _validate_df(df)
    return df


def _fetch_tqsdk(day, td):
    _load_tq_creds()
    from tqsdk import TqApi, TqAuth

    def new_api():
        return TqApi(auth=TqAuth(os.environ["TQ_USERNAME"], os.environ["TQ_PASSWORD"]))

    min_ym = int(day[2:6])
    api = new_api()
    try:
        syms = set()
        for ex in EXCHANGES:
            syms.update(api.query_quotes(ins_class="FUTURE", exchange_id=ex, expired=False))
            for s in api.query_quotes(ins_class="FUTURE", exchange_id=ex, expired=True):
                if (_contract_month(s) or 0) >= min_ym:
                    syms.add(s)
        syms = sorted(syms)
        qs = []
        if _can_use_quote_snapshot(td):
            for i in range(0, len(syms), 300):
                qs.extend(api.get_quote_list(syms[i:i + 300]))
    finally:
        api.close()

    if qs:
        same_day = sum(1 for q in qs if str(getattr(q, "datetime", ""))[:10] == td)
        if same_day >= len(qs) * 0.8:
            rows = []
            for q in qs:
                if str(getattr(q, "datetime", ""))[:10] != td:
                    continue
                vol = _num_or_nan(q.volume)
                c = _pick_close(vol, q.close, q.settlement, q.pre_settlement, q.last_price)
                rows.append({
                    "code": q.instrument_id,
                    "trade_time": f"{td} 00:00:00",
                    "open": _pos_or_nan(q.open),
                    "high": _pos_or_nan(q.highest),
                    "low": _pos_or_nan(q.lowest),
                    "close": c,
                    "vol": vol,
                    "amount": _num_or_nan(q.amount),
                    "oi": _num_or_nan(q.open_interest),
                    "date": day,
                })
            df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code")
            for c in ["open", "high", "low", "close", "vol", "amount", "oi"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df = _normalize_ohl(df)
            _validate_df(df)
            return df

    print(f"  [fallback] TqSdk 改用日K回补 {day} (非盘后未开夜盘窗口)...", flush=True)
    rows = []
    for chunk_start in range(0, len(syms), 60):
        chunk = syms[chunk_start:chunk_start + 60]
        for attempt in range(2):
            api = new_api()
            chunk_rows = []
            try:
                for sym in chunk:
                    try:
                        k = api.get_kline_serial(sym, 86400, data_length=15)
                    except ValueError as e:
                        m = re.search(r"from shape \((\d+),", str(e))
                        if not m:
                            continue
                        k = api.get_kline_serial(sym, 86400, data_length=int(m.group(1)))
                    k = k[k["datetime"] > 0].copy()
                    if k.empty:
                        continue
                    k["T"] = (
                        pd.to_datetime(k["datetime"], unit="ns", utc=True)
                        .dt.tz_convert("Asia/Shanghai")
                        .dt.strftime("%Y%m%d")
                    )
                    sub = k[k["T"] == day]
                    if sub.empty:
                        continue
                    bar = sub.iloc[-1]
                    chunk_rows.append({
                        "code": sym,
                        "trade_time": f"{td} 00:00:00",
                        "open": _pos_or_nan(bar["open"]),
                        "high": _pos_or_nan(bar["high"]),
                        "low": _pos_or_nan(bar["low"]),
                        "close": _pos_or_nan(bar["close"]),
                        "vol": _num_or_nan(bar["volume"]),
                        "amount": float("nan"),
                        "oi": _num_or_nan(bar["close_oi"]),
                        "date": day,
                    })
                rows.extend(chunk_rows)
                break
            except Exception as e:
                if attempt == 1:
                    raise
                print(f"  [retry] TqSdk chunk {chunk_start} 重试: {e}", flush=True)
                time.sleep(2)
            finally:
                api.close()
    df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code")
    for c in ["open", "high", "low", "close", "vol", "amount", "oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = _normalize_ohl(df)
    _validate_df(df)
    return df


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y%m%d")
    td = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    source = "openctp"
    try:
        df = _fetch_openctp(day, td)
    except Exception as e:
        print(f"! OpenCTP 1d_contracts 不可用 ({e})，切换 TqSdk 兜底...", flush=True)
        df = _fetch_tqsdk(day, td)
        source = "tqsdk"

    out = f"{OUT}/{day[:4]}/{day}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f".tmp.{os.getpid()}"
    df[COLS].to_parquet(tmp, index=False)
    os.replace(tmp, out)
    ex_cnt = df.code.str.split(".").str[0].value_counts().to_dict()
    print(f"OK 1d_contracts {day} ({source}): {len(df)} 行 -> {out}", flush=True)
    print(f"  交易所: {ex_cnt}", flush=True)


if __name__ == "__main__":
    main()
