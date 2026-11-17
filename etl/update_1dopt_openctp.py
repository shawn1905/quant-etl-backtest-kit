"""1d_opt 每日增量: openctp 数据中心 prices 接口为主, 缺字段/异常时自动切换 TqSdk 兜底
用法: python3 update_1dopt_openctp.py [YYYYMMDD]  # 默认最新交易日
输出: options/1d_opt/<year>/<day>.parquet (与历史格式一致)
"""
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
OUT = f"{ROOT}/options/1d_opt"
SUFFIX = {"DCE": ".DCE", "SHFE": ".SHF", "CZCE": ".ZCE", "GFEX": ".GFE", "INE": ".INE",
          "CFFEX": ".CFX", "SSE": ".SH", "SZSE": ".SZ"}
OPT_COLS = ["code", "date", "exchange", "pre_settle", "pre_close",
            "open", "high", "low", "close", "settle", "vol", "amount", "oi"]
TQ_EXCHANGES = ["SHFE", "DCE", "CZCE", "GFEX", "INE", "CFFEX"]
MIN_ROWS = 10000


def lib_code(instrument_id, exchange):
    """库内格式 (与 <=20260810 历史及购买的 1m 一致): 品种字母大写 + 交易所后缀."""
    i = 0
    while i < len(instrument_id) and instrument_id[i].isalpha():
        i += 1
    return instrument_id[:i].upper() + instrument_id[i:] + SUFFIX.get(exchange, "")


def fetch_prices(tradingday):
    url = f"http://dict.openctp.cn/prices?types=option&tradingday={tradingday}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())


def _clean_num(v):
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


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


def fetch_tqsdk_prices(tradingday):
    """用 TqSdk 批量 quote 获取当日商品/中金所期权全市场日线 (ETF 期权由 update_1dopt_etf.py 覆盖)."""
    if not _can_use_quote_snapshot(tradingday):
        raise RuntimeError(f"TqSdk 期权 quote 仅适用于当日 20:45 夜盘前 (当前非 {tradingday} 安全窗口)")
    _load_tq_creds()
    from tqsdk import TqApi, TqAuth

    def new_api():
        return TqApi(auth=TqAuth(os.environ["TQ_USERNAME"], os.environ["TQ_PASSWORD"]))

    min_ym = int(tradingday[2:4] + tradingday[5:7])
    api = new_api()
    try:
        active_syms = set()
        expired_recent_syms = set()
        for ex in TQ_EXCHANGES:
            active_syms.update(api.query_quotes(ins_class="OPTION", exchange_id=ex, expired=False))
            for s in api.query_quotes(ins_class="OPTION", exchange_id=ex, expired=True):
                if (_contract_month(s) or 0) >= min_ym:
                    expired_recent_syms.add(s)
        expired_recent_syms -= active_syms
        syms = sorted(active_syms | expired_recent_syms)
    finally:
        api.close()

    rows = []
    active_rows = 0
    dates = {}
    for chunk_start in range(0, len(syms), 2500):
        chunk = syms[chunk_start:chunk_start + 2500]
        for attempt in range(2):
            api = new_api()
            chunk_rows = []
            chunk_active_hits = 0
            try:
                for i in range(0, len(chunk), 500):
                    for q in api.get_quote_list(chunk[i:i + 500]):
                        dt = str(getattr(q, "datetime", ""))[:10]
                        dates[dt] = dates.get(dt, 0) + 1
                        if dt != tradingday:
                            continue
                        if q.instrument_id in active_syms:
                            chunk_active_hits += 1
                        ex, inst = q.instrument_id.split(".", 1)
                        chunk_rows.append({
                            "ExchangeID": ex,
                            "InstrumentID": inst,
                            "PreSettlementPrice": _clean_num(q.pre_settlement),
                            "PreClosePrice": _clean_num(q.pre_close),
                            "OpenPrice": _clean_num(q.open),
                            "HighestPrice": _clean_num(q.highest),
                            "LowestPrice": _clean_num(q.lowest),
                            "ClosePrice": _clean_num(q.close),
                            "SettlementPrice": _clean_num(q.settlement),
                            "LastPrice": _clean_num(q.last_price),
                            "Volume": _clean_num(q.volume),
                            "Turnover": _clean_num(q.amount),
                            "OpenInterest": _clean_num(q.open_interest),
                        })
                rows.extend(chunk_rows)
                active_rows += chunk_active_hits
                break
            except Exception as e:
                if attempt == 1:
                    raise
                print(f"  [retry] TqSdk opt chunk {chunk_start} 重试: {e}", flush=True)
                time.sleep(2)
            finally:
                api.close()

    if len(rows) < MIN_ROWS or active_rows < int(len(active_syms) * 0.8):
        raise RuntimeError(
            f"TqSdk 期权 quote 日期不匹配 {tradingday}: {dates}, "
            f"active={active_rows}/{len(active_syms)}, total={len(rows)}/{len(syms)}"
        )
    return {"rsp_code": 0, "data": rows}


def close_price(row):
    """Keep settlement placeholders for inactive contracts; LastPrice only for trades."""
    fields = ('ClosePrice', 'SettlementPrice', 'LastPrice') if (row.get('Volume') or 0) > 0 else ('SettlementPrice', 'ClosePrice')
    for field in fields:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value, field
    return 0.0, None


def validate_prices(df):
    bad = df[(df['vol'] > 0) & (~df['close'].map(math.isfinite) | (df['close'] <= 0))]
    if not bad.empty:
        raise ValueError(f"抓取后自检失败：{len(bad)}行有成交却无有效Close/Settlement/Last价格；拒绝写入：{bad.code.head(20).tolist()}")


def _openctp_problem(d):
    if d.get("rsp_code") != 0 or not d.get("data"):
        return f"价格接口失败: {d.get('rsp_message')}"
    data = d["data"]
    if len(data) < MIN_ROWS:
        return f"OpenCTP 返回行数过少: {len(data)} < {MIN_ROWS}"
    oi_ok = sum(1 for x in data if _clean_num(x.get("OpenInterest")) is not None) / len(data)
    vol_ok = sum(1 for x in data if _clean_num(x.get("Volume")) is not None) / len(data)
    has_vol = any((_clean_num(x.get("Volume")) or 0) > 0 for x in data)
    if oi_ok < 0.8 or vol_ok < 0.8 or not has_vol:
        return f"OpenCTP 关键字段残缺: rows={len(data)} oi={oi_ok:.1%} vol={vol_ok:.1%} has_vol={has_vol}"
    return ""


def main():
    if len(sys.argv) > 1:
        day = sys.argv[1]
        td = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    else:
        td = datetime.date.today().strftime("%Y-%m-%d")
        day = td.replace("-", "")
    print(f"拉取交易日 {td} ...", flush=True)

    source = "openctp"
    d = None
    err_msg = ""
    try:
        d = fetch_prices(td)
        err_msg = _openctp_problem(d)
    except Exception as e:
        err_msg = f"OpenCTP 请求失败: {e}"

    if err_msg:
        if _can_use_quote_snapshot(td):
            print(f"! {err_msg}，切换 TqSdk 兜底...", flush=True)
            d = fetch_tqsdk_prices(td)
            source = "tqsdk"
        else:
            raise RuntimeError(err_msg)

    rows = []
    for x in d["data"]:
        ex = x["ExchangeID"]
        vol = x.get("Volume") or 0
        settle = x.get("SettlementPrice")
        c, price_source = close_price(x)
        code = lib_code(x["InstrumentID"], ex)
        if price_source == 'LastPrice':
            print(f"  价格兜底 {code}: LastPrice={c:g}，vol={vol}，ClosePrice={x.get('ClosePrice')}，SettlementPrice={settle}", flush=True)
        op, hi, lo = _clean_num(x.get("OpenPrice")), _clean_num(x.get("HighestPrice")), _clean_num(x.get("LowestPrice"))
        if vol > 0 and c > 0:
            op = op if (op is not None and op > 0) else c
            hi = hi if (hi is not None and hi > 0) else c
            lo = lo if (lo is not None and lo > 0) else c
        rows.append({
            "code": code,
            "date": day,
            "exchange": ex,
            "pre_settle": x.get("PreSettlementPrice"),
            "pre_close": x.get("PreClosePrice"),
            "open": op,
            "high": hi,
            "low": lo,
            "close": c,
            "settle": settle,
            "vol": float(vol),
            "amount": x.get("Turnover"),
            "oi": x.get("OpenInterest"),
        })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["code"]).sort_values("code")
    validate_prices(df)
    for col in ("pre_settle", "pre_close", "open", "high", "low", "close", "settle", "vol", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").round().astype("Int64")
    if len(df) < MIN_ROWS:
        raise RuntimeError(f"落盘前行数不足: {len(df)} < {MIN_ROWS}")
    oi_ok = float(df["oi"].notna().mean())
    if oi_ok < 0.8 or not (df["vol"] > 0).any():
        raise RuntimeError(f"落盘前自检失败: rows={len(df)} oi={oi_ok:.1%}")
    out = f"{OUT}/{day[:4]}/{day}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[OPT_COLS].to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    print(f"✓ 1d_opt {day} ({source}): {len(df)} 行 -> {out}", flush=True)
    print(f"  交易所: {df.exchange.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
