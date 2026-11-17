#!/usr/bin/env python3
"""update_basic_openctp.py — 每日更新合约基础信息 (openctp instruments, 不经天勤)
  options/opt_basic.parquet         : 追加新挂牌期权 (已有行不动, 保留购买数据口径)
  futures/fut_basic.parquet         : 品种级 (KQ.m@ 主连代码一行), 刷新乘数/最小变动价位/保证金率, 补新品种
  futures/fut_contracts_basic.parquet: 合约级, 按 code upsert (上市日/到期日/交割日/乘数/保证金率)
"""
import json, os, re, sys, urllib.request
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from update_1dopt_openctp import lib_code, SUFFIX

ROOT = os.environ.get("QUANT_DATA_ROOT", "./data")
OPT_B, FUT_B, FUT_CB = f"{ROOT}/options/opt_basic.parquet", f"{ROOT}/futures/fut_basic.parquet", f"{ROOT}/futures/fut_contracts_basic.parquet"
INDEX = {"IO": "OP000300.SH", "MO": "OP000852.SH", "HO": "OP000016.SH"}

def fetch(t):
    u = f"http://dict.openctp.cn/instruments?types={t}"
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read().decode())["data"]

def ymd(s): return (s or "").replace("-", "") or np.nan

def atomic(df, p):
    df.to_parquet(p + ".tmp", index=False); os.replace(p + ".tmp", p)

def update_opt():
    old = pd.read_parquet(OPT_B)
    have = set(old["code"])
    qu = old.assign(p=old["code"].str.extract(r"^([A-Z]+)")[0]).groupby("p")["quote_unit"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
    raw_opt = fetch("option")
    if not raw_opt or len(raw_opt) < 10000:
        raise RuntimeError(f"OpenCTP option instruments 异常: len={len(raw_opt) if raw_opt else 0}")
    strike_ok = sum(1 for x in raw_opt if x.get("StrikePrice") is not None and x.get("VolumeMultiple") is not None) / len(raw_opt)
    if strike_ok < 0.99:
        raise RuntimeError(f"OpenCTP option instruments 字段残缺: valid={strike_ok:.1%}")
    rows = []
    for x in raw_opt:
        ex = x["ExchangeID"]; code = lib_code(x["InstrumentID"], ex)
        if code in have:
            continue
        und = x.get("UnderlyingInstrID") or ""
        prod = re.match(r"^([A-Za-z]+)", x["InstrumentID"]).group(1).upper() if ex not in ("SSE", "SZSE") else ""
        if ex in ("SSE", "SZSE"):
            opt_code, opt_type, ex_type = f"OP{und}{SUFFIX[ex]}", "ETF期权", "欧式"
        elif ex == "CFFEX":
            opt_code, opt_type, ex_type = INDEX.get(prod, np.nan), "指数期权", "欧式"
        else:
            m = re.match(r"^([A-Za-z]+)(\d{3,4})$", und)
            ym = m.group(2) if len(m.group(2)) == 4 else "2" + m.group(2)
            opt_code, opt_type, ex_type = f"OP{m.group(1).upper()}{ym}{SUFFIX[ex]}", "期货期权", "美式"
        rows.append({"code": code, "symbol": code.split(".")[0], "exchange": ex, "name": (x.get("InstrumentName") or "").strip(),
                     "per_unit": float(x["VolumeMultiple"]), "opt_code": opt_code, "opt_type": opt_type,
                     "call_put": "C" if str(x.get("OptionsType")) == "1" else "P", "exercise_type": ex_type,
                     "exercise_price": float(x["StrikePrice"]), "opt_multiplier": float(x["VolumeMultiple"]),
                     "s_month": f"{x['DeliveryYear']}{int(x['DeliveryMonth']):02d}", "maturity_date": ymd(x.get("ExpireDate")),
                     "list_price": np.nan, "list_date": ymd(x.get("OpenDate")), "delist_date": ymd(x.get("ExpireDate")),
                     "last_edate": np.nan, "last_ddate": np.nan, "quote_unit": qu.get(prod, np.nan) if prod else "人民币元",
                     "min_price_chg": str(x["PriceTick"])})
    if rows:
        new = pd.DataFrame(rows)[old.columns].astype(old.dtypes.to_dict(), errors="ignore")
        atomic(pd.concat([old, new], ignore_index=True), OPT_B)
    print(f"opt_basic: 原 {len(old)}, 新增 {len(rows)}", flush=True)

def update_fut():
    data = fetch("futures")
    if not data or len(data) < 500:
        raise RuntimeError(f"OpenCTP futures instruments 异常: len={len(data) if data else 0}")
    mult_ok = sum(1 for x in data if x.get("VolumeMultiple") is not None) / len(data)
    margin_ok = sum(1 for x in data if x.get("LongMarginRatioByMoney") is not None) / len(data)
    if mult_ok < 0.99 or margin_ok < 0.8:
        raise RuntimeError(f"OpenCTP futures instruments 字段残缺: mult={mult_ok:.1%} margin={margin_ok:.1%}")
    c = pd.DataFrame([{"code": f"{x['ExchangeID']}.{x['InstrumentID']}", "exchange": x["ExchangeID"], "product": x["ProductID"],
                       "name": x["InstrumentName"], "multiplier": float(x["VolumeMultiple"]), "price_tick": float(x["PriceTick"]),
                       "long_margin_ratio": x["LongMarginRatioByMoney"], "short_margin_ratio": x["ShortMarginRatioByMoney"],
                       "open_date": ymd(x.get("OpenDate")), "expire_date": ymd(x.get("ExpireDate")),
                       "delivery_date": ymd(x.get("DeliveryDate")), "updated": pd.Timestamp.now().strftime("%Y%m%d")} for x in data])
    if os.path.exists(FUT_CB):
        old = pd.read_parquet(FUT_CB); c = pd.concat([old[~old["code"].isin(set(c["code"]))], c], ignore_index=True)
    atomic(c.sort_values("code"), FUT_CB)
    fb = pd.read_parquet(FUT_B)
    for col in ("per_unit", "quote_unit", "margin"):
        fb[col] = pd.to_numeric(fb[col], errors="coerce").astype("float64")
    live = c[c["updated"] == c["updated"].max()]
    prod = live.groupby(["exchange", "product"]).agg(multiplier=("multiplier", "first"), tick=("price_tick", "first"),
                                                     margin=("long_margin_ratio", "median"), name=("name", "first")).reset_index()
    add = 0
    for _, r in prod.iterrows():
        kq = f"KQ.m@{r.exchange}.{r['product']}"
        if kq in set(fb["code"]):
            i = fb.index[fb["code"] == kq]
            fb.loc[i, "per_unit"] = r.multiplier; fb.loc[i, "quote_unit"] = r.tick; fb.loc[i, "margin"] = r.margin
        else:
            fb = pd.concat([fb, pd.DataFrame([{"code": kq, "exchange": "KQ", "name": re.sub(r"\d+$", "", r["name"]) + "主连",
                                               "per_unit": r.multiplier, "quote_unit": r.tick, "margin": r.margin}])], ignore_index=True); add += 1
    for col in ("per_unit", "quote_unit", "margin"):
        fb[col] = pd.to_numeric(fb[col], errors="coerce")
    atomic(fb, FUT_B)
    print(f"fut_contracts_basic: {len(c)} 个合约 | fut_basic: {len(fb)} 个品种 (新增 {add})", flush=True)

if __name__ == "__main__":
    update_opt(); update_fut()
