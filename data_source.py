"""
data_source.py — Hermes 的「信息采集 / Gateway」层
================================================
统一封装 akshare 的 A 股数据接口，对应卡片里的「信息采集 Agent / Gateway 对接外部数据源」。

为什么是现在这套接口：实测目标机器（带本地代理 127.0.0.1:1082）下，akshare 的部分
eastmoney 主机时通时断，且 stock_individual_info_em / 现货快照接口在该 akshare 版本下不可用。
因此这里只用「实测可达」的接口，并对每个网络调用加了重试：
  - 行情：stock_zh_a_hist            （收盘价序列）
  - 财务：stock_financial_abstract    （营收/净利/ROE/毛利率）
  - 估值+市值：stock_value_em         （PE/PB/总市值/股本，一把梭）
  - 名称↔代码：stock_info_sh_name_code + stock_info_sz_name_code（SZSE 还带行业）

设计原则：
- 每个函数都容错，取不到就返回带 `_error` 的结构，**绝不让整条投研链路崩掉**。
- 返回字段形状稳定（agent 层和前端都依赖它）。akshare 接口变动只改这里的内部解析。
- 数值统一「元」原始数（前端格式化成「亿」），日期统一 'YYYY-MM-DD'。

自测：  python data_source.py 600519        （或  python data_source.py 贵州茅台）
"""
from __future__ import annotations

import datetime as _dt
import re as _re
import time as _time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import akshare as ak
import pandas as pd
import requests as _requests

# akshare 内部用 requests 且不设超时——代理抖动时一个请求能挂起 ~75 秒。
# 给所有 requests 调用强制 12 秒超时（只影响 akshare，不动 anthropic 的 httpx）。
_ADAPTER_TIMEOUT = 12
_orig_send = _requests.adapters.HTTPAdapter.send


def _send_with_timeout(self, request, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = _ADAPTER_TIMEOUT
    return _orig_send(self, request, **kwargs)


_requests.adapters.HTTPAdapter.send = _send_with_timeout

# 名称表磁盘缓存：A 股代码↔名称变化很慢，缓存到本地避免每次启动都拉一遍慢接口。
_CACHE_DIR = Path(__file__).resolve().parent / "data"
_NAMES_CSV = _CACHE_DIR / "a_share_names.csv"
_NAMES_MAX_AGE = 30 * 86400  # 30 天内的缓存直接用


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _num(v) -> Optional[float]:
    """尽量把各种形态（含逗号、None、字符串）转成 float，失败返回 None。"""
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            f = float(v)
            return None if pd.isna(f) else f
        s = str(v).replace(",", "").strip()
        if s in ("", "-", "--", "None", "nan", "NaN"):
            return None
        return float(s)
    except Exception:
        return None


def _retry(fn: Callable, tries: int = 2, delay: float = 0.6):
    """网络抖动重试。eastmoney 偶发 SSL EOF / 连接重置，重试几次大多能成功。"""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                _time.sleep(delay * (i + 1))
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 代码 / 名称 对照表（沪 + 深，进程内缓存）
# --------------------------------------------------------------------------- #
def _fetch_name_table() -> pd.DataFrame:
    """从网络拉取并合并沪深名称表。列：code, name, industry, listing_date。"""
    frames = []
    for board in ("主板A股", "科创板"):  # 沪市主板 + 科创板(688)
        try:
            sh = _retry(lambda b=board: ak.stock_info_sh_name_code(symbol=b))
            sh = sh.rename(columns={"证券代码": "code", "证券简称": "name", "上市日期": "listing_date"})
            sh["industry"] = None
            frames.append(sh[["code", "name", "industry", "listing_date"]])
        except Exception:
            pass
    try:
        sz = _retry(lambda: ak.stock_info_sz_name_code(symbol="A股列表"))
        sz = sz.rename(columns={"A股代码": "code", "A股简称": "name",
                                "A股上市日期": "listing_date", "所属行业": "industry"})
        keep = [c for c in ["code", "name", "industry", "listing_date"] if c in sz.columns]
        frames.append(sz[keep])
    except Exception:
        pass
    if not frames:
        raise RuntimeError("沪深名称表都拉不到")
    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 交易所简称里有些带空格（如「五 粮 液」「万 科Ａ」），去掉内部空白才能按用户输入匹配
    df["name"] = df["name"].astype(str).str.replace(r"\s+", "", regex=True).str.strip()
    return df.drop_duplicates(subset="code").reset_index(drop=True)


def _read_cache() -> Optional[pd.DataFrame]:
    try:
        if _NAMES_CSV.exists():
            df = pd.read_csv(_NAMES_CSV, dtype={"code": str})
            df["name"] = df["name"].astype(str).str.replace(r"\s+", "", regex=True).str.strip()
            return df
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _name_table() -> pd.DataFrame:
    """
    沪深代码↔名称表（含行业/上市日期）。优先用磁盘缓存（30 天内）；
    过期或缺失才联网刷新；联网失败则退回旧缓存（陈旧也比没有强）。
    深市表自带「所属行业」，沪市表只有上市日期。北交所未覆盖（占比极小）。
    """
    disk = _read_cache()
    fresh = False
    try:
        if _NAMES_CSV.exists():
            fresh = (_time.time() - _NAMES_CSV.stat().st_mtime) < _NAMES_MAX_AGE
    except Exception:
        fresh = False
    if disk is not None and not disk.empty and fresh:
        return disk

    try:
        df = _fetch_name_table()
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(_NAMES_CSV, index=False)
        except Exception:
            pass
        return df
    except Exception:
        if disk is not None and not disk.empty:
            return disk  # 联网失败，用陈旧缓存兜底
        raise


def resolve_symbol(query: str) -> dict:
    """
    把用户输入（公司名称或 6 位代码）解析成 {ok, code, name}。
    例： '贵州茅台' / '茅台' / '600519' -> {'ok': True, 'code': '600519', 'name': '贵州茅台'}
    """
    q = _re.sub(r"\s+", "", query or "")  # 去掉所有空白，和已去空白的名称表对齐
    if not q:
        return {"ok": False, "error": "输入为空"}

    digits = "".join(ch for ch in q if ch.isdigit())

    try:
        tbl = _name_table()
    except Exception as e:
        if len(digits) == 6:
            return {"ok": True, "code": digits, "name": digits,
                    "_warn": f"股票对照表暂不可用（{e}），按代码处理"}
        return {"ok": False, "error": f"无法加载股票对照表：{e}"}

    if len(digits) == 6:
        hit = tbl[tbl["code"] == digits]
        name = str(hit.iloc[0]["name"]) if not hit.empty else digits
        return {"ok": True, "code": digits, "name": name}

    exact = tbl[tbl["name"] == q]
    if not exact.empty:
        row = exact.iloc[0]
        return {"ok": True, "code": str(row["code"]), "name": str(row["name"])}

    hit = tbl[tbl["name"].str.contains(q, na=False, regex=False)]
    if hit.empty:
        for suf in ("股份", "集团", "公司", "科技", "股票", "控股"):
            q2 = q.replace(suf, "")
            if q2 and q2 != q:
                hit = tbl[tbl["name"].str.contains(q2, na=False, regex=False)]
                if not hit.empty:
                    break
    if hit.empty:
        return {"ok": False, "error": f"没找到与「{q}」匹配的 A 股，请换个名称或直接输入 6 位代码"}

    row = hit.iloc[0]
    out = {"ok": True, "code": str(row["code"]), "name": str(row["name"])}
    if len(hit) > 1:
        out["candidates"] = hit["name"].head(6).tolist()
    return out


def _table_row(code: str) -> dict:
    """从名称表里取 code 对应的 name/industry/listing_date（取不到返回空 dict）。"""
    try:
        tbl = _name_table()
        hit = tbl[tbl["code"] == str(code).zfill(6)]
        if not hit.empty:
            r = hit.iloc[0]
            return {"name": str(r["name"]),
                    "industry": (None if pd.isna(r.get("industry")) else r.get("industry")),
                    "listing_date": (None if pd.isna(r.get("listing_date")) else str(r.get("listing_date")))}
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------- #
# 估值 + 市值（stock_value_em：一把梭拿 PE/PB/市值/股本）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _value_em(code: str):
    """stock_value_em 的最新一行（dict），失败返回 None。按 code 缓存。"""
    try:
        df = _retry(lambda: ak.stock_value_em(symbol=str(code).zfill(6)))
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()
    except Exception:
        return None


def get_valuation(code: str) -> dict:
    """市盈率 PE / 市净率 PB / 市销率 / 市现率 等。"""
    row = _value_em(code)
    if not row:
        return {"_error": "估值数据获取失败（数据源暂不可达）"}
    return {
        "pe_ttm": _num(row.get("PE(TTM)")),
        "pe_static": _num(row.get("PE(静)")),
        "pb": _num(row.get("市净率")),
        "ps_ttm": _num(row.get("市销率")),
        "pcf": _num(row.get("市现率")),
        "peg": _num(row.get("PEG值")),
        "total_mv": _num(row.get("总市值")),
        "as_of": str(row.get("数据日期"))[:10] if row.get("数据日期") is not None else None,
    }


# --------------------------------------------------------------------------- #
# 公司基本面（名称表 + value_em 合并而成）
# --------------------------------------------------------------------------- #
def get_basics(code: str) -> dict:
    """名称、行业、上市时间、市值、股本、最新价。"""
    out: dict = {"code": str(code).zfill(6)}
    meta = _table_row(code)
    out["name"] = meta.get("name")
    out["industry"] = meta.get("industry")
    out["listing_date"] = meta.get("listing_date")

    row = _value_em(code)
    if row:
        out.update({
            "total_mv": _num(row.get("总市值")),
            "float_mv": _num(row.get("流通市值")),
            "total_share": _num(row.get("总股本")),
            "float_share": _num(row.get("流通股本")),
            "latest_price": _num(row.get("当日收盘价")),
            "latest_pct_chg": _num(row.get("当日涨跌幅")),
            "price_date": str(row.get("数据日期"))[:10] if row.get("数据日期") is not None else None,
        })
    if not row and not meta:
        out["_error"] = "基本信息获取失败（数据源暂不可达）"
    return out


# --------------------------------------------------------------------------- #
# 价格走势（画折线图用）
# --------------------------------------------------------------------------- #
def _mkt_prefix(code: str) -> str:
    c = str(code).zfill(6)
    if c[0] in "69":
        return "sh"
    if c[0] in "8 4":
        return "bj"
    return "sz"


def _finalize_price(df: pd.DataFrame, days: int) -> dict:
    """df 需含列 date/close（high/low/pct_chg 可选），统一构造返回结构。"""
    df = df.tail(days).reset_index(drop=True)
    series = [{"date": str(d)[:10], "close": _num(c)}
              for d, c in zip(df["date"], df["close"])]
    first_close = _num(df.iloc[0]["close"])
    last = df.iloc[-1]
    last_close = _num(last["close"])
    if "pct_chg" in df.columns:
        last_pct = _num(last["pct_chg"])
    elif len(df) > 1 and _num(df.iloc[-2]["close"]):
        last_pct = round((last_close / _num(df.iloc[-2]["close"]) - 1) * 100, 2)
    else:
        last_pct = None
    return {
        "series": series,
        "latest_price": last_close,
        "latest_date": str(last["date"])[:10],
        "latest_pct_chg": last_pct,
        "period_days": len(series),
        "period_return_pct": round((last_close / first_close - 1) * 100, 2)
            if (first_close and last_close) else None,
        "high": _num(df["high"].max()) if "high" in df.columns else _num(df["close"].max()),
        "low": _num(df["low"].min()) if "low" in df.columns else _num(df["close"].min()),
    }


def _price_eastmoney(code: str, days: int) -> dict:
    end = _dt.date.today()
    start = end - _dt.timedelta(days=int(days * 1.7) + 15)
    df = ak.stock_zh_a_hist(
        symbol=str(code).zfill(6), period="daily",
        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )  # 单次尝试，失败由 get_price_history 切换到 sina
    if df is None or df.empty:
        raise RuntimeError("eastmoney 空数据")
    df = df.rename(columns={"日期": "date", "收盘": "close", "最高": "high",
                            "最低": "low", "涨跌幅": "pct_chg"})
    return _finalize_price(df, days)


def _price_sina(code: str, days: int) -> dict:
    sym = _mkt_prefix(code) + str(code).zfill(6)
    df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")  # 单次尝试
    if df is None or df.empty:
        raise RuntimeError("sina 空数据")
    return _finalize_price(df, days)


def get_price_history(code: str, days: int = 180) -> dict:
    """近 N 个交易日的收盘价序列 + 区间涨跌幅、最新价等。
    多数据源容错：eastmoney 主，sina 备（不同主机，代理抖动时互为兜底）。"""
    last_err = None
    for fn in (_price_eastmoney, _price_sina):
        try:
            return fn(code, days)
        except Exception as e:  # noqa: BLE001
            last_err = e
    return {"_error": f"价格数据获取失败（多源均不可达）：{last_err}"}


# --------------------------------------------------------------------------- #
# 财务数据
# --------------------------------------------------------------------------- #
def get_financials(code: str) -> dict:
    """营收 / 净利润趋势、同比、ROE、毛利率等关键指标。"""
    try:
        df = _retry(lambda: ak.stock_financial_abstract(symbol=str(code).zfill(6)))
    except Exception as e:
        return {"_error": f"财务数据获取失败：{e}"}

    try:
        ind_col = "指标" if "指标" in df.columns else df.columns[1]
        meta_cols = [c for c in df.columns if c in ("选项", "指标")]
        date_cols = [c for c in df.columns if c not in meta_cols and c != ind_col]

        def _as_int(c):
            try:
                return int("".join(ch for ch in str(c) if ch.isdigit())[:8])
            except Exception:
                return -1

        recent = sorted(date_cols, key=_as_int)[-8:]

        def row_for(*keys):
            for k in keys:
                m = df[df[ind_col].astype(str).str.contains(k, na=False, regex=False)]
                if not m.empty:
                    return m.iloc[0]
            return None

        def trend(*keys):
            r = row_for(*keys)
            if r is None:
                return []
            out = []
            for c in recent:
                v = _num(r.get(c))
                if v is not None:
                    out.append({"period": str(c)[:8], "value": v})
            return out

        def yoy(series):
            if len(series) >= 5 and series[-5]["value"]:
                return round((series[-1]["value"] / series[-5]["value"] - 1) * 100, 2)
            if len(series) >= 2 and series[-2]["value"]:
                return round((series[-1]["value"] / series[-2]["value"] - 1) * 100, 2)
            return None

        revenue = trend("营业总收入", "营业收入")
        net = trend("归母净利润", "净利润")
        roe_row = row_for("净资产收益率(加权)", "加权净资产收益率", "净资产收益率", "ROE")
        gm_row = row_for("销售毛利率", "毛利率")

        return {
            "revenue_trend": revenue,
            "net_profit_trend": net,
            "latest_revenue": revenue[-1]["value"] if revenue else None,
            "latest_net_profit": net[-1]["value"] if net else None,
            "latest_period": revenue[-1]["period"] if revenue else (net[-1]["period"] if net else None),
            "revenue_yoy_pct": yoy(revenue),
            "net_profit_yoy_pct": yoy(net),
            "roe_pct": _num(roe_row.get(recent[-1])) if (roe_row is not None and recent) else None,
            "gross_margin_pct": _num(gm_row.get(recent[-1])) if (gm_row is not None and recent) else None,
        }
    except Exception as e:
        return {"_error": f"财务数据解析失败：{e}"}


# --------------------------------------------------------------------------- #
# 舆情 / 新闻公告（对应卡片「信息采集 Agent 抓公告与舆情」）
# --------------------------------------------------------------------------- #
def get_news(code: str, limit: int = 8) -> dict:
    """个股最近的新闻与公告（来源：eastmoney）。"""
    try:
        df = _retry(lambda: ak.stock_news_em(symbol=str(code).zfill(6)))
        if df is None or df.empty:
            return {"items": []}
        df = df.head(limit)
        items = []
        for _, r in df.iterrows():
            items.append({
                "title": str(r.get("新闻标题", "")).strip(),
                "time": str(r.get("发布时间", ""))[:16],
                "source": str(r.get("文章来源", "")).strip(),
                "url": str(r.get("新闻链接", "")).strip(),
            })
        return {"items": items}
    except Exception as e:
        return {"_error": f"舆情获取失败：{e}", "items": []}


# --------------------------------------------------------------------------- #
# 一站式采集
# --------------------------------------------------------------------------- #
def collect_all(code: str, name: str = "") -> dict:
    """把一支股票的全部数据收齐，给 demo 模式 / 前端直接用。
    四个网络操作并行跑（价格 / 财务 / 估值市值 / 舆情），墙钟时间≈最慢的那个而非求和。"""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_price = ex.submit(get_price_history, code)
        f_fin = ex.submit(get_financials, code)
        f_val = ex.submit(_value_em, code)  # 预热缓存，basics/valuation 随后直接读
        f_news = ex.submit(get_news, code)
        price = f_price.result()
        fin = f_fin.result()
        f_val.result()
        news = f_news.result()
    return {
        "code": str(code).zfill(6),
        "name": name,
        "basics": get_basics(code),      # 读已预热的 _value_em 缓存
        "price": price,
        "financials": fin,
        "valuation": get_valuation(code),  # 读已预热的 _value_em 缓存
        "news": news,
    }


# --------------------------------------------------------------------------- #
# 自测入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "600519"
    print(f"=== 解析「{q}」 ===")
    r = resolve_symbol(q)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if not r.get("ok"):
        sys.exit(1)
    code = r["code"]
    for label, fn in [("basics", get_basics), ("price", get_price_history),
                      ("financials", get_financials), ("valuation", get_valuation)]:
        print(f"\n=== {label} ===")
        out = fn(code)
        if label == "price" and "series" in out:
            out = {**out, "series": f"<{len(out['series'])} 个数据点>"}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
