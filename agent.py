"""
agent.py — Hermes 的「数据分析 Agent」大脑
=========================================
对应卡片「从意图识别到研报生成的全链路 Agentic 投研体验」。

核心是一个**事件生成器** `analyze_events(query)`，按发生顺序 yield：
    {"type":"step", "summary": "..."}                 # Agent 轨迹的一步（前端实时滚动）
    {"type":"report", "mode": "ai"|"demo", "report": {...}}        # 单标的研报
    {"type":"comparison", "mode": ..., "comparison": {...}}        # 多标的对比
    {"type":"error", "error": "..."}
    {"type":"done"}
SSE 端点直接把它转成事件流；`analyze()` 则把它收敛成最终 dict（给飞书/非流式用）。

意图识别：
- 单标的：'分析下贵州茅台' / '600519'
- 多标的对比：'对比 茅台 和 五粮液' / '茅台 vs 五粮液 vs 泸州老窖'

两种分析模式（对外形状一致）：
- AI 模式（有 ANTHROPIC_API_KEY）：Claude 在真正的 tool-calling 循环里取数+写研报。
- 演示模式（无 key）：确定性流水线 + 模板，数据依旧真实。

模型默认 claude-opus-4-8，可用 HERMES_MODEL / HERMES_EFFORT 覆盖。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import Iterator, Optional

import backtest as bt
import data_source as ds

MODEL = os.environ.get("HERMES_MODEL", "claude-opus-4-8")
EFFORT = os.environ.get("HERMES_EFFORT", "medium")
MAX_TOOL_ROUNDS = 14


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# --------------------------------------------------------------------------- #
# 工具定义（给 Claude）+ 服务端执行（真正调用 akshare）
# --------------------------------------------------------------------------- #
def _tool(name, desc):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object",
                             "properties": {"code": {"type": "string", "description": "6 位股票代码"}},
                             "required": ["code"]}}


TOOLS = [
    {"name": "resolve_symbol",
     "description": "把公司名称或代码解析成标准 6 位 A 股代码和名称。任何分析的第一步。",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string", "description": "公司名称或6位代码，如『贵州茅台』或『600519』"}},
                      "required": ["query"]}},
    _tool("get_basics", "公司基本面：名称、行业、上市时间、总市值、流通市值、总股本、最新价。"),
    _tool("get_price_history", "近半年（约180交易日）收盘价序列、区间涨跌幅、最高/最低价。"),
    _tool("get_financials", "营收/归母净利润多期趋势、同比增速、最新 ROE 与销售毛利率。"),
    _tool("get_valuation", "估值：PE(TTM/静)、PB、市销率、市现率、PEG、总市值。"),
    _tool("get_news", "个股最近的新闻与公告（舆情）。用于判断近期催化与风险事件。"),
    {"name": "run_backtest",
     "description": "双均线策略历史回测（近三年）：快线上穿慢线持有、下穿空仓，信号次日生效、含交易成本。"
                    "返回策略与同期买入持有基准的年化收益/最大回撤/夏普/胜率对比。用户提到回测、策略、均线时使用。",
     "input_schema": {"type": "object",
                      "properties": {"code": {"type": "string", "description": "6 位股票代码"},
                                     "fast": {"type": "integer", "description": "快线窗口（日），默认 20"},
                                     "slow": {"type": "integer", "description": "慢线窗口（日），默认 60"}},
                      "required": ["code"]}},
]

SYSTEM_PROMPT = """你是 Hermes，一个专业、严谨的 A 股投研助手。

工作方式：
1. 先用 resolve_symbol 确定标的（拿到 6 位代码）。
2. 调用数据工具获取基本面、价格、财务、估值、舆情（都是真实市场数据）。
   若用户要求回测/策略验证，调用 run_backtest（双均线）；解读回测必须与买入持有基准对比，
   并说明单一参数存在过拟合风险，此时研报增加「## 策略回测」小节。
3. 写一份结构化中文迷你研报（Markdown），包含：
   ## 一句话结论
   ## 公司概览
   ## 估值分析
   ## 成长与盈利
   ## 价格走势
   ## 舆情速览
   ## 风险提示

要求：
- 判断必须基于工具返回的真实数字，文中点出关键数字（PE、ROE、营收同比等）。
- 客观中立，这是投研分析不是荐股；不要给"买入/卖出"指令，可讨论估值偏高/偏低、成长快/慢。
- 舆情速览结合 get_news 的标题，归纳近期关注点（催化/风险），不要逐条罗列。
- 金额用「亿元」更直观（工具给的是元）。简洁、专业、可读。
- 必须含风险提示，并声明"本内容由 AI 生成，仅供研究参考，不构成投资建议"。
"""

COMPARE_SYSTEM = """你是 Hermes 投研助手。下面给你几支 A 股的关键指标 JSON。
写一段简洁的中文对比分析（Markdown，300字内），包含：
- 估值对比（谁更贵/更便宜，看 PE/PB）
- 成长与盈利对比（营收同比、ROE、毛利率）
- 近半年价格表现对比
- 一句话总结各自特点
客观中立，基于数字，不给买卖指令。结尾一句风险提示。"""


def _run_tool(name: str, args: dict, collected: dict) -> dict:
    if name == "resolve_symbol":
        r = ds.resolve_symbol(args.get("query", ""))
        if r.get("ok"):
            collected["code"] = r["code"]
            collected["name"] = r["name"]
        return r
    code = args.get("code") or collected.get("code")
    if not code:
        return {"_error": "缺少股票代码，请先 resolve_symbol"}
    if name == "run_backtest":
        out = bt.run_backtest(code, args.get("fast") or bt.DEFAULT_FAST,
                              args.get("slow") or bt.DEFAULT_SLOW)
        collected["backtest"] = out
        return out
    fn = {"get_basics": ds.get_basics, "get_price_history": ds.get_price_history,
          "get_financials": ds.get_financials, "get_valuation": ds.get_valuation,
          "get_news": ds.get_news}.get(name)
    if not fn:
        return {"_error": f"未知工具 {name}"}
    out = fn(code)
    key = {"get_basics": "basics", "get_price_history": "price", "get_financials": "financials",
           "get_valuation": "valuation", "get_news": "news"}[name]
    collected[key] = out
    return out


def _summarize_tool(name: str, out: dict) -> str:
    if isinstance(out, dict) and out.get("_error") and name != "get_news":
        return f"⚠️ {out['_error']}"
    if name == "resolve_symbol":
        return f"识别标的 → {out.get('name')}（{out.get('code')}）" if out.get("ok") else out.get("error", "未识别")
    if name == "get_basics":
        return f"取基本面 → 总市值 {_yi(out.get('total_mv'))}，最新价 {out.get('latest_price')}"
    if name == "get_price_history":
        return f"取价格 → 近{out.get('period_days')}日，区间 {out.get('period_return_pct')}%"
    if name == "get_financials":
        return f"取财务 → 营收同比 {out.get('revenue_yoy_pct')}%，ROE {out.get('roe_pct')}%"
    if name == "get_valuation":
        return f"取估值 → PE(TTM) {out.get('pe_ttm')}，PB {out.get('pb')}"
    if name == "get_news":
        return f"采集舆情 → {len(out.get('items', []))} 条近期新闻/公告"
    if name == "run_backtest":
        m = out.get("metrics", {})
        return (f"回测 {out.get('strategy')} → 年化 {m.get('annual_return_pct')}%，"
                f"最大回撤 {m.get('max_drawdown_pct')}%，夏普 {m.get('sharpe')}，"
                f"对买入持有超额 {out.get('excess_return_pct')}%")
    return "完成"


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _yi(v) -> str:
    try:
        return f"{float(v) / 1e8:,.2f} 亿元"
    except Exception:
        return "—"


# --------------------------------------------------------------------------- #
# 意图识别：把 query 拆成 1 个或多个标的
# --------------------------------------------------------------------------- #
_SEP = re.compile(r"\s*(?:对比|相比|比较|和|与|跟|及|vs\.?|VS\.?|、|，|,|/)\s*")
_LEAD = re.compile(r"^(?:请?帮我?)?(?:对比|比较|分析|看看|看一下|研究)(?:一下|下)?\s*")
_BT_RE = re.compile(r"回测|双均线|均线策略")
_BT_PARAM_RE = re.compile(r"(\d{1,3})\s*[/、,，]\s*(\d{1,3})")  # 如「20/60」；要求分隔符，避免误吞 6 位股票代码


def _parse_backtest_intent(query: str) -> Optional[dict]:
    """识别「回测 XX」类意图。返回 {target, fast, slow}，非回测意图返回 None。"""
    if not _BT_RE.search(query or ""):
        return None
    fast, slow = bt.DEFAULT_FAST, bt.DEFAULT_SLOW
    q = query
    m = _BT_PARAM_RE.search(q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0 and b > 0 and a != b:
            fast, slow = min(a, b), max(a, b)
        q = q.replace(m.group(0), " ")
    q = re.sub(r"用(?=\s*(?:双均线|均线|策略|回测))", " ", q)  # 「用双均线回测」的「用」；不误伤「用友网络」
    q = re.sub(r"回测|双均线|均线策略|策略|均线|一下|帮我|请|麻烦", " ", q).strip()
    q = _LEAD.sub("", q)
    target = _SEP.split(q)[0].strip() if q.strip() else ""
    return {"target": target or query, "fast": fast, "slow": slow}


def _split_targets(query: str) -> list[str]:
    q = (query or "").strip()
    parts = [p for p in _SEP.split(q) if p.strip()]
    cleaned = []
    for p in parts:
        p = _LEAD.sub("", p).strip()
        if p:
            cleaned.append(p)
    return cleaned or [q]


# --------------------------------------------------------------------------- #
# 事件生成器（核心）
# --------------------------------------------------------------------------- #
def analyze_events(query: str) -> Iterator[dict]:
    query = (query or "").strip()
    if not query:
        yield {"type": "error", "error": "请输入要分析的股票名称或代码"}
        return
    try:
        bt_intent = _parse_backtest_intent(query)
        if bt_intent:
            r = ds.resolve_symbol(bt_intent["target"])
            if not r.get("ok"):
                yield {"type": "error",
                       "error": r.get("error", f"没找到「{bt_intent['target']}」对应的 A 股")}
                return
            yield from _events_backtest(query, r, bt_intent["fast"], bt_intent["slow"])
            yield {"type": "done"}
            return

        targets = _split_targets(query)
        resolved, seen = [], set()
        for t in targets:
            r = ds.resolve_symbol(t)
            if r.get("ok") and r["code"] not in seen:
                seen.add(r["code"])
                resolved.append(r)
        if not resolved:
            yield {"type": "error", "error": f"没找到「{query}」对应的 A 股，请换个名称或直接输入 6 位代码"}
            return

        if len(resolved) >= 2:
            yield from _events_compare(query, resolved)
        elif has_api_key():
            yield from _events_single_ai(query)
        else:
            yield from _events_single_demo(query, resolved[0])
        yield {"type": "done"}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "error": f"分析失败：{e}"}


def _events_single_demo(query: str, r: dict) -> Iterator[dict]:
    code, name = r["code"], r["name"]
    yield {"type": "step", "summary": f"识别标的 → {name}（{code}）"}
    yield {"type": "step", "summary": "通过 Gateway 并行采集 行情/财务/估值/舆情…"}
    data = ds.collect_all(code, name)
    for key, label in [("basics", "get_basics"), ("valuation", "get_valuation"),
                       ("financials", "get_financials"), ("news", "get_news"),
                       ("price", "get_price_history")]:
        yield {"type": "step", "summary": _summarize_tool(label, data[key])}
    analysis = _template_analysis(name, code, data)
    report = _assemble_report(query, {**data, "code": code, "name": name}, analysis)
    yield {"type": "report", "mode": "demo", "report": report}


def _events_single_ai(query: str) -> Iterator[dict]:
    import anthropic

    client = anthropic.Anthropic()
    collected: dict = {}
    messages = [{"role": "user", "content": f"请分析这支 A 股：{query}"}]
    final_text = ""

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=MODEL, max_tokens=16000,
            thinking={"type": "adaptive"}, output_config={"effort": EFFORT},
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        if resp.stop_reason == "refusal":
            yield {"type": "error", "error": "模型出于安全原因拒绝了本次请求，请换个问法。"}
            return
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        text_now = "".join(b.text for b in resp.content if b.type == "text")
        if text_now:
            final_text = text_now
        if resp.stop_reason != "tool_use" or not tool_uses:
            break
        results = []
        for tu in tool_uses:
            out = _run_tool(tu.name, tu.input or {}, collected)
            yield {"type": "step", "summary": _summarize_tool(tu.name, out)}
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(out, ensure_ascii=False, default=str),
                            "is_error": bool(isinstance(out, dict) and out.get("_error"))})
        messages.append({"role": "user", "content": results})

    report = _assemble_report(query, collected, final_text)
    yield {"type": "report", "mode": "ai", "report": report}


def _events_compare(query: str, resolved: list[dict]) -> Iterator[dict]:
    names = "、".join(r["name"] for r in resolved)
    yield {"type": "step", "summary": f"识别 {len(resolved)} 个标的 → {names}"}
    stocks = []
    for r in resolved:
        code, name = r["code"], r["name"]
        data = ds.collect_all(code, name)
        v, b = data["valuation"], data["basics"]
        yield {"type": "step",
               "summary": f"取数 {name} → 市值 {_yi(v.get('total_mv') or b.get('total_mv'))}，PE {v.get('pe_ttm')}"}
        stocks.append({"code": code, "name": name, "basics": b, "valuation": v,
                       "financials": data["financials"], "price": data["price"]})

    mode = "ai" if has_api_key() else "demo"
    yield {"type": "step", "summary": "生成对比分析…"}
    analysis = _ai_compare(stocks) if has_api_key() else _template_compare(stocks)
    yield {"type": "comparison", "mode": mode,
           "comparison": {"query": query, "stocks": stocks, "analysis_md": analysis,
                          "generated_at": _now()}}


def _events_backtest(query: str, r: dict, fast: int, slow: int) -> Iterator[dict]:
    """回测意图的确定性流水线：回测由代码裁决，AI（若有 key）只负责解读。"""
    code, name = r["code"], r["name"]
    yield {"type": "step", "summary": f"识别标的 → {name}（{code}）"}
    yield {"type": "step", "summary": f"取近三年复权行情，回测 双均线 MA{fast}/MA{slow}（信号次日生效，含交易成本）…"}
    result = bt.run_backtest(code, fast, slow)
    if result.get("_error"):
        yield {"type": "error", "error": f"回测失败：{result['_error']}"}
        return
    yield {"type": "step", "summary": _summarize_tool("run_backtest", result)}

    md = bt.format_report(result, name)
    mode = "demo"
    if has_api_key():
        yield {"type": "step", "summary": "生成 AI 解读…"}
        interp = _ai_backtest_interpret(name, result)
        if interp:
            md = interp + "\n\n---\n\n" + md
            mode = "ai"

    yield {"type": "step", "summary": "补齐基本面/估值/舆情，组装报告…"}
    data = ds.collect_all(code, name)
    report = _assemble_report(query, {**data, "code": code, "name": name}, md)
    report["backtest"] = result
    yield {"type": "report", "mode": mode, "report": report}


BACKTEST_SYSTEM = """你是 Hermes 投研助手。下面是一份双均线策略回测结果 JSON——
由确定性代码基于真实历史行情计算，数字不可更改，你的任务只是解读。
写一段简洁中文解读（Markdown，250 字内）：
- 策略与买入持有基准的对比结论（看年化、最大回撤、夏普，谁好、好在哪）
- 胜率与交易次数说明了什么
- 局限性：单一参数事后选择存在过拟合风险、样本区间有限
客观中立，不给买卖指令；不要复述表格数字本身，重在解读；结尾一句风险提示。"""


def _ai_backtest_interpret(name: str, result: dict) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL, max_tokens=2000,
            thinking={"type": "adaptive"}, output_config={"effort": EFFORT},
            system=BACKTEST_SYSTEM,
            messages=[{"role": "user", "content": f"标的：{name}\n" + json.dumps(result, ensure_ascii=False)}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 报告组装 + 模板（演示模式 / AI 数据兜底）
# --------------------------------------------------------------------------- #
def _assemble_report(query: str, collected: dict, analysis_md: str) -> dict:
    code = collected.get("code", "")
    name = collected.get("name", "")
    if code:  # AI 模式若 Claude 漏取某块，这里补齐，保证前端完整
        for key, fn in [("basics", ds.get_basics), ("price", ds.get_price_history),
                        ("financials", ds.get_financials), ("valuation", ds.get_valuation),
                        ("news", ds.get_news)]:
            if key not in collected:
                collected[key] = fn(code)
        if not name:
            name = collected.get("basics", {}).get("name") or code
    return {
        "query": query, "code": code, "name": name,
        "basics": collected.get("basics", {}),
        "price": collected.get("price", {}),
        "financials": collected.get("financials", {}),
        "valuation": collected.get("valuation", {}),
        "news": collected.get("news", {"items": []}),
        "analysis_md": analysis_md or "（未生成分析正文）",
        "generated_at": _now(),
    }


def _template_analysis(name: str, code: str, data: dict) -> str:
    b, v, f, p = data["basics"], data["valuation"], data["financials"], data["price"]
    news = data.get("news", {})
    L = ["## 一句话结论"]
    bits = []
    if v.get("pe_ttm") is not None:
        bits.append(f"PE(TTM) {v['pe_ttm']:.1f}" + (f"、PB {v['pb']:.2f}" if v.get("pb") else ""))
    if f.get("revenue_yoy_pct") is not None:
        bits.append(f"营收同比 {f['revenue_yoy_pct']:+.1f}%")
    if f.get("roe_pct") is not None:
        bits.append(f"ROE {f['roe_pct']:.1f}%")
    L.append(f"{name}（{code}）：" + ("；".join(bits) + "。" if bits else "已汇总关键财务与估值数据，详见下文。"))

    L.append("\n## 公司概览")
    if b.get("industry"):
        L.append(f"- 所属行业：{b['industry']}")
    if b.get("total_mv") is not None:
        L.append(f"- 总市值：约 {_yi(b['total_mv'])}")
    if b.get("listing_date"):
        L.append(f"- 上市时间：{b['listing_date']}")
    if b.get("latest_price") is not None:
        chg = b.get("latest_pct_chg")
        L.append(f"- 最新价：{b['latest_price']}" + (f"（当日 {chg:+.2f}%）" if chg is not None else ""))

    L.append("\n## 估值分析")
    if v.get("_error"):
        L.append("- 估值数据暂不可用。")
    else:
        if v.get("pe_ttm") is not None:
            L.append(f"- 市盈率 PE(TTM)：{v['pe_ttm']:.2f}" + (f"（静态 {v['pe_static']:.2f}）" if v.get("pe_static") else ""))
        if v.get("pb") is not None:
            L.append(f"- 市净率 PB：{v['pb']:.2f}")
        if v.get("ps_ttm") is not None:
            L.append(f"- 市销率 PS(TTM)：{v['ps_ttm']:.2f}")
        L.append("- 解读：估值高低需结合行业平均与公司历史区间判断，单一倍数不宜直接定论。")

    L.append("\n## 成长与盈利")
    if f.get("_error"):
        L.append("- 财务数据暂不可用。")
    else:
        if f.get("latest_revenue") is not None:
            L.append(f"- 最新一期营业收入：{_yi(f['latest_revenue'])}" + (f"，同比 {f['revenue_yoy_pct']:+.1f}%" if f.get("revenue_yoy_pct") is not None else ""))
        if f.get("latest_net_profit") is not None:
            L.append(f"- 最新一期归母净利润：{_yi(f['latest_net_profit'])}" + (f"，同比 {f['net_profit_yoy_pct']:+.1f}%" if f.get("net_profit_yoy_pct") is not None else ""))
        if f.get("roe_pct") is not None:
            L.append(f"- 净资产收益率 ROE：{f['roe_pct']:.2f}%")
        if f.get("gross_margin_pct") is not None:
            L.append(f"- 销售毛利率：{f['gross_margin_pct']:.2f}%")

    L.append("\n## 价格走势")
    if p.get("_error"):
        L.append("- 价格数据暂不可用。")
    else:
        L.append(f"- 近 {p.get('period_days')} 个交易日区间收益：{p.get('period_return_pct'):+.2f}%")
        if p.get("high") is not None and p.get("low") is not None:
            L.append(f"- 区间最高 {p['high']} / 最低 {p['low']}，最新 {p.get('latest_price')}（{p.get('latest_date')}）")

    L.append("\n## 舆情速览")
    items = news.get("items", [])
    if items:
        for it in items[:5]:
            L.append(f"- {it.get('time','')} {it.get('title','')}")
    else:
        L.append("- 近期暂无抓取到的新闻/公告。")

    L.append("\n## 风险提示")
    L.append("- 以上为基于公开数据的量化汇总，未覆盖行业景气度、政策、竞争格局等定性因素。")
    L.append("- **本内容由程序自动生成，仅供研究参考，不构成任何投资建议。**（配置 Claude API Key 后可获得 AI 深度分析）")
    return "\n".join(L)


def _compact_facts(stocks: list[dict]) -> str:
    rows = []
    for s in stocks:
        v, f, b, p = s["valuation"], s["financials"], s["basics"], s["price"]
        rows.append({
            "名称": s["name"], "代码": s["code"], "行业": b.get("industry"),
            "总市值(亿)": round((v.get("total_mv") or b.get("total_mv") or 0) / 1e8, 1),
            "PE_TTM": v.get("pe_ttm"), "PB": v.get("pb"), "PS_TTM": v.get("ps_ttm"),
            "ROE_%": f.get("roe_pct"), "毛利率_%": f.get("gross_margin_pct"),
            "营收同比_%": f.get("revenue_yoy_pct"), "净利同比_%": f.get("net_profit_yoy_pct"),
            "近半年涨幅_%": p.get("period_return_pct"),
        })
    return json.dumps(rows, ensure_ascii=False)


def _ai_compare(stocks: list[dict]) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL, max_tokens=4000,
            thinking={"type": "adaptive"}, output_config={"effort": EFFORT},
            system=COMPARE_SYSTEM,
            messages=[{"role": "user", "content": _compact_facts(stocks)}],
        )
        txt = "".join(b.text for b in resp.content if b.type == "text")
        return txt or _template_compare(stocks)
    except Exception:
        return _template_compare(stocks)


def _template_compare(stocks: list[dict]) -> str:
    def g(s, *path):
        cur = s
        for k in path:
            cur = (cur or {}).get(k)
        return cur

    L = ["## 对比分析", "", "| 指标 | " + " | ".join(s["name"] for s in stocks) + " |",
         "|---|" + "---|" * len(stocks)]
    rowdefs = [
        ("总市值", lambda s: _yi(g(s, "valuation", "total_mv") or g(s, "basics", "total_mv"))),
        ("PE(TTM)", lambda s: _fmt(g(s, "valuation", "pe_ttm"))),
        ("PB", lambda s: _fmt(g(s, "valuation", "pb"))),
        ("ROE %", lambda s: _fmt(g(s, "financials", "roe_pct"))),
        ("毛利率 %", lambda s: _fmt(g(s, "financials", "gross_margin_pct"))),
        ("营收同比 %", lambda s: _fmt(g(s, "financials", "revenue_yoy_pct"))),
        ("近半年涨幅 %", lambda s: _fmt(g(s, "price", "period_return_pct"))),
    ]
    for label, fn in rowdefs:
        L.append(f"| {label} | " + " | ".join(str(fn(s)) for s in stocks) + " |")

    # 简单结论
    def best(metric_path, reverse=False):
        vals = [(s["name"], g(s, *metric_path)) for s in stocks]
        vals = [(n, x) for n, x in vals if isinstance(x, (int, float))]
        if not vals:
            return None
        return sorted(vals, key=lambda t: t[1], reverse=reverse)[0][0]

    L.append("")
    cheap = best(("valuation", "pe_ttm"))
    roe = best(("financials", "roe_pct"), reverse=True)
    grow = best(("financials", "revenue_yoy_pct"), reverse=True)
    if cheap:
        L.append(f"- 估值最低（PE 最小）：**{cheap}**")
    if roe:
        L.append(f"- 盈利能力最强（ROE 最高）：**{roe}**")
    if grow:
        L.append(f"- 成长最快（营收同比最高）：**{grow}**")
    L.append("\n**本内容由程序自动生成，仅供研究参考，不构成投资建议。**（配置 Claude API Key 后可获得 AI 对比分析）")
    return "\n".join(L)


def _fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


# --------------------------------------------------------------------------- #
# 非流式入口（飞书 / 兜底用）
# --------------------------------------------------------------------------- #
def analyze(query: str) -> dict:
    steps: list[dict] = []
    out: dict = {"ok": False, "error": "无结果"}
    try:
        for ev in analyze_events(query):
            t = ev.get("type")
            if t == "step":
                steps.append({"summary": ev["summary"]})
            elif t == "report":
                out = {"ok": True, "mode": ev["mode"], "report": ev["report"]}
            elif t == "comparison":
                out = {"ok": True, "mode": ev["mode"], "comparison": ev["comparison"]}
            elif t == "error":
                return {"ok": False, "error": ev["error"]}
        out["steps"] = steps
        return out
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"分析失败：{e}"}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "贵州茅台"
    print(f"模式：{'AI' if has_api_key() else '演示'}  模型：{MODEL}\n")
    for ev in analyze_events(q):
        t = ev.get("type")
        if t == "step":
            print(" •", ev["summary"])
        elif t == "report":
            print("\n--- REPORT ---\n" + ev["report"]["analysis_md"])
        elif t == "comparison":
            print("\n--- COMPARISON ---\n" + ev["comparison"]["analysis_md"])
        elif t == "error":
            print("ERROR:", ev["error"])
