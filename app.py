"""
app.py — Hermes 的 Web/接入入口（对应卡片「多端接入：飞书/Web」）
================================================================
  GET  /                     → 单页前端
  GET  /api/status           → 当前模式(ai/demo)、模型、飞书是否启用
  GET  /api/analyze/stream   → SSE 流式：实时把 Agent 轨迹 + 研报/对比 推给前端
  POST /api/analyze          → 非流式 JSON（兜底/外部调用）
  POST /api/feishu/webhook   → 飞书机器人事件回调（设了 FEISHU_* 才有意义）

启动： ./run.sh   或   .venv/bin/python -m uvicorn app:app --reload
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")  # 必须在 import agent 之前，让 HERMES_MODEL / API key 生效

import agent  # noqa: E402
import data_source as ds  # noqa: E402
import requests  # noqa: E402  (akshare 依赖，已随 data_source 加了 12s 超时)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="Hermes 投研助手")
INDEX = BASE / "static" / "index.html"

FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")


def _warm() -> None:
    try:
        ds._name_table()
    except Exception:
        pass


threading.Thread(target=_warm, daemon=True).start()


class Query(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX.read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def status() -> dict:
    return {"mode": "ai" if agent.has_api_key() else "demo", "model": agent.MODEL,
            "feishu": bool(FEISHU_APP_ID and FEISHU_APP_SECRET)}


# ----- 流式（SSE）：前端 EventSource 消费 -----
def _sse(query: str):
    try:
        for ev in agent.analyze_events(query):
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
    except Exception as e:  # noqa: BLE001
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.get("/api/analyze/stream")
def analyze_stream(query: str) -> StreamingResponse:
    return StreamingResponse(
        _sse(query), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ----- 非流式 JSON（兜底 / 外部直接调用）-----
@app.post("/api/analyze")
def analyze(q: Query) -> JSONResponse:
    return JSONResponse(agent.analyze(q.query))


# --------------------------------------------------------------------------- #
# 飞书机器人接入
# --------------------------------------------------------------------------- #
def _feishu_token() -> str:
    r = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
    )
    return r.json().get("tenant_access_token", "")


def _feishu_reply(message_id: str, text: str) -> None:
    try:
        token = _feishu_token()
        if not token:
            return
        requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"msg_type": "text", "content": json.dumps({"text": text})},
        )
    except Exception:
        pass


def _result_to_text(res: dict) -> str:
    """把 analyze 结果压成飞书纯文本回复。"""
    if not res.get("ok"):
        return "⚠️ " + res.get("error", "分析失败")
    if "comparison" in res:
        c = res["comparison"]
        md = c.get("analysis_md", "")
        head = "📊 多标的对比：" + "、".join(s["name"] for s in c.get("stocks", []))
        return (head + "\n\n" + _strip_md(md))[:1600]
    r = res["report"]
    b, v, f = r.get("basics", {}), r.get("valuation", {}), r.get("financials", {})
    lines = [f"📈 {r.get('name','')}（{r.get('code','')}）"]
    if b.get("latest_price") is not None:
        lines.append(f"最新价 {b['latest_price']}　PE(TTM) {v.get('pe_ttm')}　PB {v.get('pb')}　ROE {f.get('roe_pct')}%")
    lines.append("")
    lines.append(_strip_md(r.get("analysis_md", "")))
    return "\n".join(lines)[:1600]


def _strip_md(md: str) -> str:
    out = []
    for ln in md.splitlines():
        ln = ln.replace("##", "").replace("**", "").strip()
        if ln:
            out.append(ln)
    return "\n".join(out)


def _feishu_process(text: str, message_id: str) -> None:
    res = agent.analyze(text)
    _feishu_reply(message_id, _result_to_text(res))


@app.post("/api/feishu/webhook")
async def feishu_webhook(req: Request) -> dict:
    body = await req.json()

    # 1) 回调地址验证
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge", "")}

    # 2) 消息事件（v2.0 schema）
    try:
        event = body.get("event", {})
        msg = event.get("message", {})
        content = json.loads(msg.get("content", "{}"))
        text = (content.get("text") or "").strip()
        # 去掉群里 @机器人 的占位符
        text = " ".join(t for t in text.split() if not t.startswith("@_user")).strip()
        message_id = msg.get("message_id", "")
        if text and message_id:
            # 飞书要求 3 秒内 ack，分析较慢 → 丢到后台线程处理再回复
            threading.Thread(target=_feishu_process, args=(text, message_id), daemon=True).start()
    except Exception:
        pass
    return {"code": 0}
