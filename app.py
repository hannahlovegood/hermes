"""
app.py — Hermes 的 Web/接入入口（对应卡片「多端接入：飞书/Web」）
================================================================
  GET  /                     → 单页前端
  GET  /api/status           → 当前模式(ai/demo)、模型、飞书是否启用
  GET  /api/analyze/stream   → SSE 流式：实时把 Agent 轨迹 + 研报/对比 推给前端
  POST /api/analyze          → 非流式 JSON（兜底/外部调用）
  POST /api/feishu/webhook   → 飞书机器人事件回调（设了 FEISHU_* 才有意义）
  GET  /api/wechat/webhook   → 微信公众号服务器配置校验（设了 WECHAT_* 才有意义）
  POST /api/wechat/webhook   → 微信公众号消息回调（测试号即可）

启动： ./run.sh   或   .venv/bin/python -m uvicorn app:app --reload
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import xml.etree.ElementTree as ET
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

WECHAT_TOKEN = os.environ.get("WECHAT_TOKEN", "")
WECHAT_APPID = os.environ.get("WECHAT_APPID", "")
WECHAT_APPSECRET = os.environ.get("WECHAT_APPSECRET", "")


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
            "feishu": bool(FEISHU_APP_ID and FEISHU_APP_SECRET),
            "wechat": bool(WECHAT_TOKEN and WECHAT_APPID and WECHAT_APPSECRET)}


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
    """把 analyze 结果压成纯文本回复（飞书 / 微信共用）。"""
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


# --------------------------------------------------------------------------- #
# 微信公众号接入（测试号即可，明文模式）
# --------------------------------------------------------------------------- #
# 流程：收到文本消息 → 5 秒内被动回一句「分析中」→ 后台线程跑 analyze →
#       通过客服消息接口把研报推回去（测试号自带该权限，不受 5 秒限制）。
_wechat_tok: dict = {"token": "", "exp": 0.0}
_wechat_tok_lock = threading.Lock()


def _wechat_sign_ok(signature: str, timestamp: str, nonce: str) -> bool:
    """微信服务器每次请求都带签名：sha1(sorted(token, timestamp, nonce))。"""
    if not WECHAT_TOKEN:
        return False
    raw = "".join(sorted([WECHAT_TOKEN, timestamp, nonce])).encode()
    return hmac.compare_digest(hashlib.sha1(raw).hexdigest(), signature)


def _wechat_access_token(force: bool = False) -> str:
    with _wechat_tok_lock:
        if not force and _wechat_tok["token"] and time.time() < _wechat_tok["exp"]:
            return _wechat_tok["token"]
        try:
            r = requests.get(
                "https://api.weixin.qq.com/cgi-bin/token",
                params={"grant_type": "client_credential",
                        "appid": WECHAT_APPID, "secret": WECHAT_APPSECRET},
            ).json()
            _wechat_tok["token"] = r.get("access_token", "")
            # 官方 7200s 有效，提前 5 分钟刷新
            _wechat_tok["exp"] = time.time() + int(r.get("expires_in", 7200)) - 300
        except Exception:
            _wechat_tok["token"] = ""
        return _wechat_tok["token"]


def _wechat_push(openid: str, text: str) -> None:
    """客服消息接口推文本。单条 2048 字节上限 → 按 600 字切块，最多 3 块。"""
    chunks = [text[i:i + 600] for i in range(0, len(text), 600)][:3]
    for chunk in chunks:
        try:
            token = _wechat_access_token()
            if not token:
                return
            r = requests.post(
                "https://api.weixin.qq.com/cgi-bin/message/custom/send",
                params={"access_token": token},
                data=json.dumps({"touser": openid, "msgtype": "text",
                                 "text": {"content": chunk}}, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
            ).json()
            if r.get("errcode") in (40001, 42001):  # token 失效 → 强刷重试一次
                token = _wechat_access_token(force=True)
                if not token:
                    return
                requests.post(
                    "https://api.weixin.qq.com/cgi-bin/message/custom/send",
                    params={"access_token": token},
                    data=json.dumps({"touser": openid, "msgtype": "text",
                                     "text": {"content": chunk}}, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"},
                )
        except Exception:
            pass


def _wechat_process(text: str, openid: str) -> None:
    res = agent.analyze(text)
    _wechat_push(openid, _result_to_text(res))


def _wechat_xml(to_user: str, from_user: str, content: str) -> Response:
    """被动回复 XML（必须 5 秒内返回）。"""
    content = content.replace("]]>", "]] >")  # 防 CDATA 逃逸
    xml = (f"<xml><ToUserName><![CDATA[{to_user}]]></ToUserName>"
           f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
           f"<CreateTime>{int(time.time())}</CreateTime>"
           f"<MsgType><![CDATA[text]]></MsgType>"
           f"<Content><![CDATA[{content}]]></Content></xml>")
    return Response(content=xml, media_type="application/xml")


@app.get("/api/wechat/webhook")
def wechat_verify(signature: str = "", timestamp: str = "", nonce: str = "",
                  echostr: str = "") -> Response:
    """公众号后台「服务器配置」保存时的校验：签名对得上就原样回 echostr。"""
    if _wechat_sign_ok(signature, timestamp, nonce):
        return Response(content=echostr, media_type="text/plain")
    return Response(content="bad signature", status_code=403)


@app.post("/api/wechat/webhook")
async def wechat_webhook(req: Request, signature: str = "", timestamp: str = "",
                         nonce: str = "") -> Response:
    if not _wechat_sign_ok(signature, timestamp, nonce):
        return Response(content="bad signature", status_code=403)
    try:
        root = ET.fromstring(await req.body())
        get = lambda tag: (root.findtext(tag) or "").strip()  # noqa: E731
        openid, account = get("FromUserName"), get("ToUserName")
        msg_type = get("MsgType")

        # 新关注 → 欢迎语
        if msg_type == "event" and get("Event").lower() == "subscribe":
            return _wechat_xml(openid, account,
                               "👋 欢迎使用 Hermes 投研助手！\n"
                               "直接发股票名或代码即可，如「贵州茅台」；\n"
                               "也支持「对比 茅台 和 五粮液」「回测 贵州茅台」。")

        # 文本消息 → 先 ack 再后台分析，结果走客服消息推送
        if msg_type == "text":
            text = get("Content")
            if text:
                threading.Thread(target=_wechat_process, args=(text, openid), daemon=True).start()
                return _wechat_xml(openid, account, "📥 已收到，正在取数分析（约 1~2 分钟），完成后自动发你。")
    except Exception:
        pass
    return Response(content="success", media_type="text/plain")  # 微信约定：回 success 表示不再重试
