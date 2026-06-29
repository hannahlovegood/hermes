# Hermes 投研助手

一句话提问 A 股 → 自动取真实数据 → AI 分析 → 生成带图表的迷你研报。

这是一条「Agentic 投研」的竖切片：**意图识别 → 取数 → 分析 → 出研报**。

**已实现：**
- 📈 单标的研报：基本面 / 估值 / 财务 / 价格走势图 / **舆情速览（新闻公告）**
- 📊 **多标的对比**：「对比 茅台 和 五粮液」→ 指标对比表 + 相对走势图 + 对比分析
- ⚡ **流式输出**：Agent 取数轨迹通过 SSE 实时滚动展示
- ⬇ **研报导出 PDF**：一键打印/导出整份研报
- 🤖 **飞书机器人**：私聊机器人即可拿研报（配置 `FEISHU_*` 后启用）

---

## 60 秒跑起来

```bash
cd hermes
./run.sh
```

首次会自动建虚拟环境、装依赖（akshare 较大，约几分钟），然后打开浏览器：

```
http://127.0.0.1:8000
```

输入「贵州茅台」或「600519」回车即可。**没有 API Key 也能用**——自动进入演示模式，
行情/财务/估值都是 akshare 的真实数据，只是分析正文走的是确定性模板而非大模型。

> 已经装过依赖、想手动启动：
> `.venv/bin/python -m uvicorn app:app --reload`

---

## 变成真·AI 分析

```bash
cp .env.example .env
# 编辑 .env，把 ANTHROPIC_API_KEY 填上（https://console.anthropic.com 申请）
./run.sh
```

填上 key 后，右上角徽章会变成「AI 模式」，分析正文由 Claude 在真正的
tool-calling 循环里生成：它自己决定先认标的、再取哪些数据，最后写研报。

模型/成本可在 `.env` 调：`HERMES_MODEL`（默认 `claude-opus-4-8`，可换更便宜的
`claude-sonnet-4-6` / `claude-haiku-4-5`）、`HERMES_EFFORT`（思考深度 low/medium/high）。

---

## 结构

| 文件 | 角色 | 对应卡片里的 |
|---|---|---|
| `data_source.py` | akshare 数据封装（行情/财务/估值/名称解析），全程容错 | 信息采集 Agent / Gateway |
| `agent.py` | Claude 工具调用大脑 + 无 key 时的演示流水线 | 数据分析 Agent / 意图识别→研报 |
| `app.py` | FastAPI：网页 + `/api/analyze` | 多端接入（Web） |
| `static/index.html` | 单页前端：Agent 轨迹 + 指标卡 + 图表 + 研报 | 前端 |

数据源：[akshare](https://akshare.akfamily.xyz/)（公开免费，A 股）。

---

## 已知边界（v0）

- 覆盖沪深 A 股；北交所（8/4 开头）名称可能显示为代码，但行情/财务仍可取。
- akshare 走公开接口，偶有抖动；已加重试，个别字段取不到时研报会标注「暂不可用」而非崩溃。
- 上交所标的暂无行业字段（数据源所限），深市有。
- Agent 轨迹目前在结果返回后一次性展示；流式逐步展示是下一步。

## 接入飞书（可选）

1. 飞书开放平台建「企业自建应用」，复制 **App ID / App Secret** 填进 `.env` 的 `FEISHU_APP_ID` / `FEISHU_APP_SECRET`。
2. 开通权限 `im:message`，订阅事件 `im.message.receive_v1`。
3. 事件回调地址填 `http(s)://你的域名/api/feishu/webhook`（本地用 ngrok 暴露 8000 端口；加密留空）。
4. 重启后在飞书私聊机器人「贵州茅台」或「对比 茅台 和 五粮液」，它会把研报摘要发回来。

> 注：MVP 的 webhook 未校验飞书签名，仅供研究/内部使用；上生产请加签名校验。

## 再下一步可加（按需要）

- AI 模式下也流式吐字（目前 AI 模式整段返回，演示模式已逐步流式）
- 行业对比 / 同行自动选取、龙虎榜 / 资金流
- 研报存档与历史对比、定时盯盘推送
- 不构成投资建议；仅供研究学习。

---

*仅供研究与学习，不构成任何投资建议。*
