"""
backtest.py — Hermes 的「策略回测」层
====================================
双均线策略的向量化回测：快线（默认 20 日）上穿慢线（默认 60 日）次日持有，
下穿次日空仓。纯 pandas 实现，不引入 vectorbt/numba（venv 是 Python 3.9，
numba 是安装深坑，而这个量级的回测向量化 pandas 完全够用）。

分层原则（量化智能体的核心）：
- **裁决权在这里**：指标怎么算、成本怎么扣，全是确定性代码，LLM 无权修改结论。
- LLM 只做两件事：把用户意图变成参数（agent.py），以及解读这里输出的数字。

诚实性约定：
- 信号统一 shift(1)：今天收盘算出的信号，明天才能建仓，杜绝未来函数。
- 双边各扣 10bp 交易成本（佣金+印花税+滑点的粗估）。
- 永远同时报告同期「买入持有」基准——跑不赢基准的策略一眼可见。

自测：  .venv/bin/python backtest.py 600519 [fast] [slow]
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

import data_source as ds

DEFAULT_FAST = 20
DEFAULT_SLOW = 60
DEFAULT_DAYS = 750          # ≈ 三年交易日
COST_PER_SIDE = 0.001       # 双边各 10bp
TRADING_DAYS = 252


def _close_frame(code: str, days: int) -> Optional[pd.DataFrame]:
    """取近 N 个交易日的复权收盘价，转成 DataFrame(date, close)。"""
    p = ds.get_price_history(code, days=days)
    if p.get("_error") or not p.get("series"):
        return None
    df = pd.DataFrame(p["series"])
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df if len(df) > 0 else None


def _annualize(total_return: float, n_days: int) -> Optional[float]:
    if n_days <= 0 or total_return <= -1:
        return None
    return (1 + total_return) ** (TRADING_DAYS / n_days) - 1


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _sharpe(daily_ret: pd.Series) -> Optional[float]:
    std = daily_ret.std()
    if std is None or pd.isna(std) or std == 0:
        return None
    return float(daily_ret.mean() / std * TRADING_DAYS ** 0.5)


def _round2(v) -> Optional[float]:
    try:
        return None if v is None or pd.isna(v) else round(float(v), 2)
    except Exception:
        return None


def run_backtest(code: str, fast: int = DEFAULT_FAST, slow: int = DEFAULT_SLOW,
                 days: int = DEFAULT_DAYS) -> dict:
    """双均线回测。返回稳定形状的 dict；失败返回 {'_error': ...}，不抛异常。"""
    try:
        fast, slow, days = int(fast), int(slow), int(days)
    except Exception:
        return {"_error": "参数必须是整数（fast/slow/days）"}
    if not (0 < fast < slow):
        return {"_error": f"参数不合法：需要 0 < 快线({fast}) < 慢线({slow})"}

    df = _close_frame(code, days)
    if df is None:
        return {"_error": "价格数据获取失败，无法回测"}
    if len(df) < slow + 30:
        return {"_error": f"历史数据不足：仅 {len(df)} 个交易日，慢线 {slow} 日至少需要 {slow + 30} 日"}

    close = df["close"].astype(float)
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    # 今天收盘算信号，明天才能持有 → shift(1)，杜绝未来函数
    pos = (ma_fast > ma_slow).astype(float).shift(1).fillna(0.0)

    # 从慢线首次可用之后起算，策略与基准同一起跑线
    start = slow
    close_w = close.iloc[start:].reset_index(drop=True)
    pos_w = pos.iloc[start:].reset_index(drop=True)
    dates_w = df["date"].iloc[start:].reset_index(drop=True)

    ret = close_w.pct_change().fillna(0.0)
    turnover = pos_w.diff().abs().fillna(pos_w.iloc[0])  # 建/平仓当天计一次换手
    strat_ret = pos_w * ret - turnover * COST_PER_SIDE

    equity = (1 + strat_ret).cumprod()
    bench = close_w / close_w.iloc[0]

    n = len(close_w)
    strat_total = float(equity.iloc[-1] - 1)
    bench_total = float(bench.iloc[-1] - 1)

    # 逐笔交易：0→1 建仓、1→0 平仓，统计完整回合的胜率
    diff = pos_w.diff().fillna(pos_w.iloc[0])
    entries = list(diff[diff > 0].index)
    exits = list(diff[diff < 0].index)
    trades = []
    for i, ent in enumerate(entries):
        ex = next((x for x in exits if x > ent), None)
        if ex is None:
            break
        gross = close_w.iloc[ex] / close_w.iloc[ent] - 1
        trades.append(gross - 2 * COST_PER_SIDE)
    wins = sum(1 for t in trades if t > 0)
    holding = bool(pos_w.iloc[-1] > 0)

    return {
        "code": str(code).zfill(6),
        "strategy": f"双均线 MA{fast}/MA{slow}",
        "fast": fast, "slow": slow,
        "period": {"start": str(dates_w.iloc[0]), "end": str(dates_w.iloc[-1]), "trading_days": n},
        "cost_note": f"双边各 {COST_PER_SIDE:.1%} 交易成本，信号次日生效（无未来函数）",
        "metrics": {
            "total_return_pct": _round2(strat_total * 100),
            "annual_return_pct": _round2((_annualize(strat_total, n) or 0) * 100),
            "max_drawdown_pct": _round2(_max_drawdown(equity) * 100),
            "sharpe": _round2(_sharpe(strat_ret)),
        },
        "buy_hold": {
            "total_return_pct": _round2(bench_total * 100),
            "annual_return_pct": _round2((_annualize(bench_total, n) or 0) * 100),
            "max_drawdown_pct": _round2(_max_drawdown(bench) * 100),
            "sharpe": _round2(_sharpe(ret)),
        },
        "trades": {
            "completed": len(trades),
            "win_rate_pct": _round2(wins / len(trades) * 100) if trades else None,
            "open_position": holding,
        },
        "latest_signal": "持有（快线在慢线上方）" if holding else "空仓（快线在慢线下方）",
        "excess_return_pct": _round2((strat_total - bench_total) * 100),
    }


def format_report(bt: dict, name: str = "") -> str:
    """把回测结果排成 Markdown（演示模式用；AI 模式由模型解读同一份 JSON）。"""
    if bt.get("_error"):
        return f"## 策略回测\n\n- ⚠️ {bt['_error']}"
    m, b, t, p = bt["metrics"], bt["buy_hold"], bt["trades"], bt["period"]

    def f(v, suf=""):
        return "—" if v is None else f"{v}{suf}"

    beat = bt.get("excess_return_pct")
    verdict = ("策略跑赢买入持有" if (beat or 0) > 0 else "策略未跑赢买入持有——这很常见，简单择时在多数标的上不如拿住不动")
    L = [
        f"## 策略回测：{bt['strategy']}",
        f"标的：{name}（{bt['code']}）｜区间：{p['start']} ~ {p['end']}（{p['trading_days']} 个交易日）",
        "",
        "| 指标 | 策略 | 买入持有 |",
        "|---|---|---|",
        f"| 区间总收益 | {f(m['total_return_pct'], '%')} | {f(b['total_return_pct'], '%')} |",
        f"| 年化收益 | {f(m['annual_return_pct'], '%')} | {f(b['annual_return_pct'], '%')} |",
        f"| 最大回撤 | {f(m['max_drawdown_pct'], '%')} | {f(b['max_drawdown_pct'], '%')} |",
        f"| 夏普比率 | {f(m['sharpe'])} | {f(b['sharpe'])} |",
        "",
        f"- 完整交易 {t['completed']} 笔" + (f"，胜率 {t['win_rate_pct']}%" if t["win_rate_pct"] is not None else "")
        + ("，当前尚有持仓未平" if t["open_position"] else ""),
        f"- 最新信号：{bt['latest_signal']}",
        f"- 超额收益（对买入持有）：{f(beat, '%')} —— {verdict}",
        f"- {bt['cost_note']}",
        "",
        "**回测基于历史数据，参数为事后选择，存在过拟合风险；不代表未来表现，不构成投资建议。**",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    import json
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    fast = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FAST
    slow = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SLOW
    out = run_backtest(code, fast, slow)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print()
    print(format_report(out, code))
