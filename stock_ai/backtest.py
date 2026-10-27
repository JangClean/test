"""매수/매도 신호 생성과 백테스트."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_positions(proba: pd.Series, buy_th: float, sell_th: float) -> pd.Series:
    """확률이 buy_th 초과면 매수(1), sell_th 미만이면 매도(0), 그 사이는 기존 포지션 유지."""
    pos = np.full(len(proba), np.nan)
    cur = 0.0
    for i, p in enumerate(proba.to_numpy()):
        if np.isnan(p):
            continue
        if p > buy_th:
            cur = 1.0
        elif p < sell_th:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=proba.index)


def run_backtest(df: pd.DataFrame, positions: pd.Series, cost: float) -> pd.DataFrame:
    """t일 종가에 낸 신호를 t+1일 시가에 체결한다고 가정 (시가→시가 수익률).

    cost는 한 번 매수 또는 매도할 때의 비용(수수료+세금+슬리피지) 비율.
    """
    open_ret = df["Open"].shift(-1) / df["Open"] - 1        # t일 시가 → t+1일 시가
    held = positions.shift(1)                               # 전일 종가 신호로 오늘 시가 체결
    valid = held.notna() & open_ret.notna()
    held, open_ret = held[valid], open_ret[valid]

    trades = held.diff().abs().fillna(held.iloc[0])
    strat = held * open_ret - trades * cost
    return pd.DataFrame({
        "position": held,
        "strategy_ret": strat,
        "buyhold_ret": open_ret,
        "strategy_equity": (1 + strat).cumprod(),
        "buyhold_equity": (1 + open_ret).cumprod(),
        "trade": trades,
    })


def metrics(returns: pd.Series, positions: pd.Series | None = None) -> dict:
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    out = {
        "총수익률": equity.iloc[-1] - 1,
        "연환산수익률(CAGR)": cagr,
        "샤프지수": sharpe,
        "최대낙폭(MDD)": mdd,
    }
    if positions is not None:
        out["보유비중"] = positions.mean()
        out["매매횟수"] = int(positions.diff().abs().sum())
    return out
