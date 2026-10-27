"""기술적 지표 피처와 학습 라벨 생성."""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def make_features(df: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """모든 피처는 당일 종가까지의 정보만 사용한다 (미래 정보 누출 없음)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)

    for n in (1, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 20, 60, 120):
        f[f"ma_gap_{n}"] = c / c.rolling(n).mean() - 1

    f["rsi_14"] = _rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)

    daily = c.pct_change()
    f["vol_20"] = daily.rolling(20).std()
    f["vol_ratio"] = f["vol_20"] / daily.rolling(120).std()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["range_1"] = (h - l) / c
    f["volume_ratio"] = v / v.rolling(20).mean()
    f["dist_52w_high"] = c / c.rolling(252).max() - 1
    f["dist_52w_low"] = c / c.rolling(252).min() - 1

    if market is not None:
        mc = market["Close"].reindex(df.index).ffill()
        for n in (1, 5, 20):
            f[f"mkt_ret_{n}"] = mc.pct_change(n)
        f["mkt_ma_gap_60"] = mc / mc.rolling(60).mean() - 1
        f["rel_strength_20"] = f["ret_20"] - f["mkt_ret_20"]

    return f.replace([np.inf, -np.inf], np.nan)


def make_label(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """horizon 거래일 뒤 종가가 오늘보다 높으면 1. 마지막 horizon일은 NaN(아직 모름)."""
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    return (fwd > 0).astype(float).where(fwd.notna())
