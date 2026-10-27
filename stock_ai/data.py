"""주가 데이터 다운로드 및 캐시."""
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "data_cache"


def load_prices(ticker: str, start: str = "2010-01-01", refresh: bool = False) -> pd.DataFrame:
    """일봉 OHLCV를 반환한다. 한국 주식은 '005930.KS'(코스피), '035720.KQ'(코스닥) 형식."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{ticker.replace('^', '_')}_{start}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 받지 못했습니다. 티커를 확인하세요.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Volume"] > 0]  # 거래정지일 제거
    df.to_csv(path)
    return df
