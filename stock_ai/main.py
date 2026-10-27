"""주식 매수/매도 AI 모델 실행 스크립트.

예시:
    python main.py backtest 005930.KS          # 삼성전자 워크포워드 백테스트
    python main.py predict 005930.KS AAPL      # 오늘 기준 매수/매도 신호
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from backtest import metrics, run_backtest, to_positions
from data import load_prices
from features import make_features, make_label
from model import train, walk_forward

OUT_DIR = Path(__file__).parent / "output"


def default_market(ticker: str) -> str:
    return "^KS11" if ticker.endswith((".KS", ".KQ")) else "^GSPC"


def build_dataset(ticker: str, args):
    df = load_prices(ticker, args.start, args.refresh)
    market = None if args.no_market else load_prices(default_market(ticker), args.start, args.refresh)
    X = make_features(df, market)
    y = make_label(df, args.horizon)
    keep = X.notna().all(axis=1)
    return df[keep], X[keep], y[keep]


def fmt(name, v):
    if pd.isna(v):
        return ""
    if name == "매매횟수":
        return f"{int(v):d}"
    return f"{v:.2f}" if name == "샤프지수" else f"{v:.1%}"


def cmd_backtest(ticker: str, args):
    df, X, y = build_dataset(ticker, args)
    proba = walk_forward(X, y, args.horizon, args.min_train, args.retrain_every)

    scored = proba.notna() & y.notna()
    print(f"\n=== {ticker} ({X.index[0]:%Y-%m-%d} ~ {X.index[-1]:%Y-%m-%d}) ===")
    print(f"예측 정확도: {accuracy_score(y[scored], proba[scored] > 0.5):.2%}  "
          f"AUC: {roc_auc_score(y[scored], proba[scored]):.3f}  "
          f"(실제 상승 비율: {y[scored].mean():.2%})")

    pos = to_positions(proba, args.buy_th, args.sell_th)
    bt = run_backtest(df, pos, args.cost)
    table = pd.DataFrame({
        "AI 전략": metrics(bt["strategy_ret"], bt["position"]),
        "단순 보유": metrics(bt["buyhold_ret"]),
    })
    print(table.apply(lambda col: [fmt(k, v) for k, v in col.items()]).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    name = ticker.replace("^", "_")
    bt.assign(proba=proba).to_csv(OUT_DIR / f"{name}_backtest.csv")
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        bt[["strategy_equity", "buyhold_equity"]].plot(ax=ax, logy=True)
        ax.legend(["AI strategy", "Buy & hold"])
        ax.set_title(f"{ticker} walk-forward backtest (cost {args.cost:.2%}/trade)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}_equity.png", dpi=110)
        plt.close(fig)
    print(f"결과 저장: {OUT_DIR}/{name}_*")


def cmd_predict(ticker: str, args):
    df, X, y = build_dataset(ticker, args)
    labeled = y.notna()
    model = train(X[labeled], y[labeled])
    p = model.predict_proba(X.iloc[[-1]])[0, 1]
    if p > args.buy_th:
        action = "매수(BUY)"
    elif p < args.sell_th:
        action = "매도(SELL)"
    else:
        action = "관망(HOLD, 기존 포지션 유지)"
    print(f"{ticker} {X.index[-1]:%Y-%m-%d} 종가 {df['Close'].iloc[-1]:,.2f} | "
          f"{args.horizon}일 후 상승 확률 {p:.1%} → {action}")


def main():
    ap = argparse.ArgumentParser(description="주식 매수/매도 AI 모델")
    ap.add_argument("command", choices=["backtest", "predict"])
    ap.add_argument("tickers", nargs="+", help="예: 005930.KS, 035720.KQ, AAPL")
    ap.add_argument("--start", default="2010-01-01", help="데이터 시작일")
    ap.add_argument("--horizon", type=int, default=5, help="며칠 뒤 상승 여부를 예측할지")
    ap.add_argument("--buy-th", type=float, default=0.55, help="이 확률보다 높으면 매수")
    ap.add_argument("--sell-th", type=float, default=0.45, help="이 확률보다 낮으면 매도")
    ap.add_argument("--cost", type=float, default=0.0015, help="매수/매도 1회당 비용 비율")
    ap.add_argument("--min-train", type=int, default=750, help="첫 학습에 쓸 최소 거래일 수")
    ap.add_argument("--retrain-every", type=int, default=63, help="재학습 주기(거래일)")
    ap.add_argument("--no-market", action="store_true", help="시장지수 피처 사용 안 함")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 데이터 다시 받기")
    args = ap.parse_args()

    for t in args.tickers:
        (cmd_backtest if args.command == "backtest" else cmd_predict)(t, args)


if __name__ == "__main__":
    main()
