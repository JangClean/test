"""LightGBM 분류기 + 워크포워드(walk-forward) 학습."""
import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def train(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X, y)
    return model


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    min_train: int = 750,
    retrain_every: int = 63,
) -> pd.Series:
    """과거 데이터로만 학습해 이후 구간을 예측하는 과정을 반복한다.

    학습 구간 끝에서 horizon일을 잘라내(purge) 라벨이 예측 구간의 가격을
    엿보지 않도록 한다.
    """
    proba = pd.Series(np.nan, index=X.index)
    for start in range(min_train, len(X), retrain_every):
        train_end = start - horizon
        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        mask = ytr.notna()
        model = train(Xtr[mask], ytr[mask])
        end = min(start + retrain_every, len(X))
        proba.iloc[start:end] = model.predict_proba(X.iloc[start:end])[:, 1]
    return proba
