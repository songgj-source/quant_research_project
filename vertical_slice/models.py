"""
models.py
§6.1 M0~M5 절제실험.

주의: 계획서(§6.3)의 Primary 백본은 PatchTST 단일 아키텍처로 확정되어 있다.
이 vertical slice에서는 파이프라인 전체(데이터->feature->학습->백테스트)가
정상 작동하는지를 빠르게 검증하는 것이 목적이므로, 학습이 빠르고 디버깅하기 쉬운
LightGBM(§6.1.2의 Robustness 트리 베이스라인과 동일 계열)을 임시 백본으로 사용한다.
실제 연구에서는 이 자리에 PatchTST 학습 루프를 그대로 끼워 넣으면 된다
(fit/predict_proba 인터페이스만 맞추면 아래 run_ablation.py는 수정할 필요 없음).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as cfg


def get_feature_columns(model_name: str) -> list:
    """M0~M5 각 모델이 사용하는 전체 feature 목록 (가격 feature는 모든 모델에 항상 동일하게 포함, §6.1의 '진짜 ablation의 전제조건')"""
    sentiment_cols = cfg.MODEL_FEATURE_SETS[model_name]
    return cfg.PRICE_FEATURE_COLUMNS + sentiment_cols


class AblationModel:
    """단일 모델(M0~M5 중 하나)을 감싸는 얇은 wrapper. LightGBM 멀티클래스 분류기."""

    def __init__(self, model_name: str):
        if model_name not in cfg.MODEL_FEATURE_SETS:
            raise ValueError(f"unknown model_name: {model_name}")
        self.model_name = model_name
        self.feature_cols = get_feature_columns(model_name)
        self.clf = None
        self._label_map = {-1: 0, 0: 1, 1: 2}   # LightGBM은 0-indexed 클래스가 필요
        self._inv_label_map = {v: k for k, v in self._label_map.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_ = X[self.feature_cols]
        y_ = y.map(self._label_map)
        self.clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_class=3,
            objective="multiclass",
            random_state=cfg.RANDOM_SEED,
            verbose=-1,
        )
        self.clf.fit(X_, y_)
        return self

    def predict_label(self, X: pd.DataFrame) -> pd.Series:
        X_ = X[self.feature_cols]
        pred = self.clf.predict(X_)
        return pd.Series(pred, index=X.index).map(self._inv_label_map)

    def predict_up_probability(self, X: pd.DataFrame) -> pd.Series:
        """롱온리 신호 생성을 위해 '상승(y=1)' 클래스 확률만 반환 (§6.5.1 신호 생성)"""
        X_ = X[self.feature_cols]
        proba = self.clf.predict_proba(X_)
        up_class_idx = self._label_map[1]
        return pd.Series(proba[:, up_class_idx], index=X.index)


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")


def walk_forward_splits(dates: np.ndarray, n_splits: int = 3,
                         min_train_days: int = 120, test_days: int = 30,
                         embargo_days: int = 2):
    """
    아주 단순화된 walk-forward 분할 (vertical slice용).

    Purging/Embargo (코드 리뷰 피드백 B 반영): train 구간의 마지막 날짜와 test 구간의 첫
    날짜 사이에 embargo_days만큼 강제로 공백을 둔다. target이 r_{t+1}(§5.4, 다음 거래일
    시가~종가 수익률)이므로, train의 마지막 날 라벨은 이미 train_end 다음날 가격을 내포하고
    있다. 그 다음날이 바로 test 첫날이 되면 특징 계산에 쓰인 rolling 통계량(§5.7 등)이나
    라벨 정보가 test 구간과 시간적으로 맞닿아 미세하게 겹칠 위험이 있으므로, 최소 1~2거래일의
    embargo를 두어 이 경계 누수를 차단한다.

    실제 연구에서는 §6.3의 정식 walk-forward 훈련/검증/테스트 분할 규칙(및 그 안의 embargo
    정책)으로 이 함수를 교체해야 한다.
    """
    unique_dates = np.sort(np.unique(dates))
    splits = []
    start = min_train_days
    for i in range(n_splits):
        train_end = start + i * test_days
        test_start = train_end + embargo_days   # ← embargo: train과 test 사이 강제 공백
        test_end = test_start + test_days
        if test_end > len(unique_dates):
            break
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[test_start:test_end]
        splits.append((train_dates, test_dates))
    return splits
