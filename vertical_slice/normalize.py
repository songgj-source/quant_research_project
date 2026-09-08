"""
normalize.py
§5.7 감성 점수 스케일 정규화 — 훈련구간 전체 단일 통계량이 아니라
시점 t 기준 과거 252거래일 롤링 통계량으로 point-in-time z-score 정규화한다.
"""
import numpy as np
import pandas as pd
import config as cfg


def rolling_zscore(series: pd.Series, window: int = cfg.ROLLING_NORMALIZE_WINDOW,
                    min_periods: int = 60) -> pd.Series:
    """
    S_hat_{j,t} = (S_{j,t} - mu_{j,t}^(window)) / sigma_{j,t}^(window)
    mu, sigma는 t 시점 "이전" window 거래일만 사용 (shift(1) 필수 — 룩어헤드 방지).

    min_periods (코드 리뷰 피드백 C 반영): 기존 min_periods=5는 표본이 너무 작아 표준편차가
    극단적으로 작게 나올 수 있고, 이 경우 z-score가 무한대에 가깝게 폭증해 모델 학습을 붕괴시킬
    위험이 있다. 최소 60거래일(§5.7의 window=252 대비 약 1/4 분기)이 쌓이기 전까지는 NaN을
    그대로 반환하며, 호출하는 쪽(§5.7 "연구 초기 처리" 원칙)에서 그 구간을 0으로 채우지 않고
    분석에서 제외하도록 한다.
    """
    past = series.shift(1)  # t 시점 자기 자신은 정규화 파라미터 산정에 포함하지 않음
    mu = past.rolling(window=window, min_periods=min_periods).mean()
    sigma = past.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (series - mu) / sigma
    return z.replace([np.inf, -np.inf], np.nan)


def rolling_rank_normalize(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    횡단면 순위 정규화(cross-sectional rank normalization) — 강건성 비교용 대안.
    df_wide: index=date, columns=종목, values=감성 점수. 각 날짜별로 종목 간 순위를 [0,1]로 변환.
    """
    return df_wide.rank(axis=1, pct=True)


def normalize_by_stock(df: pd.DataFrame, value_col: str, stock_col: str = "ticker",
                        date_col: str = "date", window: int = cfg.ROLLING_NORMALIZE_WINDOW) -> pd.Series:
    """종목별로 그룹화하여 rolling_zscore를 적용 (다중 종목 패널 데이터용)."""
    df_sorted = df.sort_values([stock_col, date_col])
    out = df_sorted.groupby(stock_col, group_keys=False)[value_col].apply(
        lambda s: rolling_zscore(s, window=window)
    )
    return out.reindex(df.index)
