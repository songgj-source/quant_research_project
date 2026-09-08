"""
sentiment.py
§5.6 관련성/집계 규칙, §5.6.1 스칼라화 수식, §5.7 로직 중 스칼라화까지 담당.
모든 함수는 순수 함수로 작성해서 단위테스트하기 쉽게 한다 (§9.1의 파이프라인 단위테스트 원칙).
"""
import numpy as np
import pandas as pd
import config as cfg


# ── §5.6.1 스칼라화(Scalarization) ──────────────────────────
def scalarize_finbert(p_positive: float, p_negative: float) -> float:
    """s_i^FinBERT = P(positive) - P(negative)"""
    return p_positive - p_negative


def scalarize_vader(compound: float) -> float:
    """VADER compound score를 그대로 사용 (-1~1)"""
    return compound


def scalarize_llm_polarity(polarity_score: float) -> float:
    """LLM 구조화 출력의 극성 척도 점수를 그대로 사용"""
    return polarity_score


# ── §5.6 관련성 가중 w_i^rel = 1 / |K_i| ────────────────────
def relevance_weight(n_tagged_tickers: int) -> float:
    if n_tagged_tickers <= 0:
        raise ValueError("n_tagged_tickers must be >= 1")
    return 1.0 / n_tagged_tickers


# ── §5.6 시간감쇠 가중 w_i^decay = exp(-lambda * delta_t) ───
def decay_weight(delta_hours: float, half_life_hours: float = cfg.DECAY_HALFLIFE_HOURS) -> float:
    if delta_hours < 0:
        # cut-off 이후에 발행된 기사는 애초에 그 시점의 feature에 포함되면 안 됨 (룩어헤드 방지)
        raise ValueError("delta_hours must be >= 0 (article must be published before cutoff)")
    lam = np.log(2) / half_life_hours
    return float(np.exp(-lam * delta_hours))


def aggregate_daily_sentiment(
    articles: pd.DataFrame,
    score_col: str,
    cutoff_ts: pd.Timestamp,
    half_life_hours: float = cfg.DECAY_HALFLIFE_HOURS,
) -> float:
    """
    S_{j,t} = [ sum_i w_rel_i * w_decay_i * s_i ] / [ sum_i w_rel_i * w_decay_i ]
    articles: 특정 종목 j, 특정 날짜 t에 대해 cutoff 이전에 발행된 기사만 미리 필터링된 상태여야 함.
    필요한 컬럼: score_col, 'n_tagged_tickers', 'published_at'
    """
    if len(articles) == 0:
        return np.nan

    # 룩어헤드 안전성: cutoff 이후 기사가 섞여 들어오면 즉시 에러 (단위테스트에서 항상 체크)
    if (articles["published_at"] > cutoff_ts).any():
        raise ValueError("Look-ahead violation: article published after cutoff included in aggregation")

    delta_hours = (cutoff_ts - articles["published_at"]).dt.total_seconds() / 3600.0
    w_rel = articles["n_tagged_tickers"].apply(relevance_weight)
    w_decay = delta_hours.apply(lambda dh: decay_weight(dh, half_life_hours))
    w = w_rel * w_decay

    if w.sum() == 0:
        return np.nan
    return float((w * articles[score_col]).sum() / w.sum())


# ── §5.5 사건 클러스터링 (확산성 판정) ──────────────────────
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def is_same_event(
    emb_a: np.ndarray, emb_b: np.ndarray, time_a: pd.Timestamp, time_b: pd.Timestamp,
    time_window_hours: float = cfg.CLUSTER_TIME_WINDOW_HOURS,
    cosine_threshold: float = cfg.CLUSTER_COSINE_THRESHOLD,
) -> bool:
    """cos(h_a, h_b) >= theta AND |time_a - time_b| <= time_window"""
    within_time = abs((time_a - time_b).total_seconds() / 3600.0) <= time_window_hours
    similar_enough = cosine_similarity(emb_a, emb_b) >= cosine_threshold
    return within_time and similar_enough
