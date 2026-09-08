"""
build_model.py  —  단일 파일로 합친 M0~M5(+M0_news) 절제실험 vertical slice 파이프라인
(원본: config.py, sentiment.py, normalize.py, target.py, models.py, backtest.py,
 data_synthetic.py, run_ablation.py 를 하나로 병합)

연구계획서(v9) §9.1의 vertical slice 개념 구현체.
심층 코드 리뷰 피드백 반영판:
  ① M0_news 베이스라인 추가 (뉴스 존재 vs 감성 내용 분리)
  ② fold 경계를 넘지 않는 paired block bootstrap (diff 직접 부트스트랩)
  ③ Primary 검정을 M3 vs M5로 수정 (계획서 H2와 일치)
  ④ target 시간축 정합성 명시적 문서화 (shift(-1) 미적용 — 이유는 target.py 주석 참조)
  ⑤ 합성 데이터의 텍스트감성/확산성 신호를 통계적으로 독립시킴 (M5 우위 조작 방지)
  + warnings 억제 제거, assert -> raise ValueError, FinBERT 3-class 확률 합=1 보장
실행: python build_model.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

"""
config.py
연구계획서(v9) §5.4~§5.7, §6.4의 수치를 코드 상수로 고정한다.
여기 있는 값들은 전부 "사전 고정"된 것으로, 결과를 본 뒤 바꾸면 안 된다 (threshold selection bias 방지 원칙, §5.5).
"""

# ── §5.4 예측 대상 / 실행 모델 ──────────────────────────────
CUTOFF_TIME = "09:25"          # 정보 cut-off (ET)
EXECUTION_BUFFER_MIN = 5       # cut-off ~ 체결 사이 실행 여유
ENTRY_TIME = "09:30"           # 진입 시점 (정규장 시가)
CLASSIFY_THRESHOLD_SIGMA = 0.5  # y_{t+1} = sign(r) * 1{|r| > 0.5 * sigma_j}

# ── §5.6 관련성 / 시간감쇠 ──────────────────────────────────
DECAY_HALFLIFE_HOURS = 6.0     # w_decay = exp(-ln2/h * delta_t), h=6시간
DECAY_HALFLIFE_SENSITIVITY = [3.0, 12.0]  # 강건성 분석용 대안 half-life

# ── §5.5 사건 클러스터링 (확산성) ──────────────────────────
CLUSTER_TIME_WINDOW_HOURS = 4.0
CLUSTER_COSINE_THRESHOLD = 0.85

# ── §5.7 정규화 ─────────────────────────────────────────────
ROLLING_NORMALIZE_WINDOW = 252  # 거래일 기준 rolling z-score 윈도우

# ── §6.4 Primary 지표 / 백테스트 ───────────────────────────
TRADING_DAYS_PER_YEAR = 252
COST_BPS_GRID = [5, 10, 20, 30]     # 거래비용 민감도 (bp, 왕복)
WORST_CASE_SLIPPAGE_BPS = 30        # §6.5.2 worst-case 시나리오
MESI_MIN_SHARPE_IMPROVEMENT = 0.1   # ΔSR = SR_M5 - SR_M3 >= 0.1
BLOCK_BOOTSTRAP_LENGTH = 10         # 거래일, 기본값 (§6.4 검정 단위 확정)
BLOCK_BOOTSTRAP_SENSITIVITY = [5, 20]
N_BOOTSTRAP = 2000

# ── §6.1 M0~M5 절제실험 feature 구성 ────────────────────────
# 각 모델이 사용하는 감성 feature 열 이름 (없으면 사용 안 함)
MODEL_FEATURE_SETS = {
    "M0": [],                                                                  # Price-only
    "M0_news": ["has_news"],                                                   # Price + '뉴스 유무'만 (내용 없음) — 코드 리뷰 피드백 ① 반영
    "M1": ["vader_compound", "has_news"],                                      # 사전 기반 (VADER)
    "M2": ["finbert_polarity", "has_news"],                                    # 금융 특화 인코더 (FinBERT)
    "M3": ["llm_polarity", "has_news"],                                        # LLM 단일 극성
    "M4": ["llm_polarity", "llm_intensity", "llm_uncertainty", "has_news"],    # LLM 3D
    "M5": ["llm_polarity", "llm_intensity", "llm_uncertainty",
           "diffusion_score", "has_news"],                                    # 3D + Diffusion
}
# 코드 리뷰 피드백 ①: M0은 has_news조차 없어 M1~M5의 개선이 '감성 내용' 때문인지 '뉴스가 있었다는
# 사실 자체' 때문인지 분리되지 않았다. M0_news(Price+has_news, 내용 없음)를 진짜 대조군으로 추가했다.
# 감성 '내용'의 순수 기여를 보려면 M1~M5를 M0이 아니라 M0_news와 비교해야 한다.

PRICE_FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "rsi_14", "macd", "bb_pct",
    "mkt_ret_1d", "sector_ret_1d",
]

RANDOM_SEED = 42


"""
sentiment.py
§5.6 관련성/집계 규칙, §5.6.1 스칼라화 수식, §5.7 로직 중 스칼라화까지 담당.
모든 함수는 순수 함수로 작성해서 단위테스트하기 쉽게 한다 (§9.1의 파이프라인 단위테스트 원칙).
"""


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
def decay_weight(delta_hours: float, half_life_hours: float = DECAY_HALFLIFE_HOURS) -> float:
    if delta_hours < 0:
        # cut-off 이후에 발행된 기사는 애초에 그 시점의 feature에 포함되면 안 됨 (룩어헤드 방지)
        raise ValueError("delta_hours must be >= 0 (article must be published before cutoff)")
    lam = np.log(2) / half_life_hours
    return float(np.exp(-lam * delta_hours))


def aggregate_daily_sentiment(
    articles: pd.DataFrame,
    score_col: str,
    cutoff_ts: pd.Timestamp,
    half_life_hours: float = DECAY_HALFLIFE_HOURS,
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
    time_window_hours: float = CLUSTER_TIME_WINDOW_HOURS,
    cosine_threshold: float = CLUSTER_COSINE_THRESHOLD,
) -> bool:
    """cos(h_a, h_b) >= theta AND |time_a - time_b| <= time_window"""
    within_time = abs((time_a - time_b).total_seconds() / 3600.0) <= time_window_hours
    similar_enough = cosine_similarity(emb_a, emb_b) >= cosine_threshold
    return within_time and similar_enough


"""
normalize.py
§5.7 감성 점수 스케일 정규화 — 훈련구간 전체 단일 통계량이 아니라
시점 t 기준 과거 252거래일 롤링 통계량으로 point-in-time z-score 정규화한다.
"""


def rolling_zscore(series: pd.Series, window: int = ROLLING_NORMALIZE_WINDOW,
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
                        date_col: str = "date", window: int = ROLLING_NORMALIZE_WINDOW) -> pd.Series:
    """종목별로 그룹화하여 rolling_zscore를 적용 (다중 종목 패널 데이터용)."""
    df_sorted = df.sort_values([stock_col, date_col])
    out = df_sorted.groupby(stock_col, group_keys=False)[value_col].apply(
        lambda s: rolling_zscore(s, window=window)
    )
    return out.reindex(df.index)


"""
target.py
§5.4 예측 target 정의: r_{t+1} = Close_{t+1}/Open_{t+1} - 1
                        y_{t+1} = sign(r_{t+1}) * 1{|r_{t+1}| > 0.5 * sigma_j}
§6.1.1 M0 가격 입력 변수: OHLC/거래량, RSI(14), MACD, 볼린저밴드(20,2), 시장/섹터 요인

[행(row) 날짜 정합성 — 코드 리뷰에서 제기된 t/t+1 shift 논쟁에 대한 명시적 답]
§5.4의 cutoff·진입·청산은 전부 '같은 거래일 t+1' 안에서 일어난다:
  t (정보 앵커) = 전일 종가 + 오버나이트 뉴스가 관측되는 시점
  t+1 (행동일)  = cutoff(09:25)·진입(09:30 시가)·청산(종가)이 모두 일어나는 당일
즉 패널의 각 행 날짜 D는 표기법상 "t+1"에 해당한다. 따라서:
  - build_price_features()가 가격 파생 feature를 D-1(="t")로 1일 shift하는 것은 맞다 (룩어헤드 방지).
  - compute_target_return()은 그 SAME 행 D의 시가~종가(Open_{t+1}, Close_{t+1})를 그대로 써야 하며,
    여기에 추가로 shift(-1)을 걸면 "D-1 정보로 D+1 수익률을 예측"하는 꼴이 되어 하루를 건너뛰는
    새로운 정합성 오류가 생긴다. 코드 리뷰에서 제안된 shift(-1)은 적용하지 않는다.
"""


def compute_target_return(df: pd.DataFrame) -> pd.Series:
    """
    r_{t+1} = Close_{t+1} / Open_{t+1} - 1
    df 행의 날짜 D 자체가 't+1'(행동일)이므로 SAME 행의 open/close를 그대로 사용한다 (shift 없음).
    """
    return df["close"] / df["open"] - 1.0


def compute_classification_label(r_next: pd.Series, sigma_j: pd.Series,
                                  k: float = CLASSIFY_THRESHOLD_SIGMA) -> pd.Series:
    """
    y_{t+1} = sign(r_{t+1}) * 1{|r_{t+1}| > k * sigma_j}
    sigma_j는 "훈련 구간 내부에서만" 산정된 값이어야 한다 (호출하는 쪽에서 룩어헤드 없이 전달).
    반환값: {-1, 0, 1}
    """
    threshold = k * sigma_j
    label = np.sign(r_next)
    label = np.where(r_next.abs() > threshold, label, 0)
    return pd.Series(label.astype(int), index=r_next.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """%B = (Close - LowerBand) / (UpperBand - LowerBand)"""
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    band_width = (upper - lower).replace(0, np.nan)
    return (close - lower) / band_width


def build_price_features(panel: pd.DataFrame, mkt_returns: pd.Series) -> pd.DataFrame:
    """
    panel: index=date, columns=['ticker','open','high','low','close','volume','sector']을 가진 long-format DataFrame.
    mkt_returns: date -> 시장 전체(동일가중) 일간 수익률 (§6.1.1의 '시장 요인').
    반환: M0~M5 전 모델이 공통으로 사용하는 가격/시장/섹터 feature.

    중요: target(§5.4)이 '해당 날짜의 시가~종가 수익률'이므로, 그 날짜의 가격 feature는
    "전일 종가까지"의 정보만 사용해야 한다 (그렇지 않으면 그날 종가가 feature와 target에 동시에
    섞여 들어가는 룩어헤드가 발생한다). 아래에서 모든 파생 feature를 종목별로 1일 shift한다.
    """
    out = []
    panel = panel.sort_values(["ticker", "date"])
    for ticker, g in panel.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ret_1d"] = g["close"].pct_change(1)
        g["ret_5d"] = g["close"].pct_change(5)
        g["rsi_14"] = rsi(g["close"], 14)
        g["macd"] = macd(g["close"])
        g["bb_pct"] = bollinger_pct_b(g["close"])
        g["mkt_ret_1d"] = g["date"].map(mkt_returns)

        # 룩어헤드 방지: 파생 feature 전체를 하루 shift (D일 feature는 D-1일 종가까지의 정보만 사용)
        feature_cols = ["ret_1d", "ret_5d", "rsi_14", "macd", "bb_pct", "mkt_ret_1d"]
        g[feature_cols] = g[feature_cols].shift(1)

        out.append(g)
    result = pd.concat(out, ignore_index=True)

    # §6.1.1 섹터 요인: 섹터 내 동일가중 평균 수익률도 동일하게 1일 shift 필요
    # (ret_1d가 이미 shift된 상태이므로 이를 기반으로 섹터 평균을 내면 자동으로 D-1 시점 값이 됨)
    sector_ret = result.groupby(["date", "sector"])["ret_1d"].transform("mean")
    result["sector_ret_1d"] = sector_ret

    return result


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


def get_feature_columns(model_name: str) -> list:
    """M0~M5 각 모델이 사용하는 전체 feature 목록 (가격 feature는 모든 모델에 항상 동일하게 포함, §6.1의 '진짜 ablation의 전제조건')"""
    sentiment_cols = MODEL_FEATURE_SETS[model_name]
    return PRICE_FEATURE_COLUMNS + sentiment_cols


class AblationModel:
    """단일 모델(M0~M5 중 하나)을 감싸는 얇은 wrapper. LightGBM 멀티클래스 분류기."""

    def __init__(self, model_name: str):
        if model_name not in MODEL_FEATURE_SETS:
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
            random_state=RANDOM_SEED,
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


"""
backtest.py
§6.4 Primary 지표(비용 차감 후 연율화 샤프비율) 및 §6.5 거래 전략/비용 모델.

핵심 원칙 (문서와 일치시켜야 하는 부분):
- 포트폴리오: 매일 '상승' 예측 종목 전체에 동일비중(equal-weight) 투자, 없으면 전액 현금 (§6.5.1)
- 거래비용: bp 단위 왕복 비용 * turnover (§6.5.2)
- Primary 지표: 연율화 샤프비율 SR = sqrt(252) * mean(r_net) / std(r_net) (§6.4)
- 턴오버는 항상 같이 보고 (전문 피드백 §5 반영 — 필수 보조지표)
"""


def build_daily_portfolio_returns(signals: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    signals: index=date, columns=ticker, values = 1(상승 예측) / 0(아님)
    returns: index=date, columns=ticker, values = 해당 종목의 r_{t+1} (§5.4 정의, 시가~종가 수익률)
    반환: date별 gross portfolio return, turnover

    턴오버 계산 (코드 리뷰 피드백 A 반영):
    본 전략은 §5.4에 따라 '보유기간 1일, 오버나이트 갭 익스포저 없음' — 즉 매일 시가에 진입해
    당일 종가에 전량 청산하고 밤새 현금으로 보유한다. 따라서 매일 아침 시가 시점의 '실현 비중'은
    항상 0(전액 현금)이며, 전날 목표 비중(weights_prev)과의 차이로 턴오버를 계산하면 안 된다
    (전날과 오늘 동일한 종목을 골랐어도 실제로는 어제 팔고 오늘 다시 산 것이므로 턴오버가 발생함).
    올바른 턴오버(편도, one-way)는 그날 매수한 비중의 합, 즉 sum(weights_t) 그 자체다.
    (참고: 여러 날 보유하는 전략 — §6.5.1의 3일/5일 보유기간 Robustness 확장 —에서는 오버나이트
    가격 변동으로 실현 비중이 표류하므로, 이 경우엔 build_daily_portfolio_returns_multiday()를
    사용해야 한다.)
    """
    dates = signals.index
    records = []

    for date in dates:
        sig = signals.loc[date]
        selected = sig[sig == 1].index
        if len(selected) == 0:
            weights = pd.Series(0.0, index=signals.columns)  # 전액 현금
        else:
            w = 1.0 / len(selected)
            weights = pd.Series(0.0, index=signals.columns)
            weights.loc[selected] = w

        # 턴오버(편도) = 오늘 매수한 비중의 합. 매일 밤 전량 현금화되므로(§5.4) 전날 목표 비중과
        # 무관하게, 오늘 실제로 산 금액만큼이 곧 오늘의 거래 활동이다.
        turnover = float(weights.sum())

        ret_today = returns.loc[date].reindex(signals.columns).fillna(0.0)
        gross_ret = float((weights * ret_today).sum())

        records.append({"date": date, "gross_return": gross_ret, "turnover": turnover})

    return pd.DataFrame(records).set_index("date")


def build_daily_portfolio_returns_multiday(
    signals: pd.DataFrame, daily_returns: pd.DataFrame, holding_days: int = 1
) -> pd.DataFrame:
    """
    §6.5.1 Robustness 확장(3일/5일 보유기간)용 — 코드 리뷰 피드백 A에서 제안한 방식 그대로 구현.
    holding_days > 1이면 포지션이 밤새 유지되므로, 오늘 목표 비중과 비교해야 할 '어제 비중'은
    목표 비중 그 자체가 아니라 '가격 변동으로 표류한(drift) 실현 비중'이어야 한다:

        realized_weight_i = (weight_prev_i * (1 + daily_return_i)) / sum_j(weight_prev_j * (1 + daily_return_j))

    holding_days=1(§5.4 기본 전략)일 때는 매일 전량 청산·현금화하므로 이 함수 대신
    build_daily_portfolio_returns()를 사용하는 것이 맞다.
    """
    dates = signals.index
    weights_prev_target = pd.Series(0.0, index=signals.columns)
    realized_weights = pd.Series(0.0, index=signals.columns)  # 오늘 아침, 가격 표류를 반영한 실제 비중
    days_held = 0
    records = []

    for date in dates:
        rebalance_today = (days_held >= holding_days) or (weights_prev_target.sum() == 0)

        if rebalance_today:
            sig = signals.loc[date]
            selected = sig[sig == 1].index
            if len(selected) == 0:
                target_weights = pd.Series(0.0, index=signals.columns)
            else:
                w = 1.0 / len(selected)
                target_weights = pd.Series(0.0, index=signals.columns)
                target_weights.loc[selected] = w
            days_held = 0
        else:
            target_weights = weights_prev_target  # 보유 지속 (아직 재구성 시점 아님)

        # 턴오버 = 목표 비중과 '가격 표류가 반영된 실현 비중'의 차이 (코드 리뷰 피드백 A의 제안)
        turnover = 0.5 * (target_weights - realized_weights).abs().sum()

        ret_today = daily_returns.loc[date].reindex(signals.columns).fillna(0.0)
        gross_ret = float((target_weights * ret_today).sum())

        # 다음 날 아침을 위해 실현 비중을 오늘 가격 변동만큼 표류시킴
        drifted = target_weights * (1 + ret_today)
        total = drifted.sum()
        realized_weights = drifted / total if total > 0 else pd.Series(0.0, index=signals.columns)

        weights_prev_target = target_weights
        days_held += 1

        records.append({"date": date, "gross_return": gross_ret, "turnover": turnover})

    return pd.DataFrame(records).set_index("date")


def apply_cost(port_df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """r_net = r_gross - c * turnover  (c는 왕복 거래비용, bp -> 소수)"""
    c = cost_bps / 10000.0
    out = port_df.copy()
    out["net_return"] = out["gross_return"] - c * out["turnover"]
    return out


def annualized_sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """SR = sqrt(252) * mean(r) / std(r)"""
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return float(np.sqrt(periods_per_year) * mu / sigma)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    return float(drawdown.min())


def paired_block_bootstrap_diff(
    fold_diff_series: list,
    block_length: int = BLOCK_BOOTSTRAP_LENGTH,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    §6.4 검정 단위 확정 — 코드 리뷰 피드백 ② 반영 (기존 구현의 두 가지 결함을 모두 고침):

    결함 1) 기존에는 여러 fold의 포트폴리오 수익률을 pd.concat으로 이어붙인 뒤 그 위에서
    블록을 뽑았다. fold 사이에는 embargo로 인한 시간적 공백이 있어, 서로 다른 fold의 날짜를
    하나의 블록으로 묶으면 실제로는 연속되지 않은 기간을 연속된 것처럼 취급하는 오류가 생긴다.
    → 이 함수는 fold_diff_series(리스트, 각 원소가 한 fold의 시계열)를 받아 **블록이 fold
    경계를 절대 넘지 않도록** fold별로 독립적으로 블록을 추출한 뒤에만 이어붙인다.

    결함 2) 기존에는 SR(A)와 SR(B)를 각각 부트스트랩한 뒤 빼는 방식이라 진짜 paired 검정이
    아니었다. → 이 함수는 날짜별 차이 diff_t = r_B,t - r_A,t를 먼저 만들고 그 차이 자체를
    부트스트랩한다 (표준적인 paired bootstrap 방식).

    fold_diff_series: [diff_fold_1, diff_fold_2, ...] — 각 원소는 해당 fold의 diff_t = r_B - r_A pd.Series.
    통계량: 연율화된 diff의 평균/표준편차 (mean(diff)*sqrt(252)/std(diff)). 0이 신뢰구간 밖이면 유의함.
    """
    rng = np.random.default_rng(seed)
    fold_arrays = [s.values for s in fold_diff_series if len(s) > 0]
    if len(fold_arrays) == 0:
        raise ValueError("fold_diff_series에 유효한 fold가 없습니다")

    def diff_stat(arr: np.ndarray) -> float:
        sigma = arr.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return np.nan
        return float(np.sqrt(TRADING_DAYS_PER_YEAR) * arr.mean() / sigma)

    point_estimate = diff_stat(np.concatenate(fold_arrays))

    boot_stats = []
    for _ in range(n_boot):
        resampled_folds = []
        for arr in fold_arrays:
            n = len(arr)
            bl = min(block_length, n)
            n_blocks = int(np.ceil(n / bl))
            idx = []
            for _ in range(n_blocks):
                start = rng.integers(0, n - bl + 1) if n > bl else 0
                idx.extend(range(start, min(start + bl, n)))
            resampled_folds.append(arr[np.array(idx[:n])])  # fold 내부에서만 블록 추출
        combined = np.concatenate(resampled_folds)  # fold 순서대로만 이어붙임 (경계 넘는 블록 없음)
        boot_stats.append(diff_stat(combined))

    boot_stats = np.array(boot_stats)
    boot_stats = boot_stats[~np.isnan(boot_stats)]
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return {
        "point_estimate": point_estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(ci_low > 0 or ci_high < 0),
        "n_boot_used": len(boot_stats),
    }


def summarize_backtest(port_df: pd.DataFrame) -> dict:
    return {
        "annualized_sharpe_net": annualized_sharpe(port_df["net_return"]),
        "net_cumulative_return": float((1 + port_df["net_return"]).prod() - 1),
        "max_drawdown": max_drawdown(port_df["net_return"]),
        "avg_turnover": float(port_df["turnover"].mean()),
    }


"""
data_synthetic.py
실제 API(Alpaca/Polygon/Benzinga/GPT-4o) 키가 없는 환경에서 파이프라인 전체가
정상 작동하는지 검증하기 위한 합성 데이터 생성기.

주의: 여기서 만드는 '신호'는 순전히 파이프라인 테스트용이며, 실제 시장의
예측가능성을 의미하지 않는다. 실전 투입 시에는 이 파일만 실제 데이터 수집
모듈로 교체하면 되고, 나머지 모듈(sentiment.py, normalize.py, target.py,
models.py, backtest.py)은 인터페이스가 동일하므로 그대로 재사용 가능하다.

[코드 리뷰 피드백 ⑤ 반영] 이전 버전은 '텍스트 감성(극성/강도/불확실성)'과 '확산성'이
동일한 잠재 신호(true_sig)를 공유해서, M5가 M4를 이기는 게 '진짜 확산성의 추가 정보' 때문이
아니라 데이터 자체가 M5에 유리하게 설계된 결과일 수 있었다. 이제 두 개의 독립적인 잠재 신호
(idio_signal, diffusion_signal)를 따로 생성하고, 수익률에도 각각 독립적으로 더해서, M4->M5의
개선이 있다면 그건 진짜로 확산성이 텍스트 감성과 무관한 별도의 정보를 담고 있기 때문이다.
"""

rng = np.random.default_rng(RANDOM_SEED)

_IDIO_SIGNAL_CACHE = {}       # 종목별 극성/강도/불확실성의 원천이 되는 잠재 신호
_DIFFUSION_SIGNAL_CACHE = {}  # 종목별 확산성의 원천이 되는, idio_signal과 독립적인 잠재 신호


def generate_price_panel(tickers: list, n_days: int = 220, start_date: str = "2025-01-02") -> pd.DataFrame:
    """OHLCV 합성 데이터 + 임의의 섹터 배정 (§6.1.1 M0 가격 feature 산출용)"""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    sectors = ["Tech", "Health", "Finance", "Energy"]
    ticker_sector = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}

    # 시장 공통 요인 (모든 종목에 공통으로 영향을 주는 하루짜리 충격)
    mkt_shock = rng.normal(0.0003, 0.008, size=n_days)

    rows = []
    for t in tickers:
        price = 100.0 + rng.normal(0, 10)
        sector = ticker_sector[t]
        # 두 개의 독립적인 잠재 신호를 따로 생성 (코드 리뷰 피드백 ⑤)
        idio_signal = rng.normal(0, 1, size=n_days)        # 텍스트 감성(극성/강도/불확실성)의 원천
        diffusion_signal = rng.normal(0, 1, size=n_days)   # 확산성의 원천 — idio_signal과 무상관
        for i, d in enumerate(dates):
            # 참고: 계수들은 기본 데모가 뚜렷한 결과를 보이도록 다소 증폭한 값이다.
            # 실제 시장의 신호 대비 노이즈 비율은 이보다 훨씬 낮다 (EMH 가정, §2 참조) —
            # 이 값은 어디까지나 파이프라인 동작 확인용이며, 실제 데이터로 교체 시 의미가 없어진다.
            drift = (
                0.0002
                + 0.006 * np.tanh(idio_signal[i])        # 텍스트 감성이 설명하는 부분
                + 0.004 * np.tanh(diffusion_signal[i])   # 확산성만이 설명하는, 별도의 독립적인 부분
            )
            ret = drift + mkt_shock[i] * 0.5 + rng.normal(0, 0.012)
            open_p = price
            close_p = price * (1 + ret)
            high_p = max(open_p, close_p) * (1 + abs(rng.normal(0, 0.003)))
            low_p = min(open_p, close_p) * (1 - abs(rng.normal(0, 0.003)))
            vol = rng.integers(1_000_000, 8_000_000)
            rows.append({
                "date": d, "ticker": t, "sector": sector,
                "open": open_p, "high": high_p, "low": low_p, "close": close_p,
                "volume": vol,
            })
            price = close_p
        _IDIO_SIGNAL_CACHE[t] = pd.Series(idio_signal, index=dates)
        _DIFFUSION_SIGNAL_CACHE[t] = pd.Series(diffusion_signal, index=dates)

    return pd.DataFrame(rows)


def compute_market_returns(price_panel: pd.DataFrame) -> pd.Series:
    """§6.1.1 '시장 요인': 전 종목 동일가중 평균 수익률"""
    panel = price_panel.sort_values(["ticker", "date"]).copy()
    panel["ret_1d"] = panel.groupby("ticker")["close"].pct_change(1)
    return panel.groupby("date")["ret_1d"].mean()


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def generate_news_articles(tickers: list, n_days: int = 220, start_date: str = "2025-01-02",
                            avg_articles_per_stock_day: float = 1.2) -> pd.DataFrame:
    """
    §5.6 뉴스 기사 합성: 각 기사에 발행시각, 태깅 종목 수, VADER/FinBERT/LLM 원본 점수를 부여.
    LLM 극성/강도/불확실성은 idio_signal에서, 확산성은 별도의 diffusion_signal에서 생성한다
    (코드 리뷰 피드백 ⑤ — 두 신호가 통계적으로 독립이므로 M5의 개선이 있다면 진짜 별도 정보다).
    VADER/FinBERT는 더 noisy한 버전으로 생성해 '레거시 도구 대비 LLM이 낫다'는 선행연구(§4)의 취지를 재현한다.
    """
    dates = pd.bdate_range(start=start_date, periods=n_days)
    rows = []
    article_id = 0

    for t in tickers:
        idio = _IDIO_SIGNAL_CACHE.get(t)
        diffusion_sig = _DIFFUSION_SIGNAL_CACHE.get(t)
        if idio is None or diffusion_sig is None:
            raise RuntimeError("generate_price_panel()을 먼저 호출해야 합니다 (잠재 신호 캐시 필요)")

        for i, d in enumerate(dates):
            n_articles = rng.poisson(avg_articles_per_stock_day)
            true_sig = idio.iloc[i]           # 이 날짜의 '진짜' 텍스트 감성 신호
            diff_sig = diffusion_sig.iloc[i]  # 이 날짜의 '진짜' 확산성 신호 (idio와 무관)

            for _ in range(n_articles):
                # 발행시각: 전날 15:30(장마감 근사) ~ 당일 09:25(cutoff) 사이에서만 무작위 생성
                # (d-1 15:30부터 d 09:25까지는 약 17.9시간 — 절대 cutoff를 넘지 않도록 상한을 17.5시간으로 제한)
                hour_offset = rng.uniform(0, 17.5)
                published_at = pd.Timestamp(d) - pd.Timedelta(days=1) + pd.Timedelta(hours=15.5 + hour_offset)

                # LLM 점수: 진짜 신호에 약한 노이즈만 추가 (더 정확한 추출)
                llm_polarity = np.clip(np.tanh(true_sig) + rng.normal(0, 0.35), -1, 1)
                llm_intensity = np.clip(abs(np.tanh(true_sig)) + rng.normal(0, 0.2), 0, 1)
                llm_uncertainty = np.clip(0.5 - 0.3 * abs(np.tanh(true_sig)) + rng.normal(0, 0.2), 0, 1)

                # VADER: 같은 진짜 신호를 더 강한 노이즈로 관측 (레거시 도구의 정보 손실을 시뮬레이션)
                vader_compound = np.clip(np.tanh(true_sig) + rng.normal(0, 0.7), -1, 1)

                # FinBERT: 실제 모델처럼 3-class 확률이 반드시 합=1이 되도록 softmax로 생성
                # (코드 리뷰 '기타 버그' — 기존에는 p_pos/p_neg를 독립 노이즈로 만들어 합이 1이 아니었음)
                pos_logit = 0.8 * np.tanh(true_sig) + rng.normal(0, 0.6)
                neg_logit = -0.8 * np.tanh(true_sig) + rng.normal(0, 0.6)
                neu_logit = rng.normal(0, 0.4)
                p_pos, p_neg, p_neu = _softmax(np.array([pos_logit, neg_logit, neu_logit]))

                # 확산성 원천 데이터: idio_signal이 아니라 독립적인 diffusion_signal에서 생성
                n_tagged_tickers = 1 if rng.random() > 0.1 else 2  # 대부분 단일 종목 태깅
                diffusion_raw = max(0, rng.poisson(1 + 3 * abs(np.tanh(diff_sig))))

                rows.append({
                    "article_id": article_id, "ticker": t, "published_at": published_at,
                    "target_date": d,
                    "n_tagged_tickers": n_tagged_tickers,
                    "vader_compound": vader_compound,
                    "finbert_p_pos": float(p_pos), "finbert_p_neg": float(p_neg),
                    "finbert_p_neutral": float(p_neu),
                    "llm_polarity": llm_polarity, "llm_intensity": llm_intensity,
                    "llm_uncertainty": llm_uncertainty,
                    "diffusion_raw": diffusion_raw,
                })
                article_id += 1

    return pd.DataFrame(rows)


"""
run_ablation.py
§9.1 'vertical slice' 개념을 확장한 실행 스크립트.
데이터 생성 -> 가격 feature -> 뉴스 감성 집계·정규화 -> target 정의 ->
M0~M5 학습 -> H1(Macro-F1) / H2(Primary 샤프비율) 계층별 비교까지 전 과정을 한 번에 돌린다.

[코드 리뷰 심층 피드백 반영]
- warnings.filterwarnings("ignore") 제거 (경고를 숨기지 않고 그대로 노출)
- M0_news 베이스라인 추가로 '뉴스 존재 자체' 효과와 '감성 내용' 효과를 분리 (피드백 ①)
- Primary 검정을 M3 vs M5로 변경 (계획서 H2와 일치, 피드백 ③)
- fold 경계를 넘지 않는 paired block bootstrap 사용 (피드백 ②)

실행: python run_ablation.py
"""



def build_full_panel(tickers, n_days=220):
    print(f"[1/6] 합성 가격 데이터 생성 중... (종목 {len(tickers)}개 x {n_days}거래일)")
    price_panel = generate_price_panel(tickers, n_days=n_days)
    mkt_returns = compute_market_returns(price_panel)
    price_feat = build_price_features(price_panel, mkt_returns)

    print("[2/6] 합성 뉴스 데이터 생성 중 (텍스트 감성·확산성 독립 신호)...")
    articles = generate_news_articles(tickers, n_days=n_days)
    articles["finbert_polarity"] = scalarize_finbert(articles["finbert_p_pos"], articles["finbert_p_neg"])

    print("[3/6] §5.6 관련성·시간감쇠 가중 집계 수행 중 (룩어헤드 단위테스트 포함)...")
    score_cols = ["vader_compound", "finbert_polarity", "llm_polarity",
                  "llm_intensity", "llm_uncertainty", "diffusion_raw"]
    agg_rows = []
    for (ticker, day), grp in articles.groupby(["ticker", "target_date"]):
        cutoff_ts = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=25)
        row = {"ticker": ticker, "date": day, "has_news": 1}
        for col in score_cols:
            row[col] = aggregate_daily_sentiment(grp, col, cutoff_ts)
        agg_rows.append(row)
    sentiment_daily = pd.DataFrame(agg_rows).rename(columns={"diffusion_raw": "diffusion_score"})

    print("[4/6] 종목-일 패널에 감성 feature 병합 및 결측 처리 (0 대체 + missing indicator)...")
    panel = price_feat.merge(sentiment_daily, on=["ticker", "date"], how="left")
    sentiment_value_cols = ["vader_compound", "finbert_polarity", "llm_polarity",
                             "llm_intensity", "llm_uncertainty", "diffusion_score"]
    panel["has_news"] = panel["has_news"].fillna(0).astype(int)
    for col in sentiment_value_cols:
        panel[col] = panel[col].fillna(0.0)

    print("[5/6] §5.7 rolling 252일 point-in-time z-score 정규화 수행 중 (min_periods=60)...")
    for col in sentiment_value_cols:
        panel[col] = normalize_by_stock(panel, value_col=col, window=ROLLING_NORMALIZE_WINDOW)
    # 코드 리뷰 피드백 C: 정규화 초기 구간(min_periods 미달)의 NaN을 0으로 덮지 않고
    # §5.7의 '해당 구간을 분석에서 제외' 원칙에 따라 아래 dropna에서 제거한다.

    print("[6/6] §5.4 target(r_{t+1}) 및 분류 레이블(y_{t+1}) 계산 중...")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["r_next"] = compute_target_return(panel)
    # sigma_j: 종목별 과거 60거래일 롤링 표준편차 (미래 정보 미사용)
    panel["sigma_j"] = panel.groupby("ticker")["r_next"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).std()
    )
    panel = panel.dropna(
        subset=PRICE_FEATURE_COLUMNS + sentiment_value_cols + ["r_next", "sigma_j"]
    ).reset_index(drop=True)
    panel["y"] = compute_classification_label(panel["r_next"], panel["sigma_j"])

    return panel


def run_ablation(panel: pd.DataFrame, model_names=None, n_splits=3):
    if model_names is None:
        model_names = list(MODEL_FEATURE_SETS.keys())  # M0_news 포함 (피드백 ①)

    all_dates = panel["date"].unique()
    splits = walk_forward_splits(all_dates, n_splits=n_splits, min_train_days=100, test_days=25)
    print(f"\nwalk-forward 분할: {len(splits)}개 fold 생성됨 (embargo 적용됨)")

    results = {}
    fold_returns = {m: [] for m in model_names}  # 모델별 fold-wise net return 시계열 리스트

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)]
        if len(test) == 0 or len(train) == 0:
            continue

        for m in model_names:
            model = AblationModel(m).fit(train, train["y"])
            pred_label = model.predict_label(test)

            f1 = macro_f1(test["y"], pred_label)

            # §6.5.1 신호: 예측 레이블이 '상승(1)'인 종목을 매수 (분류기 출력 그대로 사용)
            sig_df = test.assign(signal=(pred_label == 1).astype(int))
            sig_wide = sig_df.pivot(index="date", columns="ticker", values="signal").fillna(0)
            ret_wide = sig_df.pivot(index="date", columns="ticker", values="r_next")

            port = build_daily_portfolio_returns(sig_wide, ret_wide)
            port_net = apply_cost(port, cost_bps=WORST_CASE_SLIPPAGE_BPS)
            summary = summarize_backtest(port_net)
            summary["macro_f1"] = f1
            summary["fold"] = fold_i

            results.setdefault(m, []).append(summary)
            fold_returns[m].append(port_net["net_return"])

    # fold별 결과 평균
    comparison = []
    for m in model_names:
        df = pd.DataFrame(results[m])
        comparison.append({
            "model": m,
            "macro_f1_H1": df["macro_f1"].mean(),
            "sharpe_net_H2": df["annualized_sharpe_net"].mean(),
            "net_cum_return": df["net_cumulative_return"].mean(),
            "max_drawdown": df["max_drawdown"].mean(),
            "avg_turnover": df["avg_turnover"].mean(),
        })
    comparison_df = pd.DataFrame(comparison).set_index("model")

    def fold_diffs(model_b, model_a):
        """모델 B와 A의 fold별 paired 일별 수익률 차이(diff_t = r_B,t - r_A,t) 리스트를 만든다."""
        diffs = []
        for rb, ra in zip(fold_returns[model_b], fold_returns[model_a]):
            if not rb.index.equals(ra.index):
                raise ValueError(
                    f"paired evaluation 실패: {model_b}와 {model_a}의 fold 날짜 index가 일치하지 않습니다."
                )
            diffs.append(rb - ra)
        return diffs

    # Primary 검정 (코드 리뷰 피드백 ③): 계획서 H2와 일치시켜 M3 vs M5로 수정 (M0 vs M5가 아님)
    primary_test = paired_block_bootstrap_diff(fold_diffs("M5", "M3"))
    # 참고용 보조 검정: M0(가격만) 대비 M5의 차이도 함께 보고 (얼마나 개선됐는지의 참고치)
    reference_m0_vs_m5 = paired_block_bootstrap_diff(fold_diffs("M5", "M0"))
    # 피드백 ① 검증용: '뉴스 내용'의 순수 기여 = M3(LLM 극성) vs M0_news(뉴스 유무만) 비교
    content_vs_presence_test = paired_block_bootstrap_diff(fold_diffs("M3", "M0_news"))

    return comparison_df, {
        "primary_M3_vs_M5": primary_test,
        "reference_M0_vs_M5": reference_m0_vs_m5,
        "content_vs_presence_M0news_vs_M3": content_vs_presence_test,
    }


if __name__ == "__main__":
    tickers = [f"TICK{i}" for i in range(1, 9)]  # 데모용 8개 종목 (실전에서는 core sample 100종목으로 교체)

    panel = build_full_panel(tickers, n_days=300)
    print(f"\n패널 크기: {panel.shape[0]}행, 종목 {panel['ticker'].nunique()}개, "
          f"기간 {panel['date'].min().date()} ~ {panel['date'].max().date()}")

    comparison_df, tests = run_ablation(panel, n_splits=3)

    print("\n" + "=" * 84)
    print("M0~M5(+M0_news) 절제실험 결과 (H1: Macro-F1 / H2: 비용차감 후 연율화 샤프비율)")
    print("=" * 84)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print(comparison_df)

    print("\n" + "=" * 84)
    print("통계 검정 (모두 fold 경계를 넘지 않는 paired block bootstrap, L=%d)" % BLOCK_BOOTSTRAP_LENGTH)
    print("=" * 84)
    print("\n[Primary] M3(단일 극성) vs M5(3D+확산성) — 계획서 H2와 일치:")
    for k, v in tests["primary_M3_vs_M5"].items():
        print(f"  {k}: {v}")

    print("\n[참고] M0(가격 단독) vs M5 — 전체적인 개선폭 참고치:")
    for k, v in tests["reference_M0_vs_M5"].items():
        print(f"  {k}: {v}")

    print("\n[뉴스 내용 vs 존재 분리] M0_news(뉴스 유무만) vs M3(LLM 극성) — 감성 '내용'의 순수 기여:")
    for k, v in tests["content_vs_presence_M0news_vs_M3"].items():
        print(f"  {k}: {v}")

    delta_sr = comparison_df.loc["M5", "sharpe_net_H2"] - comparison_df.loc["M3", "sharpe_net_H2"]
    print(f"\nMESI 체크 (ΔSR = SR_M5 - SR_M3 >= {MESI_MIN_SHARPE_IMPROVEMENT}): "
          f"ΔSR = {delta_sr:.4f} -> {'통과' if delta_sr >= MESI_MIN_SHARPE_IMPROVEMENT else '미달'}")
