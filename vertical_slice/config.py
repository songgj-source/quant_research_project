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
