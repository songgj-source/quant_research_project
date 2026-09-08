"""
data_synthetic.py
가격/뉴스 원천 데이터 수집 모듈. Alpaca Market Data API(OHLCV) + Alpaca News
API(Benzinga)에서 실제 데이터를 가져온다. API 키는 코드에 직접 쓰지 않고
api_setup/.env에서 python-dotenv로 읽는다.

나머지 모듈(sentiment.py, normalize.py, target.py, models.py, backtest.py)은
이 파일이 반환하는 DataFrame의 컬럼/타입이 예전 합성 데이터 버전과 동일하므로
전혀 수정할 필요가 없다.

[감성 점수 관련 한계] vader_compound, finbert_*, llm_* 컬럼은 아직 실제 계산
로직이 없어 전부 NaN이다. 원문 텍스트만 수집해두고, 다음 단계에서 GPT-4o 기반
구조화 출력(극성/강도/불확실성)을 별도로 붙여 이 컬럼들을 채울 예정이다.
diffusion_raw도 같은 이유로 NaN — §5.5.2의 4개 하위요소(독립 출처 수·재보도 수·
경과시간·매체유형)를 산출하려면 사건 클러스터링(sentiment.py의 is_same_event())이
먼저 실제 임베딩 기반으로 연결되어야 하는데, README에 명시된 대로 아직 안 되어 있다.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import config as cfg

_ENV_PATH = Path(__file__).resolve().parent.parent / "api_setup" / ".env"
load_dotenv(_ENV_PATH)

_ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
_ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not _ALPACA_API_KEY or not _ALPACA_SECRET_KEY:
    raise RuntimeError(
        f"{_ENV_PATH}에서 ALPACA_API_KEY / ALPACA_SECRET_KEY를 읽지 못했습니다. "
        "api_setup/.env.template을 .env로 복사한 뒤 실제 키를 채워주세요."
    )

# 실제 GICS 섹터 매핑 테이블이 아직 없어(README에 명시된 기존 한계), 우선 대형주
# 위주로만 하드코딩한 임시 매핑. 목록에 없는 종목은 "Unknown"으로 표시된다.
TICKER_SECTOR = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech", "GOOGL": "Tech", "NVDA": "Tech", "META": "Tech",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "WFC": "Finance",
    "JNJ": "Health", "PFE": "Health", "UNH": "Health", "MRK": "Health",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
}

# generate_price_panel()이 채워두는, 실제 거래일 목록 캐시.
# generate_news_articles()가 기사를 target_date(거래일)에 매핑할 때 재사용한다
# (가격 조회를 중복하지 않기 위해 — 반드시 generate_price_panel()을 먼저 호출해야 함).
_TRADING_DAYS_CACHE = {}

# generate_news_articles()가 채워두는, article_id -> 원문 텍스트(헤드라인+요약) 캐시.
# 반환 DataFrame에는 컬럼 스키마를 유지하기 위해 원문을 넣지 않지만, sentiment_extraction.py의
# VADER/FinBERT/GPT-4o 감성 추출 단계가 article_id로 원문을 다시 찾아올 때 이 캐시를 사용한다.
_ARTICLE_TEXT_CACHE = {}


def generate_price_panel(tickers: list, n_days: int = 220, start_date: str = "2025-01-02") -> pd.DataFrame:
    """Alpaca Market Data API로 실제 OHLCV를 가져온다 (§6.1.1 M0 가격 feature 산출용).
    무료/paper 계정은 SIP가 아닌 IEX 피드만 접근 가능하므로 feed=DataFeed.IEX로 고정한다."""
    client = StockHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY)
    start = pd.Timestamp(start_date, tz="UTC")
    # 주말/공휴일을 감안해 넉넉히 조회한 뒤, 종목별로 최근 n_days 거래일만 남긴다.
    end = start + pd.Timedelta(days=int(n_days * 1.6) + 10)
    end = min(end, pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))  # IEX 무료 피드는 최근 15분 지연 제외

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        limit=10000,
    )
    bars = client.get_stock_bars(request).df.reset_index()

    rows = []
    for t in tickers:
        sector = TICKER_SECTOR.get(t, "Unknown")
        sub = bars[bars["symbol"] == t].sort_values("timestamp").head(n_days)
        for _, r in sub.iterrows():
            rows.append({
                "date": pd.Timestamp(r["timestamp"]).normalize().tz_localize(None),
                "ticker": t, "sector": sector,
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": int(r["volume"]),
            })

    panel = pd.DataFrame(rows)
    _TRADING_DAYS_CACHE["dates"] = pd.DatetimeIndex(sorted(panel["date"].unique()))
    return panel


def compute_market_returns(price_panel: pd.DataFrame) -> pd.Series:
    """§6.1.1 '시장 요인': 전 종목 동일가중 평균 수익률"""
    panel = price_panel.sort_values(["ticker", "date"]).copy()
    panel["ret_1d"] = panel.groupby("ticker")["close"].pct_change(1)
    return panel.groupby("date")["ret_1d"].mean()


def generate_news_articles(tickers: list, n_days: int = 220, start_date: str = "2025-01-02",
                            avg_articles_per_stock_day: float = 1.2) -> pd.DataFrame:
    """
    §5.6 뉴스 기사 수집: Alpaca News API(Benzinga)에서 실제 기사를 가져와 발행시각과
    태깅 종목 수만 채운다. 감성 점수(vader_compound, finbert_*, llm_*)와 diffusion_raw는
    모듈 docstring에 설명한 대로 아직 계산 로직이 없어 NaN이다.

    avg_articles_per_stock_day는 합성 데이터 전용 파라미터라 실제 수집에는 사용하지 않는다
    (인터페이스 호환을 위해 파라미터만 유지).
    """
    trading_days = _TRADING_DAYS_CACHE.get("dates")
    if trading_days is None:
        raise RuntimeError("generate_price_panel()을 먼저 호출해야 합니다 (거래일 목록 캐시 필요)")

    client = NewsClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY)
    start = pd.Timestamp(start_date, tz="UTC")
    end = start + pd.Timedelta(days=int(n_days * 1.6) + 10)
    end = min(end, pd.Timestamp.now(tz="UTC"))

    request = NewsRequest(
        symbols=",".join(tickers),
        start=start, end=end,
        limit=10000,
        include_content=False,
        exclude_contentless=True,
    )
    news_set = client.get_news(request)
    articles = news_set.news if hasattr(news_set, "news") else news_set.data.get("news", [])

    # cutoff(§5.4, config.CUTOFF_TIME=09:25 ET) 기준으로 각 기사를 실제 거래일(target_date)에
    # 매핑한다: D-1 15:30(장마감 근사) ~ D 09:25 사이에 발행된 기사는 target_date=D.
    cutoff_h, cutoff_m = (int(x) for x in cfg.CUTOFF_TIME.split(":"))
    cutoffs_et = (trading_days + pd.Timedelta(hours=cutoff_h, minutes=cutoff_m)).tz_localize("America/New_York")
    cutoffs_utc = cutoffs_et.tz_convert("UTC")

    rows = []
    for art in articles:
        published_at = pd.Timestamp(art.created_at)
        pos = cutoffs_utc.searchsorted(published_at, side="right")
        if pos >= len(trading_days):
            continue  # 조회 기간 밖(마지막 거래일 cutoff 이후)으로 밀려난 기사는 제외
        target_date = trading_days[pos]
        _ARTICLE_TEXT_CACHE[art.id] = f"{art.headline}. {art.summary or ''}".strip()

        for t in art.symbols:
            if t not in tickers:
                continue
            rows.append({
                "article_id": art.id, "ticker": t, "published_at": published_at,
                "target_date": target_date,
                "n_tagged_tickers": len(art.symbols),
                "vader_compound": np.nan,
                "finbert_p_pos": np.nan, "finbert_p_neg": np.nan, "finbert_p_neutral": np.nan,
                "llm_polarity": np.nan, "llm_intensity": np.nan, "llm_uncertainty": np.nan,
                "diffusion_raw": np.nan,
            })

    return pd.DataFrame(rows, columns=[
        "article_id", "ticker", "published_at", "target_date", "n_tagged_tickers",
        "vader_compound", "finbert_p_pos", "finbert_p_neg", "finbert_p_neutral",
        "llm_polarity", "llm_intensity", "llm_uncertainty", "diffusion_raw",
    ])
