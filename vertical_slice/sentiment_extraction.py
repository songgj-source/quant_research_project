"""
sentiment_extraction.py
data_synthetic.generate_news_articles()가 NaN으로 비워둔 감성 점수 컬럼을
실제 감성 분석 도구로 채우는 후처리 단계 (§5.5, §5.6, §6.1의 M1/M2/M3~M5 대조군에 대응).

- vader_compound                      <- nltk VADER (사전 기반 레거시 도구, M1)
- finbert_p_pos/neg/neutral           <- ProsusAI/finbert (금융 특화 인코더, M2)
- llm_polarity/intensity/uncertainty  <- GPT-4o 구조화 출력 (M3~M5)

article_id -> 원문 텍스트(헤드라인+요약) 매핑은 data_synthetic.generate_news_articles()가
호출 시점에 채워두는 _ARTICLE_TEXT_CACHE에서 가져온다. 따라서 이 모듈의 함수들에 넘기는
DataFrame은 반드시 generate_news_articles()가 직접 반환한 것이어야 한다 (캐시가 비어있지
않은 상태).

diffusion_raw은 이 단계에서 다루지 않는다 (§5.5.2의 4개 하위요소 산출에는 사건 클러스터링이
먼저 필요 — README에 명시된 별도 미해결 항목).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import data_synthetic as ds

_vader = SentimentIntensityAnalyzer()

_finbert_tokenizer = None
_finbert_model = None


def _get_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_model is None:
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert_model.eval()
    return _finbert_tokenizer, _finbert_model


def _article_text(article_id) -> str:
    text = ds._ARTICLE_TEXT_CACHE.get(article_id)
    if text is None:
        raise RuntimeError(
            f"article_id={article_id}의 원문 텍스트를 찾을 수 없습니다. "
            "generate_news_articles()가 직접 반환한 DataFrame을 그대로 넘겼는지 확인하세요."
        )
    return text


def add_vader_scores(articles: pd.DataFrame) -> pd.DataFrame:
    """vader_compound를 nltk VADER로 채운다 (M1 대조군)."""
    articles = articles.copy()
    articles["vader_compound"] = [
        _vader.polarity_scores(_article_text(aid))["compound"] for aid in articles["article_id"]
    ]
    return articles


def add_finbert_scores(articles: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """finbert_p_pos/neg/neutral을 ProsusAI/finbert로 채운다 (M2 대조군)."""
    tokenizer, model = _get_finbert()
    articles = articles.copy()
    texts = [_article_text(aid) for aid in articles["article_id"]]
    label2idx = {label: idx for idx, label in model.config.id2label.items()}

    probs_all = np.zeros((len(texts), 3))
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs_all[i:i + batch_size] = torch.softmax(logits, dim=-1).numpy()

    articles["finbert_p_pos"] = probs_all[:, label2idx["positive"]]
    articles["finbert_p_neg"] = probs_all[:, label2idx["negative"]]
    articles["finbert_p_neutral"] = probs_all[:, label2idx["neutral"]]
    return articles


_LLM_SYSTEM_PROMPT = """당신은 금융 뉴스 감성 분석 전문가입니다. 주어진 종목에 대한 뉴스 기사를
읽고 다음 세 가지를 -1~1 또는 0~1 범위의 숫자로만 평가하세요 (설명 문장 없이 값만):

- polarity: 이 뉴스가 해당 종목 주가에 미칠 방향성 톤. -1(매우 부정적) ~ 1(매우 긍정적), 0=중립.
- intensity: 감성 표현의 강도/확신 정도. 0(약하고 미온적) ~ 1(강하고 단정적).
- uncertainty: 텍스트에 내포된 정보의 불확실성. 추측성·조건부·루머성 표현이 많을수록 1에 가깝고,
  확정된 사실 위주면 0에 가깝다.
"""

_LLM_JSON_SCHEMA = {
    "name": "sentiment_scores",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "polarity": {"type": "number", "minimum": -1, "maximum": 1},
            "intensity": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["polarity", "intensity", "uncertainty"],
        "additionalProperties": False,
    },
}


def _call_gpt4o(client: OpenAI, ticker: str, text: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"[종목: {ticker}]\n{text}"},
        ],
        response_format={"type": "json_schema", "json_schema": _LLM_JSON_SCHEMA},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def add_llm_scores(articles: pd.DataFrame, model: str = "gpt-4o") -> pd.DataFrame:
    """llm_polarity/intensity/uncertainty를 GPT-4o 구조화 출력으로 채운다 (M3~M5)."""
    api_key = os.getenv("OPENAI_API_KEY")  # data_synthetic이 모듈 로드 시 이미 .env를 읽어둠
    if not api_key or api_key.startswith("여기에"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. api_setup/.env에 실제 OpenAI 키를 넣어주세요."
        )
    client = OpenAI(api_key=api_key)

    articles = articles.copy()
    polarity, intensity, uncertainty = [], [], []
    for aid, ticker in zip(articles["article_id"], articles["ticker"]):
        result = _call_gpt4o(client, ticker, _article_text(aid), model=model)
        polarity.append(result["polarity"])
        intensity.append(result["intensity"])
        uncertainty.append(result["uncertainty"])

    articles["llm_polarity"] = polarity
    articles["llm_intensity"] = intensity
    articles["llm_uncertainty"] = uncertainty
    return articles


def extract_all_sentiment(articles: pd.DataFrame, llm_model: str = "gpt-4o") -> pd.DataFrame:
    """vader_compound, finbert_*, llm_*를 한 번에 채운다. diffusion_raw는 그대로 NaN."""
    articles = add_vader_scores(articles)
    articles = add_finbert_scores(articles)
    articles = add_llm_scores(articles, model=llm_model)
    return articles
