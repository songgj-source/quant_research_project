# Quant Research Project — 코드 번들

미국 주식시장 주가 예측을 위한 다차원 LLM 기반 금융 뉴스 감성 분석 연구계획서(v10)에서
파생된 실행 가능한 코드 모음입니다. 세 개의 독립적인 하위 프로젝트로 구성되어 있습니다.

## 폴더 구성

### 1. `vertical_slice/` — M0~M5 절제실험 파이프라인
연구계획서 §9.1의 "vertical slice" 개념 구현체. 합성 데이터로 데이터 수집→감성 추출→
feature 생성→학습→백테스트→통계검정 전 과정이 정상 작동함을 검증한 코드입니다.
- `build_model.py` — 8개 모듈을 하나로 합친 단일 실행 파일 (권장 진입점)
- `config.py`, `sentiment.py`, `normalize.py`, `target.py`, `models.py`, `backtest.py`,
  `data_synthetic.py`, `run_ablation.py` — 동일 내용의 모듈 분리 버전 (읽기/수정용)
- `README.md` — 개정 이력, 실제 데이터 전환 가이드

실행: `cd vertical_slice && pip install numpy pandas scikit-learn lightgbm && python build_model.py`

### 2. `pit_universe/` — Point-in-time S&P 500 유니버스 검증
WRDS 없이 무료 데이터로 point-in-time 종목 유니버스를 확보·검증한 코드와 결과입니다.
- `pit_universe.py` — 3개 데이터소스 교차검증 + rename 후보 탐지 스크립트
- `sp500_ticker_start_end.csv`, `sp500_monthly.csv`, `hanshof_sp500_historical.csv` — 검증용 원본 데이터
- `ticker_rename_candidates_resolved.csv` — 34건 중 33건 해결된 최종 결과
- `FINDINGS.md` — 전체 검증 과정과 결론
- `VERIFICATION_PACKAGE.md` — 코드+로그+결론을 하나로 합친 재검증용 문서

실행: `cd pit_universe && pip install pandas numpy && python pit_universe.py`

### 3. `api_setup/` — 실거래 API 연동 준비
- `.env.template` — API 키 설정 템플릿 (반드시 `.env`로 이름 바꾼 뒤 실제 키 입력)
- `.gitignore` — `.env` 파일이 실수로 커밋되는 것을 방지
- `test_connection.py` — Alpaca 연결 테스트 스크립트

실행:
```bash
cd api_setup
cp .env.template .env
# .env 파일을 열어 실제 API 키로 교체 (절대 채팅/스크린샷으로 공유하지 말 것)
pip install python-dotenv alpaca-py
python test_connection.py
```

## 전체 설치 (한 번에)

```bash
pip install numpy pandas scikit-learn lightgbm python-dotenv alpaca-py --break-system-packages
```

## 주의사항

- `pit_universe/`의 CSV들은 fja05680, bkestelman, hanshof 세 GitHub 저장소에서 받은 원본
  스냅샷입니다. 최신 데이터로 갱신하려면 각 저장소를 다시 clone하세요.
- `vertical_slice/`는 현재 합성(가짜) 데이터로만 검증되었습니다. 실제 API 데이터로 전환하려면
  `data_synthetic.py`의 `generate_price_panel`, `generate_news_articles` 두 함수만
  실제 API 호출로 교체하면 됩니다 (인터페이스 동일).
- `api_setup/.env`는 실제 키가 들어가는 순간 **절대 깃에 커밋하거나 다른 곳에 공유하지 마세요**.
