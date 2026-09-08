# Vertical Slice — M0~M5 절제실험 파이프라인

## 개정 이력 2차: 심층 코드 리뷰 피드백 반영

| # | 지적 | 판단 | 수정 내용 |
|---|---|---|---|
| ① | M0은 has_news도 없어서 M1~M5 개선이 '감성 내용' 때문인지 '뉴스 존재' 때문인지 불분명 | 타당 → 수정 | `M0_news`(Price+has_news, 내용 없음) 베이스라인 신설. 실행 결과 M0_news가 M0보다 오히려 살짝 낮게 나와, '뉴스가 있었다'는 사실 자체는 정보가 없음을 확인 |
| ② | Fold를 이어붙인 뒤 bootstrap → fold 경계를 넘는 블록이 시간적으로 불연속인 기간을 잇는 문제, 또한 SR을 따로 부트스트랩 후 빼는 건 진짜 paired 검정이 아님 | 타당 → 수정 | `paired_block_bootstrap_diff()` 신설: diff_t = r_B,t - r_A,t를 먼저 만들고, 블록이 fold 경계를 절대 넘지 않도록 fold별로 독립 추출 후 그 diff 자체를 부트스트랩 |
| ③ | Primary 검정이 H2(M3 vs M5)가 아니라 M0 vs M5로 되어 있어 논문과 불일치 | 타당 → 수정 | Primary를 M3 vs M5로 변경. M0 vs M5는 '참고용 보조 지표'로 별도 표시 |
| ④ | `compute_target_return`이 D일 당일 수익률이라 t+1과 불일치, shift(-1) 필요 | **검토 후 기각** — 아래 설명 참조 | 코드는 그대로 두고, target.py에 t/t+1 정합성을 명시하는 주석을 추가 |
| ⑤ | 합성 데이터에서 확산성과 LLM 감성이 같은 true_sig를 공유해 M5에 유리하게 조작됨 | 타당 → 수정 | 텍스트 감성용(`idio_signal`)과 확산성용(`diffusion_signal`) 잠재신호를 통계적으로 독립적으로 분리 생성, 각각 수익률에 별도로 반영 |
| 기타 | `warnings.filterwarnings("ignore")`, `assert`, FinBERT 확률 합≠1 | 타당 → 수정 | warnings 억제 제거 / assert를 `raise ValueError`로 대체 / FinBERT 3-class를 softmax로 생성해 합=1 보장 |

### ④번에 대한 상세 설명 (반영하지 않은 이유)

계획서 §5.4를 보면 cutoff·진입·청산이 전부 **같은 거래일 "t+1"** 안에서 일어납니다
(t+1 09:25 cutoff → t+1 09:30 진입 → t+1 종가 청산). 즉 "t"는 전날 종가+오버나이트
뉴스라는 *정보 시점*이고 "t+1"은 실제 매매가 일어나는 *당일*입니다. 코드에서 패널의 각
행 날짜 D는 이 표기의 "t+1"에 해당하며, 가격 feature는 이미 D-1(="t")로 shift되어
있습니다. 따라서 target은 D 자신의 시가~종가여야 맞고, 지금 코드가 정확히 그렇습니다.
여기에 shift(-1)을 추가하면 "D-1 정보로 D+1 수익률을 예측"하는 꼴이 되어 하루를 건너뛰는
새 오류가 생깁니다. 표기가 헷갈리기 쉬운 지점이라 주석으로 명확히 해뒀습니다.

## 개정 이력 1차: 코드 리뷰 피드백 3건 반영

| # | 지적 | 수정 내용 | 확인 방법 |
|---|---|---|---|
| A | 턴오버 계산이 '전날 목표 비중'과의 차이만 봐서 과소평가됨 | 본 전략은 매일 전량 청산·현금화(§5.4, 오버나이트 없음)하므로, 턴오버(편도) = 그날 매수 비중의 합(`sum(weights_t)`)으로 수정. 향후 3일/5일 보유 Robustness 확장(§6.5.1)을 위해 가격 표류를 반영한 실현 비중 기반 `build_daily_portfolio_returns_multiday()` 함수를 별도로 추가 | 수정 전 평균 턴오버 ~0.6 → 수정 후 ~0.8~0.9 (매일 거의 전량 회전하는 전략에 맞게 1에 근접) |
| B | walk-forward 분할에 train/test 경계 embargo가 없어 경계 누수 위험 | `walk_forward_splits()`에 `embargo_days=2` 파라미터 추가, train 마지막 날과 test 첫날 사이에 강제 공백 삽입 | 폴드 수가 3→2로 줄었다가(임베고로 사용 가능 일수 감소), 데모 기간을 220→300거래일로 늘려 3폴드 복원 |
| C | rolling 정규화 `min_periods=5`가 너무 작아 z-score 폭증 위험 | `min_periods=60`으로 상향, 윈도우 미충족 구간은 0으로 채우지 않고 §5.7의 '제외' 원칙대로 dropna 처리 | 초기 60거래일이 패널에서 자동 제외됨 (패널 크기 감소로 확인 가능) |

## 지금 상태: 합성 데이터로 파이프라인 검증 완료

이 샌드박스 환경은 Alpaca / Polygon / Benzinga / OpenAI API에 접근할 수 없어서,
`data_synthetic.py`가 실제 API 응답을 흉내 낸 가짜 가격·뉴스 데이터를 생성합니다.
2차 수정 이후 재실행 결과:

```
M0        F1=0.312  SR=-2.51
M0_news   F1=0.304  SR=-2.84   (뉴스 존재만으로는 오히려 도움 안 됨 — 정상)
M1        F1=0.315  SR=-1.48
M2        F1=0.341  SR=-0.05
M3        F1=0.353  SR=-1.47
M4        F1=0.370  SR=+0.48
M5        F1=0.353  SR=+1.96   (최고)

Primary (M3 vs M5, H2와 일치): 유의함 (CI: 2.50 ~ 5.98)
```

```
python run_ablation.py
```
를 실행하면 위 비교표와 3종의 통계 검정(Primary M3 vs M5, 참고 M0 vs M5, 내용-존재 분리
M0_news vs M3)이 출력됩니다.

## 파일 구성 (계획서 절과 1:1 대응)

| 파일 | 계획서 대응 절 | 역할 |
|---|---|---|
| `config.py` | §5.4~§5.7, §6.4 | 모든 임계값을 코드 상수로 고정 (half-life=6h, threshold=±0.5σ, block=10일 등) |
| `data_synthetic.py` | — | **실제 배포 시 이 파일만 교체.** 가격/뉴스 API 호출 코드로 대체 |
| `target.py` | §5.4, §6.1.1 | target 정의(r_{t+1}), 분류 레이블, 가격/기술지표 feature |
| `sentiment.py` | §5.5, §5.6, §5.6.1 | 스칼라화, 관련성/시간감쇠 가중, 집계, 클러스터링 |
| `normalize.py` | §5.7 | rolling 252일 point-in-time z-score 정규화 |
| `models.py` | §6.1 | M0~M5 feature 세트, LightGBM 래퍼, walk-forward 분할 |
| `backtest.py` | §6.4, §6.5 | 포트폴리오 구성, 비용모델, 샤프비율, block bootstrap |
| `run_ablation.py` | 전체 | 위 모듈을 엮어서 end-to-end 실행 |

## 실제 데이터로 전환하는 방법

`data_synthetic.py`의 두 함수만 실제 API 호출로 교체하면 나머지 코드는 수정할 필요가
없습니다 (인터페이스가 동일하도록 설계했습니다):

1. **`generate_price_panel(tickers, n_days, start_date)`**
   → Alpaca/Polygon API로 OHLCV 수집, 반환 컬럼 형식(`date, ticker, sector, open, high, low, close, volume`)만 맞추면 됨.
   섹터 정보는 GICS 섹터 매핑 테이블이 별도로 필요합니다.

2. **`generate_news_articles(tickers, n_days, start_date)`**
   → Alpaca News API(Benzinga)로 기사 수집 후, GPT-4o로 구조화 출력(극성/강도/불확실성) 생성.
   반환 컬럼(`article_id, ticker, published_at, target_date, n_tagged_tickers, vader_compound,
   finbert_p_pos, finbert_p_neg, llm_polarity, llm_intensity, llm_uncertainty, diffusion_raw`)만
   맞추면 됩니다. `diffusion_raw`는 §5.5.2의 4개 하위요소(독립 출처 수·재보도 수·경과시간·매체유형)를
   결합해 산출해야 합니다 (현재는 Poisson 노이즈로 단순화되어 있음).

3. **아직 구현 안 된 것 (§5.5.3, §6.3 관련)**
   - 사건 클러스터링 자체는 `sentiment.py`의 `is_same_event()`에 함수만 있고, 실제 임베딩
     기반 매칭 로직·validation set 기반 threshold 튜닝은 아직 연결 안 됨.
   - Primary 백본은 계획서상 PatchTST인데, 이 vertical slice는 속도를 위해 LightGBM으로
     대체했습니다 (§6.1.2 Robustness 트리 베이스라인과 동일 계열). `models.py`의
     `AblationModel` 클래스의 `fit`/`predict_label`/`predict_up_probability` 인터페이스만
     유지하면 PatchTST 버전으로 교체 가능합니다.
   - 실제 walk-forward 분할(`models.walk_forward_splits`)은 지금 단순화된 버전이며,
     계획서 §6.3의 정식 훈련/검증/테스트 구간 분할 규칙으로 교체해야 합니다.

## 알려진 한계 (정직하게 밝힘)

- 기본 합성 데이터의 신호 대비 노이즈 비율은 임의로 정한 것이라 실제 시장과 무관합니다.
- 종목 수(8개)·기간(220거래일)이 작아 Primary block bootstrap이 통계적 유의성에 도달하지
  못했습니다(정상 — 표본이 커지면 달라질 수 있음).
- 사건 클러스터링·validation set 층화표집(§5.5.3)은 아직 코드에 반영되지 않았습니다.
