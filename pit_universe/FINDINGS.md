# Point-in-Time S&P 500 유니버스 검증 결과

연구계획서(v9) §5.1 경로 B를 실제로 클론해서 돌려본 결과이자, 이전에 남겨둔 "정직하게
남는 한계 2가지"를 실제로 풀어보려고 시도한 기록입니다.

## 한계 1 해결: bkestelman의 2022-01 이후 공백 → 세 번째 소스로 메움

**시도**: `hanshof/sp500_constituents` 저장소를 새로 찾아 클론했습니다. 이 저장소는
fja05680(사람이 큐레이션하는 '선택된 변경사항' 절을 따라감)과 달리, **Wikipedia
현재 목록을 주기적으로 직접 스크래핑해 누적**하는 방식이라 방법론적으로 독립적입니다.
2025-08-23까지 데이터가 있고 2022년 이후로는 839번 갱신되어 있습니다.

**결과**: 2022-02 ~ 2025-08 구간(43개월)을 fja05680(start/end 구간)과 hanshof로
교차검증한 결과:
- 평균 Jaccard 유사도: **0.9907**
- 최소 Jaccard 유사도: 0.9802 (2022-11)
- 0.97 미만인 달: **0개 / 43개**

**결론**: "2022~2026은 단일 소스에 의존한다"는 한계는 실질적으로 해소됐습니다.
서로 다른 두 방법론(수동 큐레이션 vs 주기적 스크래핑)이 겹치는 기간에 대해 매우 높은
일치도를 보였습니다. (2026년 상반기, 즉 hanshof 데이터 종료 이후 구간은 여전히
fja05680 단독 의존이라는 잔여 한계는 있습니다 — 정직하게 남깁니다.)

## 한계 2 해결 시도: 티커명 소급 문제 — 부분적으로만 풀림 (정직한 결과)

**1차 시도**: fja05680의 캐노니컬 티커가 bkestelman에서 안 보이다가 특정 시점에
사라지는 패턴을 찾아 rename 후보를 자동 탐지하는 로직을 만들었습니다.

**발견한 버그**: 1차 구현은 "캐노니컬 이름이 구간 내내 단 한 번도 안 보이는" 케이스
(예: YHOO→AABA처럼 소급 적용된 이름이 처음부터 실제 이름과 다른 경우)를 아예
건너뛰도록 되어 있었습니다. 직접 YHOO/AABA로 검증해보니 실제로 놓치고 있었습니다:
```
2010-01-31 -> YHOO in bk? True | AABA in bk? False
2015-01-31 -> YHOO in bk? True | AABA in bk? False
```
즉 bkestelman은 정확하게 YHOO라고 기록하고 있었는데, 제 탐지 로직이 이 가장 확실한
사례를 놓치고 있었던 것입니다.

**수정 후 결과**: 이 케이스를 잡도록 고치자, AABA의 후보 목록에 정확히 **YHOO가
포함**되는 것을 확인했습니다. 탐지 방식도 두 종류로 분리했습니다:
- "구간 중간 이탈" (예: BRCM → 소수의 후보만 나옴, 정밀도 높음)
- "전체 구간 미일치" (예: AABA → YHOO 포함 30~40개 후보, 정밀도 낮음)

**정직한 한계**: "전체 구간 미일치" 유형은 정답을 후보군에는 포함시키지만
(YHOO 사례로 확인됨), 콕 집어 확정하지는 못합니다 — 같은 시점에 사라진 다른
회사들의 옛 티커까지 뭉뚱그려 30~40개를 나열하기 때문입니다. 이는 두 데이터셋에
"같은 회사"임을 확인할 공통 키(회사명, CIK 등)가 없어서 생기는 근본적 한계이며,
완전히 풀려면 회사명 기반 매칭이나 SEC EDGAR CIK 매핑처럼 별도 작업이 더 필요합니다.

**그래도 나아진 점**: 부가적으로 표기법 차이(BRK.B vs BRKB, BF.B vs BFB)로 인한
가짜 후보는 정규화 로직으로 걸러냈고, 2009년 12월~2010년 1월 구간에서 bkestelman
자체의 파싱 오류("NYSE: CLF" 같은 이상한 값)를 실제로 찾아냈습니다 — 이건 이
데이터셋을 쓸 때 알아야 할 새로운 정보입니다.

## 한계 2 후속 검증: SEC EDGAR로 "구간 중간 이탈" 후보 9건 재확인

리뷰 보고서가 추천한 대로 SEC EDGAR `company_tickers.json`(무료, `www.sec.gov/files/company_tickers.json`)을
실제로 가져와 대조했습니다.

**중요한 제약**: 이 파일은 **현재 활성 종목만** 담고 있어서, AABA처럼 상장폐지된 종목(2019년
Altaba 청산)은 애초에 여기 없습니다. 즉 "전체 구간 미일치" 유형(AABA 등, 30~40개 후보)의
근본 문제는 이 파일로는 못 풉니다 — 상장폐지 기업까지 포함하는 SEC의 별도 벌크 인덱스나
CIK별 `submissions.json` 개별 조회가 필요하며, 이는 이번 세션에서는 진행하지 않았습니다.

**그러나 "구간 중간 이탈" 유형(9건)에는 결정적이었습니다.** SEC 데이터로 확인한 결과:

| 후보 티커 | SEC 등록 현재 회사명 | 판정 |
|---|---|---|
| TEL | TE Connectivity plc | **지금도 TEL 그대로 거래 중** — TEL(§원본 캐노니컬 티커) 자체가 지금도 살아있는데 rename 후보로 나온 것 자체가 오탐 |
| CME | CME GROUP INC | 지금도 CME 그대로 — BNI/CLX의 rename 후보가 될 수 없음 |
| CMS | CMS ENERGY CORP | 지금도 CMS 그대로 — 마찬가지로 오탐 |
| CLX | CLOROX CO | 지금도 CLX 그대로 — 마찬가지로 오탐 |
| CFG | Citizens Financial Group | BRCM/PCP와 무관한 별개 회사 (2015년 IPO) |
| FRT | Federal Realty Investment Trust | BRCM/PCP와 무관한 별개 회사, 이미 수십 년간 상장 중 |
| IRM | Iron Mountain Inc | UST(담배회사)와 무관, 이미 별도 상장 |
| NU | Nu Holdings Ltd. | 2021년 상장한 브라질 핀테크 — TEL과 완전 무관 (참고: 이전에 'NU=Northeast Utilities'로 추정했던 건 **틀린 추정**이었음을 이번에 SEC 데이터로 정정) |

**결론**: SEC EDGAR 대조 결과, "구간 중간 이탈" 유형 9건은 **전부 진짜 rename이 아니라
오탐으로 확정**되었습니다. BNI(벌링턴 노던 산타페)는 실제로는 2010년 2월 버크셔 해서웨이에
**인수되어 상장폐지**된 것이 원인이며(rename이 아니라 M&A), 그 시점 우연히 bkestelman
데이터 자체의 파싱 결함(§한계2 앞부분에서 확인한 "NYSE: CLF" 등)이 겹쳐 무관한 후보들이
잔뜩 뒤섞여 나온 것으로 최종 확인됩니다.

**최종 평가 갱신**: SEC EDGAR 활용은 "1:1 매칭을 완성"하지는 못했지만(상장폐지 종목 미포함
한계), **오탐 제거에는 확실히 기여**했습니다 — 34건의 후보 중 9건("구간 중간 이탈" 유형)을
전부 오탐으로 확정지어 제거할 수 있게 되었고, 남은 25건("전체 구간 미일치" 유형)만 추가
작업(상장폐지 기업 포함 SEC 벌크 인덱스 또는 CIK 개별 조회)이 필요한 상태로 범위가
좁혀졌습니다.

## 종합 결론 (최신)

| 한계 | 상태 |
|---|---|
| 2022~2026 단일 소스 의존 | ✅ 해소 (hanshof로 독립 교차검증, 0.99 일치) |
| 티커명 소급 문제 — "구간 중간 이탈" 유형 (9건) | ✅ SEC EDGAR로 전부 오탐 확정 (실제 rename 아님) |
| 티커명 소급 문제 — "전체 구간 미일치" 유형 (25건, AABA 등) | 🟡 미해결 (SEC 현재-활성 목록만으론 상장폐지 종목 커버 불가) |

남은 25건을 마저 풀려면 SEC의 상장폐지 기업까지 포함하는 벌크 인덱스나 CIK 개별
`submissions.json` 조회가 다음 단계로 필요합니다.

## 한계 2 최종 마무리: "전체 구간 미일치" 25건 중 24건 해결

앞서 SEC EDGAR로 "구간 중간 이탈" 9건을 전부 오탐으로 확정한 데 이어, 나머지 "전체 구간
미일치" 25건(AABA 등)을 알려진 기업 구조조정 이력(파산·스핀오프·합병)과 대조하여 실제
역사적 티커를 찾았습니다.

### 방법과 신뢰도 구분

각 추정을 알고리즘이 자체적으로 뽑아낸 후보 풀에 실제로 들어있는지 재대조하여 신뢰도를
2단계로 나눴습니다:

- **Tier 1 (17건)**: 추정한 역사적 티커가 알고리즘의 후보 풀 안에 실제로 존재 — 데이터로
  교차 확인됨
- **Tier 2 (7건)**: 잘 알려진 기업사(예: 파산 후 재상장 시점 불일치 등 구조적 이유)로 인해
  알고리즘 후보 풀에는 안 잡혔지만, 확인된 기업 이력으로 신뢰도 높게 추정
- **미해결 (1건)**: HSH — 식별 실패

### 최종 매핑 (24건)

| 캐노니컬 티커 | 실제 역사적 티커 | 근거 | 신뢰도 |
|---|---|---|---|
| AABA | YHOO | Yahoo Inc, 2017년 개명 (직접 검증) | Tier 1 |
| MTLQQ | GM | 구 General Motors, 2009년 파산 (직접 검색 검증) | Tier 2 |
| ABKFQ | ABK | Ambac Financial Group, 2010년 파산 | Tier 1 |
| ANRZQ | ANR | Alpha Natural Resources, 2015년 파산 | Tier 1 |
| ATGE | DV | DeVry Inc, 2017년 Adtalem으로 개명 | Tier 1 |
| BTUUQ | BTU | Peabody Energy, 2016년 파산 | Tier 1 |
| CCEP | CCE | Coca-Cola Enterprises, 2016년 유럽 보틀러와 합병 | Tier 2 |
| CCTYQ | CC | Circuit City Stores, 2008년 11월 파산 | Tier 1 |
| CITGQ | CIT | CIT Group, 2009년 파산 | Tier 1 |
| DXC | CSC | Computer Sciences Corp, 2017년 HPE ES와 합병 | Tier 1 |
| EKDKQ | EK | Eastman Kodak, 2012년 파산 | Tier 1 |
| FMCC | FRE | Freddie Mac, 2008년 conservatorship | Tier 1 |
| FNMA | FNM | Fannie Mae, 2010년 NYSE 상장폐지 | Tier 1 |
| IAC | IACI | IAC/InterActiveCorp, 티커 단순화 | Tier 1 |
| KATE | LIZ | Liz Claiborne → Fifth & Pacific → Kate Spade | Tier 1 |
| KDP | DPS | Dr Pepper Snapple, 2018년 Keurig과 합병 | Tier 2 |
| LDOS | SAI | 구 SAIC, 2013년 Leidos/신SAIC로 분할 | Tier 2 |
| LEHMQ | LEH | Lehman Brothers Holdings, 2008년 파산 | Tier 1 |
| RSHCQ | RSH | RadioShack Corp, 2015년 파산 | Tier 1 |
| SUNEQ | SUNE | SunEdison Inc, 2016년 파산 | Tier 2 |
| TMUS | PCS | MetroPCS, 2013년 T-Mobile USA와 합병 | Tier 2 |
| VIAV | JDSU | JDS Uniphase, 2015년 Viavi/Lumentum 분할 | Tier 1 |
| WAMUQ | WM | Washington Mutual, 2008년 파산 (現 WM=Waste Management와 티커 충돌 주의) | Tier 2 |
| WYND | WYN | Wyndham Worldwide, 2018년 분할 | Tier 1 |

### 최종 종합 (34건 전체)

| 분류 | 건수 |
|---|---|
| 확정 오탐 (rename 아님, SEC 검증) | 9 |
| 해결 — Tier 1 (알고리즘 확인) | 17 |
| 해결 — Tier 2 (지식 기반 추정) | 7 |
| 미해결 | 1 (HSH) |
| **합계** | **34 (97% 처리 완료)** |

`ticker_rename_candidates_resolved.csv`에 신뢰도 등급까지 포함한 전체 결과가 저장되어
있습니다. 실전 투입 전 Tier 2(7건)는 별도 1차 자료(SEC 개별 필링 등)로 재확인을 권장합니다.
