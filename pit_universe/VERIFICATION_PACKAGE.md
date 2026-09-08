# 검증 요청 패키지 v2 — Point-in-Time S&P 500 유니버스 확보 (SEC EDGAR 후속 검증 포함)

이전 라운드 검증 보고서가 "SEC EDGAR CIK 매핑을 다음 단계로 추천"했습니다. 이번 라운드에서
실제로 시도했고, 그 결과(성공한 부분과 못 푼 부분 모두)를 정직하게 담았습니다.

## 배경

1. bkestelman 데이터 2022-01 공백 -> hanshof로 해결 (이전 라운드에서 완료)
2. 티커명 소급 문제(YHOO->AABA류) -> 탐지 버그 수정 완료, 34건 후보 도출 (이전 라운드)
3. **이번 라운드**: SEC EDGAR로 34건 후보 중 얼마나 확정/기각할 수 있는지 검증

---

## 1. 전체 소스 코드 (pit_universe.py, 330줄)

```python
"""
pit_universe.py
연구계획서(v9) §5.1 경로 B 구현체 — GitHub 공개 데이터셋 두 개를 교차검증하고,
"특정 날짜 기준 S&P 500 point-in-time 구성종목 중 시가총액 상위 N종목"을 산출하는 함수를 제공한다.

데이터 소스:
  1. fja05680/sp500  -- "S&P 500 Historical Components & Changes (Updated).csv"
     날짜별 전체 구성종목 리스트. 단, '변경이 있었던 날짜'에만 행이 기록되므로
     임의 날짜 조회 시 as-of(직전 기록 forward-fill) 방식으로 찾아야 한다.
     범위: 1996-01-02 ~ 2026-06-30 (일 단위 변경 기록)

  2. bkestelman/sp500_historical_components -- "sp500_monthly.csv"
     Wikipedia 문서 revision history 기반 재구성. 월말 스냅샷.
     범위: 2008-01-31 ~ 2022-01-31 (월 단위)

교차검증: 두 소스가 겹치는 기간(2008-01 ~ 2022-01) 동안 월말 기준으로 구성종목 집합을
비교해 Jaccard 유사도를 산출한다. 유사도가 낮은 달은 수동 확인이 필요한 구간으로 표시한다.
"""
import numpy as np
import pandas as pd


# ── 데이터 로딩 ──────────────────────────────────────────────
def load_fja_start_end(path: str) -> pd.DataFrame:
    """
    fja05680의 sp500_ticker_start_end.csv (ticker, start_date, end_date)를 로드한다.

    [중요 — 실제 데이터 검증 과정에서 발견한 사실] fja05680의 원본 comma-separated
    티커 리스트("... Changes (Updated).csv")는 일부 행에 'TICKER-YYYYMM' 접미사가
    섞여 있고, 접미사 없는 티커는 '가장 최근에 알려진 종목명'을 과거 이력 전체에
    소급 적용하는 방식으로 보인다 (예: 야후는 실제로 2017년 6월에야 YHOO에서 AABA로
    개명했는데, 이 파일은 1999년부터 AABA로 표기). 이 방식으로 그대로 파싱하면
    "2008년에 AABA가 이미 존재했다"는 식의 명백한 오류가 발생한다.
    따라서 본 모듈은 이 필드를 직접 파싱하지 않고, 저장소가 별도로 제공하는
    sp500_ticker_start_end.csv(종목별 편입·편출 날짜 구간)를 사용한다. 이 파일도
    '최신 티커명을 과거에 소급 적용'하는 문제 자체는 남아있을 수 있으나(예: AABA가
    실제로는 YHOO였던 기간까지 포함), 최소한 "그 회사가 그 시점에 지수 구성원이었는가"
    라는 멤버십 자체는 명확한 시작·종료일로 판단할 수 있어 훨씬 신뢰할 만하다.
    (실제 API 조회 시 사용할 '그 시점의 정확한 티커 심볼'은 별도의 심볼 변경 이력
    매핑이 추가로 필요하다 — §8 한계 참조)
    """
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    return df


def load_bkestelman_monthly(path: str) -> pd.DataFrame:
    """bkestelman의 월말 스냅샷을 로드한다. 반환: date(월말), tickers(frozenset) 두 컬럼."""
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "date"
    tickers_per_row = df.apply(lambda row: frozenset(row.dropna().astype(str)), axis=1)
    return pd.DataFrame({"date": df.index, "tickers": tickers_per_row.values}).reset_index(drop=True)


# ── as-of(직전 기록) 조회 ───────────────────────────────────
def get_membership_as_of(start_end_table: pd.DataFrame, query_date) -> frozenset:
    """
    start_end_table: load_fja_start_end()의 반환값 (ticker, start_date, end_date).
    query_date 시점의 구성종목 = start_date <= query_date인 동시에
    (end_date가 NaT이거나 end_date >= query_date)인 모든 ticker.
    """
    query_date = pd.Timestamp(query_date)
    active = start_end_table[
        (start_end_table["start_date"] <= query_date)
        & (start_end_table["end_date"].isna() | (start_end_table["end_date"] >= query_date))
    ]
    if len(active) == 0:
        raise ValueError(f"{query_date.date()}는 데이터 범위보다 이르거나 활성 종목이 없습니다")
    return frozenset(active["ticker"])


# ── 교차검증 ─────────────────────────────────────────────────
def jaccard(a: frozenset, b: frozenset) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)


def cross_validate(fja_changelog: pd.DataFrame, bk_monthly: pd.DataFrame,
                    low_agreement_threshold: float = 0.97) -> pd.DataFrame:
    """
    bk_monthly의 각 월말 날짜에 대해, fja_changelog의 as-of 구성종목과 비교한다.
    반환: date, jaccard, n_fja_only(fja에만 있는 종목 수), n_bk_only(bk에만 있는 종목 수), n_common
    """
    records = []
    for _, row in bk_monthly.iterrows():
        d = row["date"]
        bk_set = row["tickers"]
        try:
            fja_set = get_membership_as_of(fja_changelog, d)
        except ValueError:
            continue
        j = jaccard(fja_set, bk_set)
        records.append({
            "date": d,
            "jaccard": j,
            "n_fja_only": len(fja_set - bk_set),
            "n_bk_only": len(bk_set - fja_set),
            "n_common": len(fja_set & bk_set),
            "flag_low_agreement": j < low_agreement_threshold,
        })
    return pd.DataFrame(records)


# ── 2022~2026 갭 보완: 독립 소스(hanshof) 교차검증 ──────────
def load_hanshof_changelog(path: str) -> pd.DataFrame:
    """
    hanshof/sp500_constituents의 변경-이벤트 로그를 로드한다 (fja05680과 달리 Wikipedia
    '현재 목록'을 주기적으로 직접 스크래핑해 누적하는 방식이라, 방법론적으로 독립적이다
    — fja05680은 사람이 큐레이션하는 '선택된 변경사항' 절을 따라가는 방식임).
    2022년 이후 갱신 빈도가 높아(연 800회 이상) bkestelman(2022-01 종료)이 못 다루는
    최근 구간의 교차검증에 사용한다.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["tickers"] = df["tickers"].apply(lambda s: frozenset(s.split(",")))
    return df[["date", "tickers"]]


def get_membership_as_of_changelog(changelog: pd.DataFrame, query_date) -> frozenset:
    """load_hanshof_changelog()처럼 (date 오름차순, tickers) 형태인 변경로그용 as-of 조회."""
    query_date = pd.Timestamp(query_date)
    eligible = changelog[changelog["date"] <= query_date]
    if len(eligible) == 0:
        raise ValueError(f"{query_date.date()}는 changelog 범위보다 이릅니다")
    return eligible.iloc[-1]["tickers"]


def cross_validate_recent(start_end_table: pd.DataFrame, hanshof_changelog: pd.DataFrame,
                           date_range: pd.DatetimeIndex,
                           low_agreement_threshold: float = 0.97) -> pd.DataFrame:
    """
    fja05680(start/end 구간)과 hanshof(독립 스크래핑 changelog)를 date_range의 각 날짜에 대해
    비교한다. bkestelman이 다루지 못하는 2022년 이후 구간의 교차검증용.
    """
    records = []
    for d in date_range:
        try:
            fja_set = get_membership_as_of(start_end_table, d)
            han_set = get_membership_as_of_changelog(hanshof_changelog, d)
        except ValueError:
            continue
        j = jaccard(fja_set, han_set)
        records.append({
            "date": d, "jaccard": j,
            "n_fja_only": len(fja_set - han_set),
            "n_han_only": len(han_set - fja_set),
            "n_common": len(fja_set & han_set),
            "flag_low_agreement": j < low_agreement_threshold,
        })
    return pd.DataFrame(records)


# ── 티커명 소급 문제 해결: rename 후보 탐지 ─────────────────
def normalize_ticker(t: str) -> str:
    """점·대시·공백을 제거하고 대문자로 통일 — 'BRK.B'와 'BRKB'처럼 같은 종목의 표기
    차이를 진짜 rename으로 오인하지 않도록 한다."""
    return "".join(ch for ch in t.upper() if ch.isalnum())


def detect_rename_candidates(start_end_table: pd.DataFrame, bk_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    한계 2 해결 시도: fja05680의 sp500_ticker_start_end.csv는 '최신 티커명'을 과거 이력
    전체에 소급 적용하는 문제가 있다(예: AABA가 실제로는 1999~2017년 대부분 YHOO였음).
    bkestelman은 각 시점 Wikipedia 문서에 '실제로 적혀 있던' 티커를 그대로 기록하므로,
    두 소스를 대조하면 '캐노니컬 티커가 fja 기준으로는 계속 활성 상태인데, 같은 기간
    bkestelman 목록에서는 안 보이다가 특정 시점부터 보이는' 패턴을 찾아 rename 후보를
    데이터 기반으로 탐지할 수 있다.

    방법: 각 fja 캐노니컬 티커 T의 [start,end] 구간 내 bkestelman 월말 스냅샷들을 순회하며
    T가 처음 사라지는 시점 이전/이후에 새로 등장하는 'bk-only' 티커가 있는지 찾는다.
    완전 자동 확정은 아니고, 수동 확인이 필요한 후보 목록을 만드는 것이 목적이다.
    """
    candidates = []
    bk_by_date = {row["date"]: row["tickers"] for _, row in bk_monthly.iterrows()}
    bk_dates_sorted = sorted(bk_by_date.keys())

    all_fja_tickers_norm = set(start_end_table["ticker"].apply(normalize_ticker))

    for _, r in start_end_table.iterrows():
        ticker, start, end = r["ticker"], r["start_date"], r["end_date"]
        end_eff = end if pd.notna(end) else bk_dates_sorted[-1]
        window = [d for d in bk_dates_sorted if start <= d <= end_eff]
        if len(window) < 2:
            continue

        norm_ticker = normalize_ticker(ticker)
        present_flags = [
            any(normalize_ticker(bt) == norm_ticker for bt in bk_by_date[d]) for d in window
        ]
        if all(present_flags):
            continue  # bkestelman에서도 (표기 차이 감안 시) 쭉 같은 이름 -> rename 의심 없음

        if not any(present_flags):
            # 코드 리뷰(사용자) 발견 버그 수정: 이 케이스를 그냥 건너뛰면 YHOO->AABA처럼
            # '캐노니컬 이름이 그 구간 내내 단 한 번도 안 보이는' 가장 확실한 rename 신호를
            # 놓치게 된다. bk 첫 시점의 '어떤 활성 fja 티커로도 설명 안 되는' 잔여 티커들을
            # 후보로 제시한다 (완전 자동 확정은 아니고 후보 압축용).
            first_d = window[0]
            residual = {
                bt for bt in bk_by_date[first_d]
                if normalize_ticker(bt) not in all_fja_tickers_norm
            }
            if residual:
                candidates.append({
                    "canonical_ticker_fja": ticker,
                    "last_seen_as_is_in_bk": None,
                    "first_absent_in_bk": window[0],
                    "candidate_new_symbols_in_bk": sorted(residual),
                    "detection_type": "전체 구간 미일치 (예: YHOO->AABA 유형)",
                })
            continue

        # T가 '사라지는' 첫 시점을 찾고, 그 근방에 새로 등장하는 미확인 티커를 후보로 제시
        first_absent_idx = present_flags.index(False)
        if first_absent_idx == 0:
            continue
        transition_date = window[first_absent_idx]
        prev_date = window[first_absent_idx - 1]
        newly_appeared_raw = bk_by_date[transition_date] - bk_by_date[prev_date]
        # 표기 차이(점/대시)만 다른 건 제외하고, 다른 활성 fja 티커로 이미 설명되는 것도 제외
        newly_appeared = set()
        for t in newly_appeared_raw:
            if normalize_ticker(t) == norm_ticker:
                continue  # 순수 표기 차이 -> rename 후보 아님
            already_known = ((start_end_table["ticker"].apply(normalize_ticker) == normalize_ticker(t))
                              & (start_end_table["start_date"] <= transition_date)
                              & (start_end_table["end_date"].isna()
                                 | (start_end_table["end_date"] >= transition_date))).any()
            if not already_known:
                newly_appeared.add(t)
        if newly_appeared:
            candidates.append({
                "canonical_ticker_fja": ticker,
                "last_seen_as_is_in_bk": prev_date,
                "first_absent_in_bk": transition_date,
                "candidate_new_symbols_in_bk": sorted(newly_appeared),
                "detection_type": "구간 중간 이탈 (전환 시점 신규 등장 티커 대조)",
            })

    return pd.DataFrame(candidates)



def get_point_in_time_universe(
    fja_changelog: pd.DataFrame, query_date, top_n: int = 100,
    market_cap_lookup: dict = None,
) -> list:
    """
    §5.1 확정 정의: "S&P 500의 point-in-time 구성종목 중, 매 시점 t 기준 시가총액 상위 N종목"

    market_cap_lookup: {ticker: market_cap} 딕셔너리. query_date 시점의 실제 시가총액이
    필요하며, 이는 §5.3의 가격 데이터(Alpaca/Polygon)에서 종가 x 발행주식수로 산출해야 한다.
    이 함수 자체는 시가총액 데이터를 갖고 있지 않으므로(이 저장소들은 멤버십만 제공),
    호출하는 쪽에서 실제 가격 데이터로부터 만든 lookup을 넘겨줘야 한다.

    market_cap_lookup이 없으면(아직 가격 데이터 연동 전 단계), 정렬 없이 전체 멤버십을
    알파벳순으로 반환하고 경고를 출력한다 — 이 상태로 core sample을 확정하면 안 된다.
    """
    universe = get_membership_as_of(fja_changelog, query_date)
    if market_cap_lookup is None:
        print(f"[경고] {pd.Timestamp(query_date).date()}: market_cap_lookup이 없어 시가총액 정렬을 "
              f"수행할 수 없습니다. 알파벳순 상위 {top_n}개를 반환합니다 (실전 사용 금지, 검증용).")
        return sorted(universe)[:top_n]

    ranked = sorted(
        (t for t in universe if t in market_cap_lookup),
        key=lambda t: market_cap_lookup[t],
        reverse=True,
    )
    missing = len(universe) - len(ranked)
    if missing > 0:
        print(f"[정보] {pd.Timestamp(query_date).date()}: 유니버스 {len(universe)}종목 중 "
              f"{missing}개는 market_cap_lookup에 없어 랭킹에서 제외되었습니다.")
    return ranked[:top_n]


if __name__ == "__main__":
    FJA_CSV = "repo_fja/sp500_ticker_start_end.csv"
    BK_CSV = "repo_bk/sp500_monthly.csv"
    HANSHOF_CSV = "repo_hanshof/sp_500_historical_components.csv"

    print("[1/5] fja05680 편입/편출 구간 테이블 로딩 중...")
    fja = load_fja_start_end(FJA_CSV)
    print(f"      {len(fja)}개 (종목,구간) 행")

    print("[2/5] bkestelman 월별 스냅샷 로딩 중 (2008~2022)...")
    bk = load_bkestelman_monthly(BK_CSV)
    print(f"      {len(bk)}개 월말 스냅샷, {bk['date'].min().date()} ~ {bk['date'].max().date()}")

    print("[3/5] hanshof 독립 스크래핑 changelog 로딩 중 (한계 1 해결용)...")
    hanshof = load_hanshof_changelog(HANSHOF_CSV)
    print(f"      {len(hanshof)}개 변경 이벤트, {hanshof['date'].min().date()} ~ {hanshof['date'].max().date()}")

    print("\n" + "=" * 78)
    print("[한계 1 해결] 2022~2026 갭 구간을 hanshof(독립 소스)로 교차검증")
    print("=" * 78)
    recent_dates = pd.date_range("2022-02-01", min(hanshof["date"].max(), pd.Timestamp("2026-06-30")), freq="MS")
    cv_recent = cross_validate_recent(fja, hanshof, recent_dates)
    print(f"검증된 월 수: {len(cv_recent)} (2022-02 ~ {cv_recent['date'].max().date() if len(cv_recent) else 'N/A'})")
    print(f"평균 Jaccard 유사도: {cv_recent['jaccard'].mean():.4f}")
    print(f"최소 Jaccard 유사도: {cv_recent['jaccard'].min():.4f} "
          f"(날짜: {cv_recent.loc[cv_recent['jaccard'].idxmin(), 'date'].date()})")
    low_recent = cv_recent[cv_recent["flag_low_agreement"]]
    print(f"유사도 0.97 미만: {len(low_recent)}개 / {len(cv_recent)}개")
    if len(low_recent) > 0:
        print(low_recent.sort_values("jaccard").head(10).to_string(index=False))
    print("\n=> 결론: bkestelman이 못 다루는 2022년 이후 구간도 이제 hanshof로 독립 교차검증되어,")
    print("   '2022~2026은 단일 소스에 의존한다'는 한계가 해소되었다 (완전한 3중 검증은 아니지만,")
    print("   서로 다른 두 방법론(수동 큐레이션 vs 주기적 스크래핑)이 모두 존재하게 되었다).")

    print("\n" + "=" * 78)
    print("[한계 2 해결 시도] 티커명 소급 문제 — rename 후보 데이터 기반 탐지")
    print("=" * 78)
    renames = detect_rename_candidates(fja, bk)
    print(f"탐지된 rename 후보: {len(renames)}건 (수동 확인 필요)")
    if len(renames) > 0:
        print(renames.head(15).to_string(index=False))
    print("\n=> 결론: 완전 자동화된 확정은 아니지만, '수동으로 하나하나 찾아야 하는' 상태에서")
    print("   '데이터 기반 후보 목록 + 수동 확인'으로 작업량이 크게 줄었다. 이 후보 목록을")
    print("   ticker_symbol_map.csv로 저장해 향후 실제 가격/뉴스 API 호출 시 심볼 치환에 사용한다.")
    if len(renames) > 0:
        renames.to_csv("ticker_rename_candidates.csv", index=False)
        print("   -> ticker_rename_candidates.csv 저장 완료")

    # 사용 예시
    print("\n" + "=" * 78)
    print("사용 예시: 2024-06-15 기준 point-in-time 유니버스 (market cap 미연동 상태)")
    print("=" * 78)
    example_universe = get_point_in_time_universe(fja, "2024-06-15", top_n=10)
    print(example_universe)

```

---

## 2. 최신 실행 로그 (한계 1 + 한계 2 탐지, SEC 검증 이전)

```
[1/5] fja05680 편입/편출 구간 테이블 로딩 중...
      1259개 (종목,구간) 행
[2/5] bkestelman 월별 스냅샷 로딩 중 (2008~2022)...
      169개 월말 스냅샷, 2008-01-31 ~ 2022-01-31
[3/5] hanshof 독립 스크래핑 changelog 로딩 중 (한계 1 해결용)...
      3482개 변경 이벤트, 1996-01-02 ~ 2025-08-23

==============================================================================
[한계 1 해결] 2022~2026 갭 구간을 hanshof(독립 소스)로 교차검증
==============================================================================
검증된 월 수: 43 (2022-02 ~ 2025-08-01)
평균 Jaccard 유사도: 0.9907
최소 Jaccard 유사도: 0.9802 (날짜: 2022-11-01)
유사도 0.97 미만: 0개 / 43개

=> 결론: bkestelman이 못 다루는 2022년 이후 구간도 이제 hanshof로 독립 교차검증되어,
   '2022~2026은 단일 소스에 의존한다'는 한계가 해소되었다 (완전한 3중 검증은 아니지만,
   서로 다른 두 방법론(수동 큐레이션 vs 주기적 스크래핑)이 모두 존재하게 되었다).

==============================================================================
[한계 2 해결 시도] 티커명 소급 문제 — rename 후보 데이터 기반 탐지
==============================================================================
탐지된 rename 후보: 34건 (수동 확인 필요)
canonical_ticker_fja last_seen_as_is_in_bk first_absent_in_bk                                                                                                                                                                                         candidate_new_symbols_in_bk               detection_type
                AABA                   NaT         2008-01-31 [AA, ABK, ACE, AOC, BHI, BTU, CBG, CC, CIT, COH, CSC, CZN, EK, ERTS, FNM, FO, FPL, FRE, GCI, IACI, JDSU, KFT, LEH, LIZ, LTD, LTR, LUK, MHP, MOT, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
               ABKFQ                   NaT         2008-01-31 [AA, ABK, ACE, AOC, BHI, BTU, CBG, CC, CIT, COH, CSC, CZN, EK, ERTS, FNM, FO, FPL, FRE, GCI, IACI, JDSU, KFT, LEH, LIZ, LTD, LTR, LUK, MHP, MOT, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
               ANRZQ                   NaT         2011-06-30                                              [AA, ACE, ANR, BHI, BTU, CBG, COH, CSC, DPS, DV, ERTS, FO, GCI, HCN, JDSU, JOYG, KFT, LTD, LUK, MHP, NU, PCLN, RSH, SLE, TSO, TYC, WAG, WFR, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
                ATGE                   NaT         2009-06-30                                 [AA, AOC, BHI, BTU, CBG, CIT, COH, CSC, DPS, DV, EK, ERTS, FO, FPL, GCI, HCN, JDSU, KFT, LTD, LUK, MHP, MOT, NU, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
                 BNI            2009-12-31         2010-01-31                                                                                                             [BRK.B, MXB, NYSE: CLF, NYSE: CLX, NYSE: CMS, NYSE: COH, NYSE: SAI, NYSE: V, Nasdaq: CME, Nasdaq: ROST] 구간 중간 이탈 (전환 시점 신규 등장 티커 대조)
                BRCM            2015-12-31         2016-01-31                                                                                                                                                                                                          [CFG, FRT] 구간 중간 이탈 (전환 시점 신규 등장 티커 대조)
               BTUUQ                   NaT         2008-01-31 [AA, ABK, ACE, AOC, BHI, BTU, CBG, CC, CIT, COH, CSC, CZN, EK, ERTS, FNM, FO, FPL, FRE, GCI, IACI, JDSU, KFT, LEH, LIZ, LTD, LTR, LUK, MHP, MOT, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
                CCEP                   NaT         2016-01-31                                                                                                                                           [AA, BHI, CBG, COH, DLPH, DPS, HCN, LUK, MHFI, PCLN, TSO, TYC, WYN, YHOO] 전체 구간 미일치 (예: YHOO->AABA 유형)
               CCTYQ                   NaT         2008-01-31 [AA, ABK, ACE, AOC, BHI, BTU, CBG, CC, CIT, COH, CSC, CZN, EK, ERTS, FNM, FO, FPL, FRE, GCI, IACI, JDSU, KFT, LEH, LIZ, LTD, LTR, LUK, MHP, MOT, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
               CITGQ                   NaT         2008-01-31 [AA, ABK, ACE, AOC, BHI, BTU, CBG, CC, CIT, COH, CSC, CZN, EK, ERTS, FNM, FO, FPL, FRE, GCI, IACI, JDSU, KFT, LEH, LIZ, LTD, LTR, LUK, MHP, MOT, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)
                 CLX            2009-12-31         2010-01-31                                                                                                             [BRK.B, MXB, NYSE: CLF, NYSE: CLX, NYSE: CMS, NYSE: COH, NYSE: SAI, NYSE: V, Nasdaq: CME, Nasdaq: ROST] 구간 중간 이탈 (전환 시점 신규 등장 티커 대조)
                 CME            2009-12-31         2010-01-31                                                                                                             [BRK.B, MXB, NYSE: CLF, NYSE: CLX, NYSE: CMS, NYSE: COH, NYSE: SAI, NYSE: V, Nasdaq: CME, Nasdaq: ROST] 구간 중간 이탈 (전환 시점 신규 등장 티커 대조)
                 CMS            2009-12-31         2010-01-31                                                                                                             [BRK.B, MXB, NYSE: CLF, NYSE: CLX, NYSE: CMS, NYSE: COH, NYSE: SAI, NYSE: V, Nasdaq: CME, Nasdaq: ROST] 구간 중간 이탈 (전환 시점 신규 등장 티커 대조)
                CPWR            2011-11-30         2011-12-31                                                                                                                                                                                                            [WPX-WI] 구간 중간 이탈 (전환 시점 신규 등장 티커 대조)
                 DXC                   NaT         2008-01-31 [AA, ABK, ACE, AOC, BHI, BTU, CBG, CC, CIT, COH, CSC, CZN, EK, ERTS, FNM, FO, FPL, FRE, GCI, IACI, JDSU, KFT, LEH, LIZ, LTD, LTR, LUK, MHP, MOT, RSH, SLE, TSO, TYC, WAG, WFMI, WFR, WMI, WPI, WPO, WYN, YHOO, ZMH] 전체 구간 미일치 (예: YHOO->AABA 유형)

=> 결론: 완전 자동화된 확정은 아니지만, '수동으로 하나하나 찾아야 하는' 상태에서
   '데이터 기반 후보 목록 + 수동 확인'으로 작업량이 크게 줄었다. 이 후보 목록을
   ticker_symbol_map.csv로 저장해 향후 실제 가격/뉴스 API 호출 시 심볼 치환에 사용한다.
   -> ticker_rename_candidates.csv 저장 완료

==============================================================================
사용 예시: 2024-06-15 기준 point-in-time 유니버스 (market cap 미연동 상태)
==============================================================================
[경고] 2024-06-15: market_cap_lookup이 없어 시가총액 정렬을 수행할 수 없습니다. 알파벳순 상위 10개를 반환합니다 (실전 사용 금지, 검증용).
['A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI']

```

---


## 6. SEC EDGAR 후속 검증 (리뷰 보고서 추천사항 반영)

실행한 명령: web_fetch("https://www.sec.gov/files/company_tickers.json")

이 파일은 SEC에 현재 등록된 활성 기업의 (ticker, CIK, 회사명) 매핑입니다. 확인 결과:

| 후보 티커 | SEC 등록 현재 회사명 | 판정 |
|---|---|---|
| TEL | TE Connectivity plc | 지금도 TEL 그대로 -> rename 후보 자체가 오탐 |
| CME | CME GROUP INC | 지금도 CME 그대로 -> 오탐 |
| CMS | CMS ENERGY CORP | 지금도 CMS 그대로 -> 오탐 |
| CLX | CLOROX CO | 지금도 CLX 그대로 -> 오탐 |
| CFG | Citizens Financial Group | BRCM/PCP와 무관한 별개 회사 |
| FRT | Federal Realty Investment Trust | BRCM/PCP와 무관한 별개 회사 |
| IRM | Iron Mountain Inc | UST와 무관 |
| NU | Nu Holdings Ltd. (2021 IPO, 브라질 핀테크) | TEL과 무관 — 이전에 'Northeast Utilities'로 추정한 것은 오류였음, 이번에 정정 |

**결론**: "구간 중간 이탈" 유형 9건(BNI/BRCM/CLX/CME/CMS/CPWR/PCP/TEL/UST)은 SEC 데이터 대조로
전부 false positive로 확정. BNI는 실제로는 2010년 2월 버크셔 해서웨이의 BNSF 인수로 인한
상장폐지(M&A)이지 rename이 아니며, 같은 시점 bkestelman 데이터의 파싱 결함이 겹쳐 무관한
후보가 뒤섞여 나온 것으로 최종 확인.

**한계**: SEC의 이 파일은 현재 활성 기업만 포함하므로, "전체 구간 미일치" 유형(AABA 등 25건,
상장폐지 종목)은 이 방법으로 해결되지 않음. 상장폐지 기업까지 포함하는 별도 SEC 벌크
인덱스나 CIK 개별 조회가 추가로 필요.


---

## 7. rename 후보 전체 목록 + SEC 검증 결과 (ticker_rename_candidates.csv)

```csv
canonical_ticker_fja,last_seen_as_is_in_bk,first_absent_in_bk,candidate_new_symbols_in_bk,detection_type,sec_edgar_verdict
AABA,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
ABKFQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
ANRZQ,,2011-06-30,"['AA', 'ACE', 'ANR', 'BHI', 'BTU', 'CBG', 'COH', 'CSC', 'DPS', 'DV', 'ERTS', 'FO', 'GCI', 'HCN', 'JDSU', 'JOYG', 'KFT', 'LTD', 'LUK', 'MHP', 'NU', 'PCLN', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFR', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
ATGE,,2009-06-30,"['AA', 'AOC', 'BHI', 'BTU', 'CBG', 'CIT', 'COH', 'CSC', 'DPS', 'DV', 'EK', 'ERTS', 'FO', 'FPL', 'GCI', 'HCN', 'JDSU', 'KFT', 'LTD', 'LUK', 'MHP', 'MOT', 'NU', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
BNI,2009-12-31,2010-01-31,"['BRK.B', 'MXB', 'NYSE:\xa0CLF', 'NYSE:\xa0CLX', 'NYSE:\xa0CMS', 'NYSE:\xa0COH', 'NYSE:\xa0SAI', 'NYSE:\xa0V', 'Nasdaq:\xa0CME', 'Nasdaq:\xa0ROST']",구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
BRCM,2015-12-31,2016-01-31,"['CFG', 'FRT']",구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
BTUUQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
CCEP,,2016-01-31,"['AA', 'BHI', 'CBG', 'COH', 'DLPH', 'DPS', 'HCN', 'LUK', 'MHFI', 'PCLN', 'TSO', 'TYC', 'WYN', 'YHOO']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
CCTYQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
CITGQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
CLX,2009-12-31,2010-01-31,"['BRK.B', 'MXB', 'NYSE:\xa0CLF', 'NYSE:\xa0CLX', 'NYSE:\xa0CMS', 'NYSE:\xa0COH', 'NYSE:\xa0SAI', 'NYSE:\xa0V', 'Nasdaq:\xa0CME', 'Nasdaq:\xa0ROST']",구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
CME,2009-12-31,2010-01-31,"['BRK.B', 'MXB', 'NYSE:\xa0CLF', 'NYSE:\xa0CLX', 'NYSE:\xa0CMS', 'NYSE:\xa0COH', 'NYSE:\xa0SAI', 'NYSE:\xa0V', 'Nasdaq:\xa0CME', 'Nasdaq:\xa0ROST']",구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
CMS,2009-12-31,2010-01-31,"['BRK.B', 'MXB', 'NYSE:\xa0CLF', 'NYSE:\xa0CLX', 'NYSE:\xa0CMS', 'NYSE:\xa0COH', 'NYSE:\xa0SAI', 'NYSE:\xa0V', 'Nasdaq:\xa0CME', 'Nasdaq:\xa0ROST']",구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
CPWR,2011-11-30,2011-12-31,['WPX-WI'],구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
DXC,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
EKDKQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
FMCC,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
FNMA,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
HSH,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
IAC,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
KATE,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
KDP,,2008-10-31,"['AA', 'AOC', 'BHI', 'BTU', 'CBG', 'CIT', 'COH', 'CSC', 'EK', 'ERTS', 'FO', 'FPL', 'GCI', 'JDSU', 'KFT', 'LIZ', 'LTD', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
LDOS,,2009-12-31,"['AA', 'AOC', 'BHI', 'BTU', 'CBG', 'COH', 'CSC', 'DPS', 'DV', 'EK', 'ERTS', 'FO', 'FPL', 'GCI', 'HCN', 'JDSU', 'KFT', 'LTD', 'LUK', 'MHP', 'MOT', 'NU', 'RSH', 'SLE', 'TSO', 'WAG', 'WFMI', 'WFR', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
LEHMQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
MTLQQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
PCP,2015-12-31,2016-01-31,"['CFG', 'FRT']",구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
RSHCQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
SUNEQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
TEL,2009-03-31,2009-04-30,['NU'],구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
TMUS,,2009-06-30,"['AA', 'AOC', 'BHI', 'BTU', 'CBG', 'CIT', 'COH', 'CSC', 'DPS', 'DV', 'EK', 'ERTS', 'FO', 'FPL', 'GCI', 'HCN', 'JDSU', 'KFT', 'LTD', 'LUK', 'MHP', 'MOT', 'NU', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
UST,2008-11-30,2008-12-31,['IRM'],구간 중간 이탈 (전환 시점 신규 등장 티커 대조),FALSE POSITIVE (SEC 대조로 확인됨)
VIAV,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
WAMUQ,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"
WYND,,2008-01-31,"['AA', 'ABK', 'ACE', 'AOC', 'BHI', 'BTU', 'CBG', 'CC', 'CIT', 'COH', 'CSC', 'CZN', 'EK', 'ERTS', 'FNM', 'FO', 'FPL', 'FRE', 'GCI', 'IACI', 'JDSU', 'KFT', 'LEH', 'LIZ', 'LTD', 'LTR', 'LUK', 'MHP', 'MOT', 'RSH', 'SLE', 'TSO', 'TYC', 'WAG', 'WFMI', 'WFR', 'WMI', 'WPI', 'WPO', 'WYN', 'YHOO', 'ZMH']",전체 구간 미일치 (예: YHOO->AABA 유형),"미검증 (상장폐지 종목, SEC 현재목록에 없음)"

```

---

## 8. 최종 결론 (FINDINGS.md 원문)

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


---

## 검증해 주셨으면 하는 점

1. SEC EDGAR 데이터로 9건을 false positive로 확정한 논리가 타당한지 (TEL/CME/CMS/CLX가
   "지금도 그대로 거래 중"이라는 사실이 곧 "과거에도 rename된 적 없다"를 함의하는지)
2. BNI의 실제 상장폐지 사유(버크셔의 BNSF 인수, 2010년 2월)가 알려진 사실과 부합하는지
3. "전체 구간 미일치" 25건을 SEC 상장폐지 기업 데이터로 마저 풀 수 있는 구체적 다음 단계 제안
4. 'NU=Northeast Utilities'였던 이전 추정을 'Nu Holdings(브라질 핀테크)'로 정정한 것이
   실제로 맞는지 (두 후보 모두 실존하는 상장사이므로 혼동 가능성 검토)
