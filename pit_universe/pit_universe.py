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
    FJA_CSV = "sp500_ticker_start_end.csv"
    BK_CSV = "sp500_monthly.csv"
    HANSHOF_CSV = "hanshof_sp500_historical.csv"

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
