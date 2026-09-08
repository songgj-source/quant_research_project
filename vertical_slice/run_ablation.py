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
import numpy as np
import pandas as pd

import config as cfg
import data_synthetic as ds
import sentiment as sen
import normalize as norm
import target as tgt
import models as mdl
import backtest as bt


def build_full_panel(tickers, n_days=220):
    print(f"[1/6] 합성 가격 데이터 생성 중... (종목 {len(tickers)}개 x {n_days}거래일)")
    price_panel = ds.generate_price_panel(tickers, n_days=n_days)
    mkt_returns = ds.compute_market_returns(price_panel)
    price_feat = tgt.build_price_features(price_panel, mkt_returns)

    print("[2/6] 합성 뉴스 데이터 생성 중 (텍스트 감성·확산성 독립 신호)...")
    articles = ds.generate_news_articles(tickers, n_days=n_days)
    articles["finbert_polarity"] = sen.scalarize_finbert(articles["finbert_p_pos"], articles["finbert_p_neg"])

    print("[3/6] §5.6 관련성·시간감쇠 가중 집계 수행 중 (룩어헤드 단위테스트 포함)...")
    score_cols = ["vader_compound", "finbert_polarity", "llm_polarity",
                  "llm_intensity", "llm_uncertainty", "diffusion_raw"]
    agg_rows = []
    for (ticker, day), grp in articles.groupby(["ticker", "target_date"]):
        cutoff_ts = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=25)
        row = {"ticker": ticker, "date": day, "has_news": 1}
        for col in score_cols:
            row[col] = sen.aggregate_daily_sentiment(grp, col, cutoff_ts)
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
        panel[col] = norm.normalize_by_stock(panel, value_col=col, window=cfg.ROLLING_NORMALIZE_WINDOW)
    # 코드 리뷰 피드백 C: 정규화 초기 구간(min_periods 미달)의 NaN을 0으로 덮지 않고
    # §5.7의 '해당 구간을 분석에서 제외' 원칙에 따라 아래 dropna에서 제거한다.

    print("[6/6] §5.4 target(r_{t+1}) 및 분류 레이블(y_{t+1}) 계산 중...")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["r_next"] = tgt.compute_target_return(panel)
    # sigma_j: 종목별 과거 60거래일 롤링 표준편차 (미래 정보 미사용)
    panel["sigma_j"] = panel.groupby("ticker")["r_next"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).std()
    )
    panel = panel.dropna(
        subset=cfg.PRICE_FEATURE_COLUMNS + sentiment_value_cols + ["r_next", "sigma_j"]
    ).reset_index(drop=True)
    panel["y"] = tgt.compute_classification_label(panel["r_next"], panel["sigma_j"])

    return panel


def run_ablation(panel: pd.DataFrame, model_names=None, n_splits=3):
    if model_names is None:
        model_names = list(cfg.MODEL_FEATURE_SETS.keys())  # M0_news 포함 (피드백 ①)

    all_dates = panel["date"].unique()
    splits = mdl.walk_forward_splits(all_dates, n_splits=n_splits, min_train_days=100, test_days=25)
    print(f"\nwalk-forward 분할: {len(splits)}개 fold 생성됨 (embargo 적용됨)")

    results = {}
    fold_returns = {m: [] for m in model_names}  # 모델별 fold-wise net return 시계열 리스트

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)]
        if len(test) == 0 or len(train) == 0:
            continue

        for m in model_names:
            model = mdl.AblationModel(m).fit(train, train["y"])
            pred_label = model.predict_label(test)

            f1 = mdl.macro_f1(test["y"], pred_label)

            # §6.5.1 신호: 예측 레이블이 '상승(1)'인 종목을 매수 (분류기 출력 그대로 사용)
            sig_df = test.assign(signal=(pred_label == 1).astype(int))
            sig_wide = sig_df.pivot(index="date", columns="ticker", values="signal").fillna(0)
            ret_wide = sig_df.pivot(index="date", columns="ticker", values="r_next")

            port = bt.build_daily_portfolio_returns(sig_wide, ret_wide)
            port_net = bt.apply_cost(port, cost_bps=cfg.WORST_CASE_SLIPPAGE_BPS)
            summary = bt.summarize_backtest(port_net)
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
    primary_test = bt.paired_block_bootstrap_diff(fold_diffs("M5", "M3"))
    # 참고용 보조 검정: M0(가격만) 대비 M5의 차이도 함께 보고 (얼마나 개선됐는지의 참고치)
    reference_m0_vs_m5 = bt.paired_block_bootstrap_diff(fold_diffs("M5", "M0"))
    # 피드백 ① 검증용: '뉴스 내용'의 순수 기여 = M3(LLM 극성) vs M0_news(뉴스 유무만) 비교
    content_vs_presence_test = bt.paired_block_bootstrap_diff(fold_diffs("M3", "M0_news"))

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
    print("통계 검정 (모두 fold 경계를 넘지 않는 paired block bootstrap, L=%d)" % cfg.BLOCK_BOOTSTRAP_LENGTH)
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
    print(f"\nMESI 체크 (ΔSR = SR_M5 - SR_M3 >= {cfg.MESI_MIN_SHARPE_IMPROVEMENT}): "
          f"ΔSR = {delta_sr:.4f} -> {'통과' if delta_sr >= cfg.MESI_MIN_SHARPE_IMPROVEMENT else '미달'}")
