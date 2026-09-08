"""
backtest.py
§6.4 Primary 지표(비용 차감 후 연율화 샤프비율) 및 §6.5 거래 전략/비용 모델.

핵심 원칙 (문서와 일치시켜야 하는 부분):
- 포트폴리오: 매일 '상승' 예측 종목 전체에 동일비중(equal-weight) 투자, 없으면 전액 현금 (§6.5.1)
- 거래비용: bp 단위 왕복 비용 * turnover (§6.5.2)
- Primary 지표: 연율화 샤프비율 SR = sqrt(252) * mean(r_net) / std(r_net) (§6.4)
- 턴오버는 항상 같이 보고 (전문 피드백 §5 반영 — 필수 보조지표)
"""
import numpy as np
import pandas as pd
import config as cfg


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


def annualized_sharpe(returns: pd.Series, periods_per_year: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
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
    block_length: int = cfg.BLOCK_BOOTSTRAP_LENGTH,
    n_boot: int = cfg.N_BOOTSTRAP,
    seed: int = cfg.RANDOM_SEED,
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
        return float(np.sqrt(cfg.TRADING_DAYS_PER_YEAR) * arr.mean() / sigma)

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
