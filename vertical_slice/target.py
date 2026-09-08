"""
target.py
§5.4 예측 target 정의: r_{t+1} = Close_{t+1}/Open_{t+1} - 1
                        y_{t+1} = sign(r_{t+1}) * 1{|r_{t+1}| > 0.5 * sigma_j}
§6.1.1 M0 가격 입력 변수: OHLC/거래량, RSI(14), MACD, 볼린저밴드(20,2), 시장/섹터 요인

[행(row) 날짜 정합성 — 코드 리뷰에서 제기된 t/t+1 shift 논쟁에 대한 명시적 답]
§5.4의 cutoff·진입·청산은 전부 '같은 거래일 t+1' 안에서 일어난다:
  t (정보 앵커) = 전일 종가 + 오버나이트 뉴스가 관측되는 시점
  t+1 (행동일)  = cutoff(09:25)·진입(09:30 시가)·청산(종가)이 모두 일어나는 당일
즉 패널의 각 행 날짜 D는 표기법상 "t+1"에 해당한다. 따라서:
  - build_price_features()가 가격 파생 feature를 D-1(="t")로 1일 shift하는 것은 맞다 (룩어헤드 방지).
  - compute_target_return()은 그 SAME 행 D의 시가~종가(Open_{t+1}, Close_{t+1})를 그대로 써야 하며,
    여기에 추가로 shift(-1)을 걸면 "D-1 정보로 D+1 수익률을 예측"하는 꼴이 되어 하루를 건너뛰는
    새로운 정합성 오류가 생긴다. 코드 리뷰에서 제안된 shift(-1)은 적용하지 않는다.
"""
import numpy as np
import pandas as pd
import config as cfg


def compute_target_return(df: pd.DataFrame) -> pd.Series:
    """
    r_{t+1} = Close_{t+1} / Open_{t+1} - 1
    df 행의 날짜 D 자체가 't+1'(행동일)이므로 SAME 행의 open/close를 그대로 사용한다 (shift 없음).
    """
    return df["close"] / df["open"] - 1.0


def compute_classification_label(r_next: pd.Series, sigma_j: pd.Series,
                                  k: float = cfg.CLASSIFY_THRESHOLD_SIGMA) -> pd.Series:
    """
    y_{t+1} = sign(r_{t+1}) * 1{|r_{t+1}| > k * sigma_j}
    sigma_j는 "훈련 구간 내부에서만" 산정된 값이어야 한다 (호출하는 쪽에서 룩어헤드 없이 전달).
    반환값: {-1, 0, 1}
    """
    threshold = k * sigma_j
    label = np.sign(r_next)
    label = np.where(r_next.abs() > threshold, label, 0)
    return pd.Series(label.astype(int), index=r_next.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """%B = (Close - LowerBand) / (UpperBand - LowerBand)"""
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    band_width = (upper - lower).replace(0, np.nan)
    return (close - lower) / band_width


def build_price_features(panel: pd.DataFrame, mkt_returns: pd.Series) -> pd.DataFrame:
    """
    panel: index=date, columns=['ticker','open','high','low','close','volume','sector']을 가진 long-format DataFrame.
    mkt_returns: date -> 시장 전체(동일가중) 일간 수익률 (§6.1.1의 '시장 요인').
    반환: M0~M5 전 모델이 공통으로 사용하는 가격/시장/섹터 feature.

    중요: target(§5.4)이 '해당 날짜의 시가~종가 수익률'이므로, 그 날짜의 가격 feature는
    "전일 종가까지"의 정보만 사용해야 한다 (그렇지 않으면 그날 종가가 feature와 target에 동시에
    섞여 들어가는 룩어헤드가 발생한다). 아래에서 모든 파생 feature를 종목별로 1일 shift한다.
    """
    out = []
    panel = panel.sort_values(["ticker", "date"])
    for ticker, g in panel.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ret_1d"] = g["close"].pct_change(1)
        g["ret_5d"] = g["close"].pct_change(5)
        g["rsi_14"] = rsi(g["close"], 14)
        g["macd"] = macd(g["close"])
        g["bb_pct"] = bollinger_pct_b(g["close"])
        g["mkt_ret_1d"] = g["date"].map(mkt_returns)

        # 룩어헤드 방지: 파생 feature 전체를 하루 shift (D일 feature는 D-1일 종가까지의 정보만 사용)
        feature_cols = ["ret_1d", "ret_5d", "rsi_14", "macd", "bb_pct", "mkt_ret_1d"]
        g[feature_cols] = g[feature_cols].shift(1)

        out.append(g)
    result = pd.concat(out, ignore_index=True)

    # §6.1.1 섹터 요인: 섹터 내 동일가중 평균 수익률도 동일하게 1일 shift 필요
    # (ret_1d가 이미 shift된 상태이므로 이를 기반으로 섹터 평균을 내면 자동으로 D-1 시점 값이 됨)
    sector_ret = result.groupby(["date", "sector"])["ret_1d"].transform("mean")
    result["sector_ret_1d"] = sector_ret

    return result
