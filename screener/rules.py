"""규칙 레지스트리 — 각 규칙은 출처(책·인용문)를 달고 등록된다.

출처를 필수로 만든 이유: 스크리닝을 돌리다 보면 "이것도 해볼까" 하고 파라미터를
조금씩 바꾼 변형이 늘어난다. 그게 곧 전구간 그리드서치이고 REPORT.md가 지목한
v1의 실패 원인이다. 출처를 적을 수 없는 규칙은 등록하지 않는 것을 원칙으로 한다.

새 규칙 추가 시:
  1. @rule 데코레이터로 등록 (출처·인용 필수)
  2. selftest.py의 SELFTEST에 '이 규칙이 켜져야 하는 합성 상황'을 추가
  3. run_screen.py 실행 — 자체검사를 통과해야 스크리닝으로 넘어간다
"""
import numpy as np

from .indicators import (sma, shift, rolling_max, rolling_min, rsi, stoch,
                         bollinger, true_trend, body_parts)

RULES = []


def rule(name, book, quote, side, family):
    """side: +1 롱, -1 숏.  family: 결과를 계열별로 묶어 보기 위한 태그."""
    def deco(fn):
        RULES.append({"name": name, "book": book, "quote": quote,
                      "side": side, "family": family, "fn": fn})
        return fn
    return deco


def _ohlcv(df):
    return (df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
            df["close"].to_numpy(), df["vol"].to_numpy())


# ──────────────────────────────────────────────────────────────────────
# 이동평균 계열
# ──────────────────────────────────────────────────────────────────────
@rule("MA 5/20 골든크로스", "거래량으로 투자하라",
      "5일이동평균과 20일이동평균의 교차를 이용해 매수신호를 생성", +1, "추세")
def _(df):
    *_, c, _v = _ohlcv(df)
    a, b = sma(c, 5), sma(c, 20)
    return (a > b) & (shift(a) <= shift(b))


@rule("MA 5/20 데드크로스", "거래량으로 투자하라",
      "단기 이동평균이 장기를 하향교차하면 하락세, 즉 매도 신호", -1, "추세")
def _(df):
    *_, c, _v = _ohlcv(df)
    a, b = sma(c, 5), sma(c, 20)
    return (a < b) & (shift(a) >= shift(b))


@rule("20일선 상향돌파", "차트박사의 승률 80% 新매매기법",
      "20일 이평선을 기준점으로 돌파 시 매수 후 20일 이평선 부근을 마지노선으로", +1, "추세")
def _(df):
    *_, c, _v = _ohlcv(df)
    m = sma(c, 20)
    return (c > m) & (shift(c) <= shift(m))


@rule("120일선 지지 매수", "세력주 매매 공식",
      "주식 시장이 밀리거나 120일선 밑에 있거나 지지선을 근거로 매수", +1, "평균회귀")
def _(df):
    _o, _h, l, c, _v = _ohlcv(df)
    m = sma(c, 120)
    return (l <= m * 1.01) & (c > m) & (shift(c) > shift(m))


@rule("5/20 이격도 7% 이내 역배열 후 5일선 돌파", "차트박사의 승률 80% 新매매기법",
      "60이평, 20이평, 5이평이 역배열을 이루고 20이평과 5이평의 이격도가 7% 이내인 구간에서 5일선 위로",
      +1, "추세")
def _(df):
    *_, c, _v = _ohlcv(df)
    m5, m20, m60 = sma(c, 5), sma(c, 20), sma(c, 60)
    rev = (m5 < m20) & (m20 < m60)
    gap = np.abs(m5 - m20) / m20 <= 0.07
    return ((shift(rev.astype(float)) == 1) & (shift(gap.astype(float)) == 1)
            & (c > m5) & (shift(c) <= shift(m5)))


# ──────────────────────────────────────────────────────────────────────
# 돌파 계열
# ──────────────────────────────────────────────────────────────────────
for _N in (10, 20, 55):
    @rule(f"{_N}일 고점 돌파 롱", "기술적 분석 모르고 절대 주식투자 하지 마라(슈웨거)",
          "이전 고점 돌파를 매수 신호로 간주", +1, "추세")
    def _(df, N=_N):
        _o, h, _l, c, _v = _ohlcv(df)
        return c > rolling_max(h, N)

    @rule(f"{_N}일 저점 이탈 숏", "기술적 분석 모르고 절대 주식투자 하지 마라(슈웨거)",
          "이전 저점 돌파를 매도 신호로 간주", -1, "추세")
    def _(df, N=_N):
        _o, _h, l, c, _v = _ohlcv(df)
        return c < rolling_min(l, N)


@rule("고점 돌파 3회 종가 확정", "기술적 분석 모르고 절대 주식투자 하지 마라(슈웨거)",
      "고점을 넘어서는 종가가 세 차례 나타나는 것을 돌파 확정 조건으로", +1, "추세")
def _(df):
    import pandas as pd
    _o, h, _l, c, _v = _ohlcv(df)
    b = pd.Series((c > rolling_max(h, 20)).astype(float))
    s = b.rolling(3).sum().to_numpy()
    return (s == 3) & (shift(s) < 3)


@rule("10일 저점 하향돌파 되돌림 숏", "기술적 분석 모르고 절대 주식투자 하지 마라(슈웨거)",
      "종가가 이전 10일간의 저점을 하향 돌파할 때를 가격 되돌림으로 간주", -1, "추세")
def _(df):
    _o, _h, l, c, _v = _ohlcv(df)
    r = rolling_min(l, 10)
    return (c < r) & (shift(c) >= shift(r))


# ──────────────────────────────────────────────────────────────────────
# 거래량 계열
# ──────────────────────────────────────────────────────────────────────
@rule("거래량 140% + 고점돌파", "거래량으로 투자하라",
      "이탈 시 거래량이 이전 거래범위의 적어도 140% 되는 것을 선호", +1, "추세")
def _(df):
    _o, h, _l, c, v = _ohlcv(df)
    return (c > rolling_max(h, 20)) & (v >= sma(v, 20) * 1.40)


@rule("거래량 140% + 저점이탈 숏", "거래량으로 투자하라",
      "동일 거래량 기준을 하락 이탈에 대칭 적용", -1, "추세")
def _(df):
    _o, _h, l, c, v = _ohlcv(df)
    return (c < rolling_min(l, 20)) & (v >= sma(v, 20) * 1.40)


@rule("거래량 급증 장대양봉", "거래량으로 투자하라 / 세력주 매매 공식",
      "롱바디 양봉이 거래량을 동반할 때", +1, "추세")
def _(df):
    o, _h, _l, c, v = _ohlcv(df)
    return (c > o) & ((c - o) / o >= 0.03) & (v >= sma(v, 20) * 2.0)


# ──────────────────────────────────────────────────────────────────────
# 밴드 / 오실레이터 계열
# ──────────────────────────────────────────────────────────────────────
@rule("볼린저 하단 이탈 매수", "차트박사의 승률 80% 新매매기법",
      "밴드의 상한선에서는 매도하고 하한선에서는 매수로 대응해야", +1, "평균회귀")
def _(df):
    *_, c, _v = _ohlcv(df)
    lo, _m, _up = bollinger(c, 20, 2.0)
    return (c < lo) & (shift(c) >= shift(lo))


@rule("볼린저 상단 돌파 매도", "차트박사의 승률 80% 新매매기법",
      "밴드 상한선에서는 매도", -1, "평균회귀")
def _(df):
    *_, c, _v = _ohlcv(df)
    _lo, _m, up = bollinger(c, 20, 2.0)
    return (c > up) & (shift(c) <= shift(up))


@rule("Envelope(20MA-10%) 하단 지지 매수", "차트박사의 승률 80% 新매매기법",
      "Envelope 지지선에 닿아 매수시점을 준 후 상승하여 7~10% 이상의 수익", +1, "평균회귀")
def _(df):
    _o, _h, l, c, _v = _ohlcv(df)
    m = sma(c, 20)
    return (l <= m * 0.90) & (c > m * 0.90)


@rule("RSI(14) 30 하향 진입 매수", "기술적 분석 모르고 절대 주식투자 하지 마라(슈웨거)",
      "매수 신호는 RSI가 30 밑으로 떨어진 지점", +1, "평균회귀")
def _(df):
    *_, c, _v = _ohlcv(df)
    r = rsi(c, 14)
    return (r < 30) & (shift(r) >= 30)


@rule("RSI(14) 70 상향 진입 매도", "기술적 분석 모르고 절대 주식투자 하지 마라(슈웨거)",
      "과매수선 대칭 적용", -1, "평균회귀")
def _(df):
    *_, c, _v = _ohlcv(df)
    r = rsi(c, 14)
    return (r > 70) & (shift(r) <= 70)


@rule("스토캐스틱 5-3 과매도 반등", "실전 스윙 트레이딩 기법(앨런 팔리)",
      "지수가 저점을 찍을 때 5-3-3 스토캐스틱이 상승을 시작", +1, "평균회귀")
def _(df):
    _o, h, l, c, _v = _ohlcv(df)
    k = stoch(h, l, c, 5, 3)
    return (shift(k) < 20) & (k > shift(k)) & (k >= 20)


@rule("스토캐스틱 5-3 과매수 반락 숏", "실전 스윙 트레이딩 기법(앨런 팔리)",
      "대칭 적용", -1, "평균회귀")
def _(df):
    _o, h, l, c, _v = _ohlcv(df)
    k = stoch(h, l, c, 5, 3)
    return (shift(k) > 80) & (k < shift(k)) & (k <= 80)


for _F, _label in ((0.618, "0.618"), (0.786, "0.786")):
    @rule(f"피보 {_label} 되돌림 지지 매수", "Guide to Precision Harmonic Pattern Trading",
          f"{_label} retracement level", +1, "평균회귀")
    def _(df, F=_F):
        _o, h, l, c, _v = _ohlcv(df)
        hi, lo = rolling_max(h, 20), rolling_min(l, 20)
        tgt = hi - (hi - lo) * F
        return (l <= tgt) & (c > tgt) & (hi > lo)


# ──────────────────────────────────────────────────────────────────────
# 캔들 패턴 계열
#   꼬리 한도는 전체 범위 기준, 추세 맥락은 패턴 시작 전 시점에서 판정한다.
#   (둘 다 초기 구현에서 틀렸던 지점 — indicators.py 주석 참고)
# ──────────────────────────────────────────────────────────────────────
def _candle(df):
    o, h, l, c, _v = _ohlcv(df)
    body, rng, upper, lower = body_parts(o, h, l, c)
    return o, h, l, c, body, rng, upper, lower


@rule("망치형", "실전 캔들 매매법", "일단의 급락세 후 양봉 망치형에서 매수 대응이 가능", +1, "평균회귀")
def _(df):
    o, h, l, c, body, rng, upper, lower = _candle(df)
    dn, _up = true_trend(c, 5, 1)
    return (body <= rng * 0.35) & (lower >= rng * 0.55) & (upper <= rng * 0.12) & dn


@rule("유성형 숏", "실전 캔들 매매법", "위꼬리가 긴 유성형은 매도세의 반발을 의미", -1, "평균회귀")
def _(df):
    o, h, l, c, body, rng, upper, lower = _candle(df)
    _dn, up = true_trend(c, 5, 1)
    return (body <= rng * 0.35) & (upper >= rng * 0.55) & (lower <= rng * 0.12) & up


@rule("상승 장악형", "실전 캔들 매매법", "직전 캔들을 감싸는 양봉", +1, "평균회귀")
def _(df):
    o, h, l, c, *_ = _candle(df)
    dn, _up = true_trend(c, 5, 2)
    return (c > o) & (shift(c) < shift(o)) & (c > shift(o)) & (o < shift(c)) & dn


@rule("하락 장악형 숏", "실전 캔들 매매법",
      "하락 장악형, 즉 양 자리 음(양봉이 나와야 할 자리에 음봉이 나오는 것)", -1, "평균회귀")
def _(df):
    o, h, l, c, *_ = _candle(df)
    _dn, up = true_trend(c, 5, 2)
    return (c < o) & (shift(c) > shift(o)) & (c < shift(o)) & (o > shift(c)) & up


@rule("샛별형", "실전 캔들 매매법", "일단의 급락세 후 유효한 샛별형이 완성되었으므로 매수 대응이 가능", +1, "평균회귀")
def _(df):
    o, h, l, c, body, rng, *_ = _candle(df)
    dn, _up = true_trend(c, 5, 3)
    return ((shift(c, 2) < shift(o, 2))
            & (np.abs(shift(c, 1) - shift(o, 1)) <= shift(rng, 1) * 0.4)
            & (c > o) & (c > (shift(o, 2) + shift(c, 2)) / 2) & dn)


@rule("저녁별형 숏", "실전 캔들 매매법", "저녁별형 후 강한 하락 움직임", -1, "평균회귀")
def _(df):
    o, h, l, c, body, rng, *_ = _candle(df)
    _dn, up = true_trend(c, 5, 3)
    return ((shift(c, 2) > shift(o, 2))
            & (np.abs(shift(c, 1) - shift(o, 1)) <= shift(rng, 1) * 0.4)
            & (c < o) & (c < (shift(o, 2) + shift(c, 2)) / 2) & up)


@rule("적삼병", "차트의 해석 - 김정환", "삼병 — 세 개의 연속 양봉", +1, "추세")
def _(df):
    o, *_rest = _ohlcv(df)
    c = df["close"].to_numpy()
    up = c > o
    return up & (shift(up.astype(float)) == 1) & (shift(up.astype(float), 2) == 1)


@rule("흑삼병 숏", "차트의 해석 - 김정환", "흑삼병 — 세 개의 연속 음봉", -1, "추세")
def _(df):
    o, *_rest = _ohlcv(df)
    c = df["close"].to_numpy()
    dn = c < o
    return dn & (shift(dn.astype(float)) == 1) & (shift(dn.astype(float), 2) == 1)
