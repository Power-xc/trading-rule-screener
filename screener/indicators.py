"""지표 함수 — 전부 '현재 봉까지의 정보만' 사용한다.

룩어헤드 규약:
  rolling 최대/최소는 반드시 shift(1)을 걸어 현재 봉을 제외한다.
  `c > rolling(20).max()` 는 현재 종가가 자기 자신을 포함한 최대값을 넘을 수
  없으므로 영원히 False가 되거나, 고가 기준이면 미래를 참조하게 된다.
  이 파일 밖에서 rolling을 직접 쓰지 말 것 — selftest.py가 이 규약을 검사한다.
"""
import numpy as np
import pandas as pd


def sma(a, n):
    return pd.Series(a).rolling(n).mean().to_numpy()


def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def shift(a, n=1):
    return pd.Series(a).shift(n).to_numpy()


def rolling_max(a, n):
    """현재 봉을 제외한 직전 n봉의 최대값."""
    return pd.Series(a).rolling(n).max().shift(1).to_numpy()


def rolling_min(a, n):
    """현재 봉을 제외한 직전 n봉의 최소값."""
    return pd.Series(a).rolling(n).min().shift(1).to_numpy()


def rsi(c, n=14):
    """Wilder RSI. 표준 구현과 소수점 6자리까지 일치함을 selftest에서 검증한다."""
    d = pd.Series(c).diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + au / ad)).to_numpy()


def stoch(h, l, c, k=5, d=3):
    hh = pd.Series(h).rolling(k).max()
    ll = pd.Series(l).rolling(k).min()
    fk = 100 * (pd.Series(c) - ll) / (hh - ll).replace(0, np.nan)
    return fk.rolling(d).mean().to_numpy()


def bollinger(c, n=20, k=2.0):
    m = pd.Series(c).rolling(n).mean()
    s = pd.Series(c).rolling(n).std()
    return (m - k * s).to_numpy(), m.to_numpy(), (m + k * s).to_numpy()


def atr(h, l, c, n=14):
    pc = pd.Series(c).shift(1)
    tr = pd.concat([pd.Series(h) - pd.Series(l),
                    (pd.Series(h) - pc).abs(),
                    (pd.Series(l) - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def true_trend(c, n=5, k=0):
    """패턴 '시작 전' 시점의 추세.

    k는 패턴 길이. 반전형 패턴을 검사할 때 추세를 패턴 마지막 봉에서 재면
    큰 양봉이 스스로 추세 조건을 깨서 탐지가 사라진다(실제로 겪은 결함).
    반드시 패턴이 시작되기 직전 시점으로 k만큼 밀어서 판정한다.
    """
    s = shift(c, k)
    return s < shift(c, k + n), s > shift(c, k + n)


def body_parts(o, h, l, c):
    """캔들 구성 요소. 꼬리 한도는 '몸통'이 아니라 '전체 범위' 기준으로 쓸 것.

    몸통 기준으로 잡으면 도지형(몸통이 거의 0)에서 한도가 0에 수렴해
    전형적인 망치형이 전부 탈락한다(실제로 겪은 결함).
    """
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    return body, rng, upper, lower
