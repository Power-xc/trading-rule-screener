"""자체 검사 — 스크리닝 전에 반드시 통과해야 하는 관문.

이 파일이 있는 이유: 규칙 정의가 틀리면 스크리닝 결과 전체가 무의미해진다.
실제로 초기 구현에서 아래 결함들이 나왔고, 각각을 검사로 고정해뒀다.

  · 꼬리 한도를 '몸통' 기준으로 잡아 도지형 망치가 전부 탈락    → test_wick_basis
  · 추세 맥락을 패턴 마지막 봉에서 재서 반전형이 자기 조건을 파괴 → test_trend_timing
  · rolling 최대/최소에 shift 누락 → 룩어헤드                  → test_no_lookahead
  · 규칙이 아예 발동하지 않는데 '엣지 없음'으로 오판            → test_rules_fire

테스트 시계열을 잘못 만들어 생기는 오탐도 겪었다(단조 시계열은 교차·통과 이벤트가
없어 0회가 정상이다). 그래서 합성 데이터는 등락이 있는 형태로 만든다.
"""
import numpy as np
import pandas as pd

from . import indicators as I
from .rules import RULES

RNG = np.random.default_rng(0)


def synth(close, vol=None, seed=None):
    """합성 OHLCV.

    꼬리 길이와 시가 갭을 무작위로 준다. 고정 폭으로 만들면 망치형(긴 꼬리)이나
    장악형(직전 종가와 다른 시가)이 구조적으로 발생할 수 없어, 규칙이 멀쩡한데도
    '한 번도 발동 안 함'으로 나온다(실제로 겪은 오탐).
    seed=None이면 결정적인 꼬리 패턴을 쓴다 — 특정 봉을 직접 덮어쓰는 테스트용.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    r = np.random.default_rng(seed if seed is not None else 12345)
    prev = np.concatenate([[c[0]], c[:-1]])
    o = prev * (1 + r.normal(0, 0.002, n))          # 시가 갭
    hi_body, lo_body = np.maximum(o, c), np.minimum(o, c)
    span = np.maximum(np.abs(c - o), c * 0.001)
    h = hi_body + span * np.abs(r.normal(0, 1.2, n))  # 위꼬리
    l = lo_body - span * np.abs(r.normal(0, 1.2, n))  # 아래꼬리
    return pd.DataFrame({
        "dt": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
        "ts": np.arange(n),
        "open": o, "high": h, "low": l, "close": c,
        "vol": np.full(n, 100.0) if vol is None else np.asarray(vol, dtype=float),
    })


def _gbm(n, vol=0.02, seed=0):
    r = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(r.normal(0, vol, n)))


def test_rsi_reference():
    """Wilder RSI가 표준 구현과 일치하는지."""
    c = _gbm(3000, seed=1)
    for n in (7, 14, 21):
        d = pd.Series(c).diff()
        au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        ad = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        ref = (100 - 100 / (1 + au / ad)).to_numpy()
        got = I.rsi(c, n)
        m = ~np.isnan(ref) & ~np.isnan(got)
        err = np.abs(ref[m] - got[m]).max()
        assert err < 1e-9, f"RSI{n} 오차 {err}"
        assert got[m].min() >= 0 and got[m].max() <= 100, f"RSI{n} 범위 이탈"
    return "RSI(7/14/21) 표준 구현과 일치, 범위 [0,100]"


def test_rolling_shift():
    """rolling 최대/최소가 현재 봉을 제외하는지."""
    # a[-1]=10을 제외한 직전 3봉은 a[2:5]=[3,4,5] → max 5, min 3
    a = np.array([1, 2, 3, 4, 5, 10.0])
    assert I.rolling_max(a, 3)[-1] == 5, f"rolling_max={I.rolling_max(a,3)[-1]}, 5여야"
    assert I.rolling_min(a, 3)[-1] == 3, f"rolling_min={I.rolling_min(a,3)[-1]}, 3이여야"
    # 신고가 봉에서 자기 자신이 제외되는지 (룩어헤드 핵심)
    assert a[-1] > I.rolling_max(a, 3)[-1], "신고가가 자기 자신에 막혀 돌파 판정 불가"
    return "rolling_max/min이 현재 봉 제외(shift 1) 확인"


def test_wick_basis():
    """몸통이 거의 없는 망치형이 탐지되는지 (꼬리 한도가 범위 기준이어야 함)."""
    pre = list(np.linspace(100, 88, 12))
    df = synth(pre + [88.0] + list(np.linspace(88, 95, 5)))
    df.loc[12, ["open", "high", "low", "close"]] = [86.0, 86.4, 80.0, 86.2]
    fn = [r for r in RULES if r["name"] == "망치형"][0]["fn"]
    hit = np.nan_to_num(np.asarray(fn(df), dtype=float))[12]
    assert hit == 1, "도지형 몸통의 망치형이 탐지되지 않음 (꼬리 한도가 몸통 기준일 가능성)"
    return "도지형 몸통 망치형 탐지 확인"


def test_trend_timing():
    """큰 양봉으로 끝나는 반전형이 스스로 추세 조건을 깨지 않는지."""
    pre = list(np.linspace(120, 96, 14))
    rows = [(95, 95.2, 88, 88.5), (88.3, 88.8, 87.9, 88.4), (88.5, 93, 88.3, 92.5)]
    df = synth(pre + [r[3] for r in rows] + [92.0, 92.5, 93.0])
    for k, r in enumerate(rows):
        df.loc[14 + k, ["open", "high", "low", "close"]] = list(r)
    fn = [r for r in RULES if r["name"] == "샛별형"][0]["fn"]
    hit = np.nan_to_num(np.asarray(fn(df), dtype=float))[16]
    assert hit == 1, "샛별형이 탐지되지 않음 (추세 맥락을 패턴 마지막 봉에서 쟀을 가능성)"
    return "반전형의 추세 맥락이 패턴 시작 전 시점에서 판정됨"


def test_no_lookahead():
    """미래 캔들을 바꿔도 과거 신호가 변하지 않는지 — 전 규칙 대상."""
    x = _gbm(2000, seed=2)
    base = synth(x)
    mod = synth(np.concatenate([x[:1200], x[1200:] * 1.5]))
    bad = []
    for r in RULES:
        a = np.nan_to_num(np.asarray(r["fn"](base), dtype=float))
        b = np.nan_to_num(np.asarray(r["fn"](mod), dtype=float))
        if not np.array_equal(a[:1195], b[:1195]):
            bad.append(r["name"])
    assert not bad, f"룩어헤드 의심 규칙: {bad}"
    return f"전 규칙 {len(RULES)}개 룩어헤드 없음"


def test_rules_fire():
    """모든 규칙이 현실적인 시계열에서 최소 1회는 발동하는지.

    발동하지 않는 규칙은 '엣지 없음'이 아니라 '정의가 틀림'이다. 둘을 구분해야 한다.
    """
    x = _gbm(6000, vol=0.02, seed=3)
    v = np.abs(RNG.lognormal(4.6, 0.6, 6000))
    df = synth(x, v)
    dead = []
    for r in RULES:
        n = np.nansum(np.asarray(r["fn"](df), dtype=float))
        if n < 1:
            dead.append(r["name"])
    assert not dead, f"합성 데이터에서 한 번도 발동하지 않는 규칙: {dead}"
    return f"전 규칙 {len(RULES)}개 발동 확인"


def test_symmetry():
    """롱/숏 대칭 규칙이 랜덤워크에서 비슷한 빈도로 발동하는지."""
    df = synth(_gbm(20000, vol=0.01, seed=4))
    pairs = [("볼린저 하단 이탈 매수", "볼린저 상단 돌파 매도"),
             ("20일 고점 돌파 롱", "20일 저점 이탈 숏")]
    msgs = []
    for a, b in pairs:
        fa = [r for r in RULES if r["name"] == a][0]["fn"]
        fb = [r for r in RULES if r["name"] == b][0]["fn"]
        na = np.nansum(np.asarray(fa(df), dtype=float))
        nb = np.nansum(np.asarray(fb(df), dtype=float))
        ratio = na / max(nb, 1)
        assert 0.4 < ratio < 2.5, f"{a}/{b} 발동 비율 {ratio:.2f} — 비대칭 의심"
        msgs.append(f"{a.split()[0]} {na:.0f}/{nb:.0f}")
    return "롱숏 대칭 확인 (" + ", ".join(msgs) + ")"


TESTS = [test_rsi_reference, test_rolling_shift, test_wick_basis,
         test_trend_timing, test_no_lookahead, test_rules_fire, test_symmetry]


def run(verbose=True):
    ok = True
    for t in TESTS:
        try:
            msg = t()
            if verbose:
                print(f"  PASS  {t.__name__:<22} {msg}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {t.__name__:<22} {e}")
        except Exception as e:
            ok = False
            print(f"  ERROR {t.__name__:<22} {type(e).__name__}: {e}")
    return ok


if __name__ == "__main__":
    import sys
    print(f"자체 검사 — 규칙 {len(RULES)}개")
    sys.exit(0 if run() else 1)
