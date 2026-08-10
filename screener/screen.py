"""규칙 스크리닝 관문.

한 규칙이 '거래 후보'가 되려면 아래를 전부 통과해야 한다. 하나라도 걸리면 탈락이다.

  G1 구간 일관성 : 데이터를 3등분한 각 구간에서 모두 비용선을 넘을 것
  G2 중첩 보정   : 신호 간 최소 H봉 간격을 강제한 비중첩 표본에서도 CI 하한이 비용선 위
  G3 시계열 보정 : 블록 부트스트랩(시간 블록 재표집) CI 하한도 비용선 위
  G4 표본        : 비중첩 표본이 최소 60건

G2가 핵심이다. 돌파 신호는 강한 추세에서 연속 발생하므로 forward window가 겹치고,
겹친 관측을 독립으로 세면 유효 표본수가 부풀려져 CI가 실제보다 좁아진다. 실측에서
55일 고점돌파의 신호 64%가 중첩이었고, 보정하니 초과수익이 +94bp에서 +51bp로
떨어지면서 CI가 0을 포함하게 됐다. 보정 없이는 통과로 오판한다.

측정량은 '초과수익'이다 — 신호 후 H봉 수익률에서 같은 구간 전체 봉의 평균
H봉 수익률을 뺀 값(bp). BTC의 장기 상승 드리프트를 제거해야 신호 자체의
기여만 남는다. 롱 편향 신호는 이 보정 없이는 전부 좋아 보인다.
"""
import json
import os

import numpy as np

from . import data as D

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_screen_state.json")
MIN_SEG_SIGNALS = 30
MIN_NONOVERLAP = 60
BOOT = 2000


def _segments(years):
    """데이터 연도 범위를 3등분한다. 하드코딩하지 않아 데이터가 늘어도 유효하다."""
    lo, hi = int(years.min()), int(years.max())
    span = hi - lo + 1
    a = lo + span // 3
    b = lo + 2 * span // 3
    return [(f"S1 {lo}-{a-1}", lo, a - 1),
            (f"S2 {a}-{b-1}", a, b - 1),
            (f"S3 {b}-{hi}", b, hi)]


def _excess(c, idx, H, base):
    if len(idx) == 0:
        return np.nan
    return ((c[idx + H] / c[idx] - 1) * 10000).mean() - base


def _nonoverlapping(idx, H):
    """직전 채택 신호로부터 H봉 이상 떨어진 신호만 남긴다."""
    keep, last = [], -(10 ** 9)
    for i in idx:
        if i - last >= H:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def _block_bootstrap(c, sig, H, base, rng, block):
    """시간 블록 단위로 재표집해 시계열 의존성을 반영한 CI."""
    n = len(c)
    nb = max(2, n // block)
    blocks = [np.arange(k * block, min((k + 1) * block, n)) for k in range(nb)]
    vals = []
    for _ in range(BOOT):
        pick = np.concatenate([blocks[k] for k in rng.integers(0, nb, nb)])
        ii = pick[sig[pick]]
        ii = ii[ii < n - H - 1]
        if len(ii) >= MIN_SEG_SIGNALS:
            vals.append(_excess(c, ii, H, base))
    if len(vals) < 100:
        return np.nan, np.nan
    return tuple(np.percentile(np.array(vals), [5, 95]))


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"cells_tested": 0, "runs": []}


def save_state(state):
    json.dump(state, open(STATE_PATH, "w"), ensure_ascii=False, indent=1)


def screen_rule(df, rule, H, market, rng=None):
    """규칙 하나를 평가해 판정 딕셔너리를 반환한다."""
    rng = rng or np.random.default_rng(11)
    cost = D.COSTS[market]
    c = df["close"].to_numpy()
    years = df["dt"].dt.year.to_numpy()
    valid = np.arange(len(c)) < len(c) - H - 1

    try:
        raw = rule["fn"](df)
    except Exception as e:
        return {"name": rule["name"], "error": f"{type(e).__name__}: {e}"[:80]}
    sig = np.nan_to_num(np.asarray(raw, dtype=float), nan=0).astype(bool) & valid

    bi = np.where(valid)[0]
    base = ((c[bi + H] / c[bi] - 1) * 10000).mean()
    side = rule["side"]
    idx = np.where(sig)[0]

    res = {"name": rule["name"], "book": rule["book"], "side": side,
           "family": rule.get("family", "-"), "n": len(idx), "cost": cost}

    # G1 구간 일관성
    seg_vals, seg_n = [], []
    for _, a, b in _segments(years):
        m = (years >= a) & (years <= b) & valid
        mb = np.where(m)[0]
        sbase = ((c[mb + H] / c[mb] - 1) * 10000).mean()
        si = np.where(m & sig)[0]
        seg_n.append(len(si))
        seg_vals.append(_excess(c, si, H, sbase) * side
                        if len(si) >= MIN_SEG_SIGNALS else None)
    res["seg"], res["seg_n"] = seg_vals, seg_n
    graded = [v for v in seg_vals if v is not None]
    res["g1_pass"] = len(graded) == 3 and all(v > cost for v in graded)

    if len(idx) == 0:
        res.update(total=None, verdict="표본없음")
        return res
    res["total"] = _excess(c, idx, H, base) * side

    # G2 중첩 보정
    keep = _nonoverlapping(idx, H)
    res["n_nonoverlap"] = len(keep)
    res["overlap_ratio"] = float((np.diff(idx) < H).mean()) if len(idx) > 1 else 0.0
    if len(keep) >= MIN_NONOVERLAP:
        res["nonoverlap"] = _excess(c, keep, H, base) * side
        bs = np.array([_excess(c, rng.choice(keep, len(keep)), H, base) * side
                       for _ in range(BOOT)])
        res["g2_lo"], res["g2_hi"] = np.percentile(bs, [5, 95])
        res["g2_pass"] = bool(res["g2_lo"] > cost)
    else:
        res["nonoverlap"] = None
        res["g2_lo"] = res["g2_hi"] = np.nan
        res["g2_pass"] = False

    # G3 블록 부트스트랩
    lo, hi = _block_bootstrap(c, sig, H, base, rng, block=max(H * 10, 50))
    res["g3_lo"], res["g3_hi"] = lo * side if side > 0 else hi * side, \
                                 hi * side if side > 0 else lo * side
    res["g3_pass"] = bool(np.isfinite(res["g3_lo"]) and res["g3_lo"] > cost)

    # G4 표본
    res["g4_pass"] = len(keep) >= MIN_NONOVERLAP

    gates = [res["g1_pass"], res["g2_pass"], res["g3_pass"], res["g4_pass"]]
    res["verdict"] = "통과" if all(gates) else "탈락"
    res["failed"] = [g for g, ok in zip(["G1구간", "G2중첩", "G3블록", "G4표본"], gates) if not ok]
    return res


def screen_all(rules, timeframes, market, symbol="BTCUSDT", seed=11, record=True):
    """규칙 목록 × 타임프레임 전체를 평가하고 다중비교 카운터를 갱신한다."""
    rng = np.random.default_rng(seed)
    state = load_state()
    out = {}
    cells = 0
    for tf, H in timeframes:
        df = D.klines(tf, market, symbol)
        rows = [screen_rule(df, r, H, market, rng) for r in rules]
        out[tf] = {"H": H, "rows": rows, "bars": len(df),
                   "range": (str(df["dt"].iloc[0].date()), str(df["dt"].iloc[-1].date()))}
        cells += len(rows)
    if record:
        state["cells_tested"] += cells
        state["runs"].append({"market": market, "symbol": symbol,
                              "timeframes": [t for t, _ in timeframes],
                              "rules": len(rules), "cells": cells})
        save_state(state)
    out["_meta"] = {"cells_this_run": cells,
                    "cells_cumulative": state["cells_tested"],
                    "cost_bp": D.COSTS[market]}
    return out
