#!/usr/bin/env python3
"""스크리닝 실행 진입점.

사용법:
  python -m research.run_screen                      # 현물 1d/4h/1h 전체
  python -m research.run_screen --market futures     # 선물 (비용 14bp)
  python -m research.run_screen --tf 4h --horizon 12 # 단일 타임프레임
  python -m research.run_screen --selftest-only      # 자체 검사만

자체 검사에 실패하면 스크리닝으로 넘어가지 않는다. 정의가 틀린 규칙의 결과는
읽을 가치가 없기 때문이다.
"""
import argparse
import sys

from . import screen as S
from .data import COSTS
from .rules import RULES
from .selftest import run as run_selftest

DEFAULT_TF = [("1d", 5), ("4h", 12), ("1h", 24)]


def fmt(rows, cost):
    print(f"{'규칙':<32}{'계열':>6}{'S1':>8}{'S2':>8}{'S3':>8}"
          f"{'전체':>8}{'비중첩':>8}{'G2하한':>8}{'중첩%':>7}{'n':>7}  판정")
    for r in sorted(rows, key=lambda x: -(x.get("total") if x.get("total") is not None else -9e9)):
        if r.get("error"):
            print(f"{r['name'][:31]:<32} ERROR {r['error']}")
            continue
        seg = "".join(f"{v:>8.0f}" if v is not None else f"{'-':>8}" for v in r["seg"])
        tot = f"{r['total']:>8.0f}" if r.get("total") is not None else f"{'-':>8}"
        no = f"{r['nonoverlap']:>8.0f}" if r.get("nonoverlap") is not None else f"{'-':>8}"
        lo = f"{r['g2_lo']:>8.0f}" if r.get("g2_lo") == r.get("g2_lo") else f"{'-':>8}"
        ov = f"{r.get('overlap_ratio', 0)*100:>6.0f}%"
        v = r["verdict"]
        if v == "탈락" and r.get("failed"):
            v = "탈락(" + ",".join(x[:2] for x in r["failed"]) + ")"
        print(f"{r['name'][:31]:<32}{r.get('family','-'):>6}{seg}{tot}{no}{lo}{ov}{r['n']:>7}  {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="spot", choices=["spot", "futures"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--tf", default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--selftest-only", action="store_true")
    p.add_argument("--no-record", action="store_true",
                   help="다중비교 카운터에 기록하지 않는다 (실험용)")
    a = p.parse_args()

    print("=" * 104)
    print(f"자체 검사 — 규칙 {len(RULES)}개")
    if not run_selftest():
        print("\n자체 검사 실패. 규칙 정의를 고치기 전까지 스크리닝 결과는 신뢰할 수 없다.")
        return 1
    if a.selftest_only:
        return 0

    tfs = [(a.tf, a.horizon or 12)] if a.tf else DEFAULT_TF
    res = S.screen_all(RULES, tfs, a.market, a.symbol, record=not a.no_record)
    meta = res.pop("_meta")

    for tf, blk in res.items():
        lo, hi = blk["range"]
        print(f"\n{'='*104}")
        print(f"[{tf}] +{blk['H']}봉 · {a.market} {a.symbol} · {blk['bars']:,}봉 · {lo} ~ {hi} "
              f"· 비용 {meta['cost_bp']:.0f}bp")
        fmt(blk["rows"], meta["cost_bp"])

    passed = [r for blk in res.values() for r in blk["rows"] if r.get("verdict") == "통과"]
    print(f"\n{'='*104}")
    print(f"이번 실행 검정 칸수 {meta['cells_this_run']}  |  누적 {meta['cells_cumulative']}")
    print(f"통과 {len(passed)}개: {[r['name'] for r in passed] or '없음'}")
    if meta["cells_cumulative"] > 50:
        print(f"주의: 누적 {meta['cells_cumulative']}칸을 검정했다. 통과 1~2개는 우연히도 나온다. "
              f"새 통과 건은 별도 기간·심볼에서 재확인할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
