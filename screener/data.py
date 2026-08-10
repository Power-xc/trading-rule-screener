"""연구용 데이터 로더 + 시장별 비용 상수.

운영 코드(`backtest/data_loader.py`)와 분리되어 있다. 운영 쪽은 v1 전용이고
여기는 스크리닝 전용이라 캐시 포맷과 기간 기준이 다르다.

현물/선물을 명시적으로 선택하게 만든 이유:
v1 백테스트가 현물 캔들로 검증하고 선물로 실거래하던 불일치가 REPORT.md에서
지적됐다. 여기서는 market 인자를 필수로 받아 그 실수를 구조적으로 막는다.
"""
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

SPOT_BASE = "https://api.binance.com"
FUT_BASE = "https://fapi.binance.com"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")

# 왕복 비용 (bp). 진입+청산 양쪽의 수수료와 슬리피지를 합산한 값이다.
# 스크리닝의 합격선이 되므로 낙관적으로 잡으면 안 된다.
COSTS = {
    # 바이낸스 현물 VIP0 테이커 0.10% × 2 + 슬리피지 0.02% × 2
    "spot": (0.0010 + 0.0002) * 2 * 10000,   # = 24bp
    # 바이낸스 USDT-M 선물 테이커 0.05% × 2 + 슬리피지 0.02% × 2
    "futures": (0.0005 + 0.0002) * 2 * 10000,  # = 14bp
}

# 심볼별 최소 주문 제약 (소액 운용 시 신호가 주문으로 이어지는지 확인용)
MIN_ORDER = {
    ("BTCUSDT", "futures"): {"min_notional": 50.0, "min_qty": 0.001},
    ("BTCUSDT", "spot"): {"min_notional": 5.0, "min_qty": 0.00001},
}

_COLS = ["ts", "open", "high", "low", "close", "vol",
         "ct", "qv", "n", "tb", "tq", "ig"]


def _endpoint(market: str) -> tuple:
    if market == "spot":
        return SPOT_BASE, "/api/v3/klines", 1000
    if market == "futures":
        return FUT_BASE, "/fapi/v1/klines", 1500
    raise ValueError(f"market은 'spot' 또는 'futures'여야 한다: {market!r}")


def klines(interval: str, market: str, symbol: str = "BTCUSDT",
           start: str = None, use_cache: bool = True) -> pd.DataFrame:
    """캔들 로드. 반환 컬럼: dt(UTC), ts, open, high, low, close, vol

    start 기본값은 시장별 상장 시점이다. 현물이 선물보다 2년 이상 길어서
    구간 분할 검증에 유리하다.
    """
    if start is None:
        start = "2017-08-01" if market == "spot" else "2019-09-01"
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{market}_{interval}.csv")
    if use_cache and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["dt"])
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        return df

    base, route, limit = _endpoint(market)
    start_ms = int(datetime.strptime(start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(f"{base}{route}", timeout=30, params={
            "symbol": symbol, "interval": interval,
            "startTime": cur, "limit": limit})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][6] + 1
        if len(batch) < limit:
            break
        time.sleep(0.1)

    df = pd.DataFrame(rows, columns=_COLS).drop_duplicates("ts")
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[["dt", "ts", "open", "high", "low", "close", "vol"]].reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def funding(symbol: str = "BTCUSDT", start: str = "2019-09-01") -> pd.Series:
    """8시간 펀딩 요율. 선물 전용 — 현물 스크리닝에는 쓰지 않는다."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_funding.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        return pd.Series(d["rate"].to_numpy(), index=d["ts"].to_numpy())
    start_ms = int(datetime.strptime(start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    out, cur = {}, start_ms
    while cur < end_ms:
        r = requests.get(f"{FUT_BASE}/fapi/v1/fundingRate", timeout=30, params={
            "symbol": symbol, "startTime": cur, "limit": 1000})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for x in batch:
            out[int(x["fundingTime"])] = float(x["fundingRate"])
        if batch[-1]["fundingTime"] <= cur:
            break
        cur = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.1)
    s = pd.Series(out).sort_index()
    pd.DataFrame({"ts": s.index, "rate": s.to_numpy()}).to_csv(path, index=False)
    return s
