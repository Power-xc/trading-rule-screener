# Trading Rule Screener

I extracted 33 mechanical rules from **13 trading books (4,521 pages)** and machine-tested every one of them on 9 years of BTCUSDT spot data.

**99 cells tested. 0 passed.**

This repo is the test harness. You can run it yourself.

```bash
pip install -r requirements.txt
python -m screener.run_screen
```

---

## What it does

Most backtests answer "did this make money in the past?" That question is nearly useless — with enough parameters, everything made money in the past.

This asks a harder question: **does the signal survive when you stop lying to yourself?**

A rule must clear four gates. All four, or it fails.

| Gate | Test | Why |
|---|---|---|
| **G1** | Positive excess return in all 3 time segments | Kills rules that only worked in one regime |
| **G2** | **Overlap-corrected** bootstrap CI above cost | See below — this is the one that matters |
| **G3** | Block bootstrap CI above cost | Accounts for serial dependence |
| **G4** | ≥60 non-overlapping signals | Kills small-sample flukes |

The measured quantity is **excess return**: the mean forward return after a signal, minus the mean forward return of *all* bars in the same period. BTC's long-term drift is removed. Without this, every long-biased rule looks brilliant.

Costs are subtracted, not ignored: 24bp round trip for spot (0.10% taker × 2 + slippage), 14bp for perpetuals.

---

## G2: why overlapping observations wreck backtests

This is the gate that killed my most promising finding, so it gets its own section.

If you measure "return over the next N bars after a signal," and signals fire on consecutive bars, your observations **share most of their forward window**. They are not independent. Bootstrap them as if they were and your confidence interval is far too narrow.

Real numbers from this repo — 55-day Donchian breakout on 4h bars:

```
Raw:                657 signals, +94bp excess, 90% CI [+60, +128]   → PASSES
64% of signals overlapped with the previous one.

Overlap-corrected:  278 signals, +51bp excess, 90% CI [-2, +106]    → FAILS
```

Same data. Same rule. The only change is refusing to count the same information twice. The point estimate halves and the interval swallows zero.

I would have shipped this rule. The gate stopped me.

---

## What was tested

Rules are ported verbatim from the source books, with the quote attached in code:

```python
@rule("55-day high breakout", "Schwager, A Complete Guide to the Futures Market",
      "treat a breakout above the previous high as a buy signal", side=+1, family="trend")
def _(df):
    o, h, l, c, v = _ohlcv(df)
    return c > rolling_max(h, 55)
```

Families covered: moving average crossovers, Donchian breakouts, RSI, Bollinger, Stochastic, Fibonacci retracement, Envelope, volume confirmation, and 16 candlestick reversal patterns (hammer, engulfing, harami, morning/evening star, three soldiers/crows, and so on).

### Results

Nothing cleared all four gates. But the *sign pattern* was clean and consistent:

| Family | Direction |
|---|---|
| Trend / breakout / volume-confirmed breakout | **positive**, every rule |
| Mean reversion (RSI, Bollinger, Stochastic, Fibonacci, Envelope) | **negative**, every rule |

Longer breakout lookbacks scored monotonically higher (10d → 20d → 55d). That dose-response is more interesting than any single result.

Separately, on perpetual futures with walk-forward validation:

| Timeframe | Out-of-sample result |
|---|---|
| 15m | **−10.6bp per trade** (t = −2.78, n = 1,536) |
| 1h | −0.2bp (no edge) |
| 4h | −0.64bp (no edge) |

For a fixed 15m Donchian rule the full-sample figure was **−14.4bp over 5,738 trades, t = −16.66**. Round-trip cost is 14bp. The loss is almost exactly the fee. Directional predictive power at that horizon is indistinguishable from zero — you are paying the exchange for the privilege of guessing.

---

## Statistical power

A negative result only means something if the test could have detected a positive one. So it is checked directly, on synthetic series with a known edge planted in them:

```
Planted edge   Walk-forward OOS      Verdict
0 (rand walk)  -25.6bp  CI ≤ 0       correctly negative
+10bp/bar      +66.7bp  CI [+24, …]  DETECTED
+25bp/bar      +314bp   CI [+243,…]  DETECTED
```

It finds a 10bp-per-bar edge. It does not hallucinate one in a random walk. The negatives above are about the market, not about the instrument.

---

## Multiple comparisons

`_screen_state.json` tracks the cumulative number of cells you have tested, and every run prints it back at you.

This is not bookkeeping — it is the whole game. A random signal clears the 3-segment gate roughly **0.5–1% of the time**. Test 200 combinations of indicators and you should *expect* one or two "discoveries" that are pure noise.

There are ~100 indicators on any charting platform. Pairwise, that is ~5,000 combinations, of which ~50 will look significant by chance alone. Searching harder makes this worse, not better.

**A pass is a candidate for re-testing on independent data. It is never a conclusion.**

---

## Self-tests run before any screening

If a rule's definition is wrong, its screening result is noise dressed as evidence. So `selftest.py` runs first and blocks the screen on failure.

Every check here exists because it caught a real bug in my own code:

| Check | The bug it caught |
|---|---|
| `test_wick_basis` | Wick limits computed against *body* instead of *range* — silently rejected every doji-bodied hammer |
| `test_trend_timing` | Trend context measured at the pattern's *last* bar — a big bullish candle destroyed its own precondition, erasing morning stars |
| `test_rolling_shift` | Missing `shift(1)` on rolling max/min — the current bar was inside its own lookback window |
| `test_rules_fire` | Rules that never triggered at all, recorded as "no edge" rather than "broken definition" |
| `test_no_lookahead` | Future bars changing past signals |
| `test_rsi_reference` | Wilder RSI drift against reference implementation |

Total defects the harness caught in the code that was supposed to be testing the market: **15**.

Six AI agents were run in parallel to search for edges during this work. **Five returned materially wrong results** — including one that reported a profitable 1h strategy that reproduced at −0.2bp when tested independently. Every finding here was re-derived by hand before being written down.

---

## Adding a rule

```python
@rule("name", "source book", "verbatim quote from the book", side=+1, family="trend")
def _(df):
    o, h, l, c, v = _ohlcv(df)
    return c > rolling_max(h, 20)      # return a boolean array
```

The source and quote are **required fields**, and that is deliberate. Once you start screening, the temptation is to nudge a parameter and try again — which is grid search with extra steps, and it is exactly how overfitted strategies get born. If you cannot cite where a rule came from, do not register it.

Add a synthetic scenario to `selftest.py` at the same time, asserting the rule fires when it should.

---

## What this is not

It is not a strategy, and it will not tell you what to trade. It only tells you what to stop trading.

The one thing that survived this process, for me, was daily-timeframe trend following — which is neither novel nor exciting, and underperforms buy-and-hold on return while cutting drawdown to about a third. That result is not in this repo.

Data is pulled from Binance public endpoints and cached locally. No API key needed.

MIT.
