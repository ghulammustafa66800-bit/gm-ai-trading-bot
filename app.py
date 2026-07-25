from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime, timezone

app = Flask(__name__)

# =========================================================
# GM AI TRADING BOT - ALL IN ONE
# =========================================================

MEXC_BASE = "https://api.mexc.com"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 500

# =========================================================
# 1. MEXC API - SYMBOL VALIDATION
# =========================================================

def get_exchange_symbols():
    url = f"{MEXC_BASE}/api/v3/exchangeInfo"

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    data = r.json()

    symbols = set()

    for item in data.get("symbols", []):
        symbol = item.get("symbol")

        if symbol:
            symbols.add(symbol.upper())

    return symbols


def normalize_symbol(symbol):

    symbol = str(symbol).upper().strip()

    # Common formatting fixes
    symbol = symbol.replace("/", "")
    symbol = symbol.replace("-", "")
    symbol = symbol.replace("_", "")

    return symbol


# =========================================================
# 2. GET MARKET DATA
# =========================================================

def get_data(
    symbol=DEFAULT_SYMBOL,
    interval=DEFAULT_INTERVAL,
    limit=500
):

    symbol = normalize_symbol(symbol)

    try:
        limit = int(limit)
    except:
        limit = 500

    limit = max(100, min(limit, 1000))

    # Validate symbol first
    try:

        available_symbols = get_exchange_symbols()

        if symbol not in available_symbols:

            raise Exception(
                f"Invalid MEXC symbol: {symbol}. "
                f"Example: BTCUSDT"
            )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"MEXC exchangeInfo API error: {str(e)}"
        )

    url = f"{MEXC_BASE}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"MEXC connection error: {str(e)}"
        )

    if response.status_code != 200:

        raise Exception(
            f"MEXC API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    if not isinstance(data, list):

        raise Exception(
            f"Unexpected MEXC response: {data}"
        )

    if len(data) < 100:

        raise Exception(
            "Not enough market candles"
        )

    # MEXC candle format can vary slightly.
    # We only use the first 6 required fields.

    rows = []

    for candle in data:

        if len(candle) < 6:
            continue

        rows.append([
            candle[0],
            candle[1],
            candle[2],
            candle[3],
            candle[4],
            candle[5]
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(
        drop=True
    )

    return df


# =========================================================
# 3. TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # RSI - Wilder style
    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = (
        ema12 -
        ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # ATR
    high_low = (
        df["high"] -
        df["low"]
    )

    high_close = abs(
        df["high"] -
        df["close"].shift()
    )

    low_close = abs(
        df["low"] -
        df["close"].shift()
    )

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(
        axis=1
    )

    df["ATR"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    # Volume
    df["VOL_AVG"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["VOLUME_RATIO"] = (
        df["volume"] /
        df["VOL_AVG"].replace(
            0,
            np.nan
        )
    )

    # Momentum
    df["MOMENTUM_5"] = (
        df["close"]
        .pct_change(5)
        * 100
    )

    df["MOMENTUM_15"] = (
        df["close"]
        .pct_change(15)
        * 100
    )

    # Candle body
    df["BODY"] = abs(
        df["close"] -
        df["open"]
    )

    # Upper wick
    df["UPPER_WICK"] = (
        df["high"] -
        df[
            ["open", "close"]
        ].max(axis=1)
    )

    # Lower wick
    df["LOWER_WICK"] = (
        df[
            ["open", "close"]
        ].min(axis=1)
        -
        df["low"]
    )

    return df


# =========================================================
# 4. SUPPORT / RESISTANCE
# =========================================================

def support_resistance(df):

    recent = df.tail(100)

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    # Pivot-style levels
    last = df.iloc[-1]

    pivot = (
        last["high"] +
        last["low"] +
        last["close"]
    ) / 3

    r1 = (
        2 * pivot -
        last["low"]
    )

    s1 = (
        2 * pivot -
        last["high"]
    )

    return {
        "support": float(support),
        "resistance": float(resistance),
        "pivot": float(pivot),
        "s1": float(s1),
        "r1": float(r1)
    }


# =========================================================
# 5. MARKET STRUCTURE
# =========================================================

def market_structure(df):

    recent = df.tail(20)

    first = recent.iloc[:10]

    second = recent.iloc[10:]

    first_high = first["high"].max()
    second_high = second["high"].max()

    first_low = first["low"].min()
    second_low = second["low"].min()

    if (
        second_high > first_high
        and
        second_low > first_low
    ):

        return "BULLISH"

    if (
        second_high < first_high
        and
        second_low < first_low
    ):

        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# 6. TRENDLINE
# =========================================================

def trendline_analysis(df):

    recent = df.tail(50).copy()

    x = np.arange(
        len(recent)
    )

    y = recent[
        "close"
    ].values

    if len(x) < 10:

        return {
            "trendline": "UNKNOWN",
            "slope_percent": 0
        }

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    avg_price = np.mean(y)

    slope_percent = (
        slope /
        avg_price
    ) * 100

    if slope_percent > 0.02:

        trend = "UPTREND"

    elif slope_percent < -0.02:

        trend = "DOWNTREND"

    else:

        trend = "FLAT"

    return {
        "trendline": trend,
        "slope_percent": round(
            float(
                slope_percent
            ),
            4
        )
    }


# =========================================================
# 7. BREAKOUT / BREAKDOWN
# =========================================================

def breakout_analysis(
    df,
    levels
):

    last = df.iloc[-1]

    previous = df.iloc[-2]

    support = levels[
        "support"
    ]

    resistance = levels[
        "resistance"
    ]

    if (
        last["close"] >
        resistance
        and
        previous["close"] <=
        resistance
    ):

        return "BREAKOUT"

    if (
        last["close"] <
        support
        and
        previous["close"] >=
        support
    ):

        return "BREAKDOWN"

    return "NO_BREAKOUT"


# =========================================================
# 8. GAP UP / GAP DOWN
# =========================================================

def gap_analysis(df):

    if len(df) < 2:

        return {
            "gap": "NO_GAP",
            "gap_percent": 0
        }

    last = df.iloc[-1]

    previous = df.iloc[-2]

    gap_percent = (
        (
            last["open"] -
            previous["close"]
        )
        /
        previous["close"]
    ) * 100

    if gap_percent > 0.3:

        gap = "GAP_UP"

    elif gap_percent < -0.3:

        gap = "GAP_DOWN"

    else:

        gap = "NO_GAP"

    return {
        "gap": gap,
        "gap_percent": round(
            float(
                gap_percent
            ),
            4
        )
    }


# =========================================================
# 9. LIQUIDITY / TRAP ANALYSIS
# =========================================================

def trap_analysis(
    df,
    levels
):

    last = df.iloc[-1]

    support = levels[
        "support"
    ]

    resistance = levels[
        "resistance"
    ]

    volume_ratio = float(
        last[
            "VOLUME_RATIO"
        ]
    )

    if math.isnan(
        volume_ratio
    ):

        volume_ratio = 1.0

    fake_breakout = (
        last["high"] >
        resistance
        and
        last["close"] <
        resistance
    )

    fake_breakdown = (
        last["low"] <
        support
        and
        last["close"] >
        support
    )

    if (
        fake_breakout
        or
        fake_breakdown
    ):

        if volume_ratio < 1:

            risk = "HIGH"

        else:

            risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "trap_risk": risk,
        "fake_breakout": bool(
            fake_breakout
        ),
        "fake_breakdown": bool(
            fake_breakdown
        )
    }


# =========================================================
# 10. LIQUIDATION RISK ESTIMATE
# =========================================================

def liquidation_risk(
    df,
    signal,
    atr
):

    last = df.iloc[-1]

    price = float(
        last["close"]
    )

    volume_ratio = float(
        last[
            "VOLUME_RATIO"
        ]
    )

    if math.isnan(
        volume_ratio
    ):

        volume_ratio = 1.0

    atr_percent = (
        atr /
        price
    ) * 100

    risk_score = 0

    if atr_percent > 1:

        risk_score += 3

    elif atr_percent > 0.5:

        risk_score += 2

    else:

        risk_score += 1

    if volume_ratio > 2:

        risk_score += 2

    elif volume_ratio > 1.5:

        risk_score += 1

    if signal in [
        "BUY",
        "STRONG BUY",
        "SELL",
        "STRONG SELL"
    ]:

        pass

    if risk_score >= 5:

        risk = "HIGH"

    elif risk_score >= 3:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "liquidation_risk": risk,
        "liquidation_risk_score": risk_score,
        "atr_percent": round(
            atr_percent,
            4
        )
    }


# =========================================================
# 11. MULTI-TIMEFRAME ANALYSIS
# =========================================================

def timeframe_trend(df):

    last = df.iloc[-1]

    if (
        last["EMA20"] >
        last["EMA50"]
    ):

        return "BULLISH"

    elif (
        last["EMA20"] <
        last["EMA50"]
    ):

        return "BEARISH"

    return "SIDEWAYS"


def get_mtf_confirmation(
    symbol
):

    results = {}

    intervals = [
        "5m",
        "15m",
        "1h"
    ]

    for interval in intervals:

        try:

            data = get_data(
                symbol,
                interval,
                300
            )

            data = calculate_indicators(
                data
            )

            results[
                interval
            ] = timeframe_trend(
                data
            )

        except Exception:

            results[
                interval
            ] = "UNAVAILABLE"

    available = [
        v
        for v in results.values()
        if v != "UNAVAILABLE"
    ]

    bullish = available.count(
        "BULLISH"
    )

    bearish = available.count(
        "BEARISH"
    )

    if bullish >= 2:

        confirmation = "BULLISH"

    elif bearish >= 2:

        confirmation = "BEARISH"

    else:

        confirmation = "MIXED"

    return {
        "timeframes": results,
        "confirmation": confirmation
    }


# =========================================================
# 12. MONEY MANAGEMENT
# =========================================================

def money_management(
    price,
    stop_loss,
    account_size=1000,
    risk_percent=1
):

    if stop_loss is None:

        return {
            "account_size": account_size,
            "risk_percent": risk_percent,
            "risk_amount": 0,
            "position_size": 0,
            "position_value": 0
        }

    risk_amount = (
        account_size *
        risk_percent /
        100
    )

    stop_distance = abs(
        price -
        stop_loss
    )

    if stop_distance <= 0:

        return {
            "account_size": account_size,
            "risk_percent": risk_percent,
            "risk_amount": 0,
            "position_size": 0,
            "position_value": 0
        }

    position_size = (
        risk_amount /
        stop_distance
    )

    position_value = (
        position_size *
        price
    )

    return {
        "account_size": round(
            account_size,
            2
        ),
        "risk_percent": round(
            risk_percent,
            2
        ),
        "risk_amount": round(
            risk_amount,
            2
        ),
        "position_size": round(
            position_size,
            8
        ),
        "position_value": round(
            position_value,
            2
        )
    }


# =========================================================
# 13. COMPLETE SIGNAL ENGINE
# =========================================================

def generate_analysis(
    df,
    symbol="BTCUSDT",
    account_size=1000,
    risk_percent=1
):

    last = df.iloc[-1]

    levels = support_resistance(
        df
    )

    structure = market_structure(
        df
    )

    trendline = trendline_analysis(
        df
    )

    breakout = breakout_analysis(
        df,
        levels
    )

    gap = gap_analysis(
        df
    )

    trap = trap_analysis(
        df,
        levels
    )

    price = float(
        last["close"]
    )

    atr = float(
        last["ATR"]
    )

    score = 0

    reasons = []

    # =====================================================
    # EMA TREND
    # =====================================================

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):

        score += 3

        reasons.append(
            "Strong bullish EMA alignment"
        )

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):

        score -= 3

        reasons.append(
            "Strong bearish EMA alignment"
        )

    # =====================================================
    # RSI
    # =====================================================

    rsi = float(
        last["RSI"]
    )

    if 50 < rsi < 70:

        score += 1

        reasons.append(
            "Bullish RSI momentum"
        )

    elif 30 < rsi < 50:

        score -= 1

        reasons.append(
            "Bearish RSI momentum"
        )

    elif rsi >= 70:

        reasons.append(
            "RSI overbought warning"
        )

    elif rsi <= 30:

        reasons.append(
            "RSI oversold warning"
        )

    # =====================================================
    # MACD
    # =====================================================

    if (
        last["MACD"] >
        last["MACD_SIGNAL"]
    ):

        score += 2

        reasons.append(
            "MACD bullish"
        )

    else:

        score -= 2

        reasons.append(
            "MACD bearish"
        )

    # =====================================================
    # MARKET STRUCTURE
    # =====================================================

    if structure == "BULLISH":

        score += 2

        reasons.append(
            "Bullish market structure"
        )

    elif structure == "BEARISH":

        score -= 2

        reasons.append(
            "Bearish market structure"
        )

    # =====================================================
    # TRENDLINE
    # =====================================================

    if trendline[
        "trendline"
    ] == "UPTREND":

        score += 1

        reasons.append(
            "Price trendline is rising"
        )

    elif trendline[
        "trendline"
    ] == "DOWNTREND":

        score -= 1

        reasons.append(
            "Price trendline is falling"
        )

    # =====================================================
    # BREAKOUT
    # =====================================================

    if breakout == "BREAKOUT":

        score += 2

        reasons.append(
            "Resistance breakout detected"
        )

    elif breakout == "BREAKDOWN":

        score -= 2

        reasons.append(
            "Support breakdown detected"
        )

    # =====================================================
    # VOLUME
    # =====================================================

    volume_ratio = float(
        last[
            "VOLUME_RATIO"
        ]
    )

    if math.isnan(
        volume_ratio
    ):

        volume_ratio = 1

    volume_confirmed = (
        volume_ratio > 1.2
    )

    if volume_confirmed:

        reasons.append(
            "Volume confirmation present"
        )

        if score > 0:

            score += 1

        elif score < 0:

            score -= 1

    # =====================================================
    # TRAP FILTER
    # =====================================================

    if trap[
        "trap_risk"
    ] == "HIGH":

        score = int(
            score * 0.5
        )

        reasons.append(
            "HIGH trap risk - signal reduced"
        )

    elif trap[
        "trap_risk"
    ] == "MEDIUM":

        reasons.append(
            "MEDIUM trap risk"
        )

    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    if score >= 7:

        signal = "STRONG BUY"

    elif score >= 3:

        signal = "BUY"

    elif score <= -7:

        signal = "STRONG SELL"

    elif score <= -3:

        signal = "SELL"

    else:

        signal = "NEUTRAL"

    # =====================================================
    # ENTRY / STOP / TARGET
    # =====================================================

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        entry = price

        stop_loss = (
            price -
            atr * 1.5
        )

        take_profit_1 = (
            price +
            atr * 2
        )

        take_profit_2 = (
            price +
            atr * 3
        )

    elif signal in [
        "SELL",
        "STRONG SELL"
    ]:

        entry = price

        stop_loss = (
            price +
            atr * 1.5
        )

        take_profit_1 = (
            price -
            atr * 2
        )

        take_profit_2 = (
            price -
            atr * 3
        )

    else:

        entry = price

        stop_loss = None

        take_profit_1 = None

        take_profit_2 = None

    # =====================================================
    # RISK REWARD
    # =====================================================

    if (
        stop_loss is not None
        and
        take_profit_1 is not None
    ):

        risk = abs(
            entry -
            stop_loss
        )

        reward = abs(
            take_profit_1 -
            entry
        )

        risk_reward = (
            reward /
            risk
            if risk > 0
            else 0
        )

    else:

        risk_reward = 0

    # =====================================================
    # MONEY MANAGEMENT
    # =====================================================

    money = money_management(
        price,
        stop_loss,
        account_size,
        risk_percent
    )

    # =====================================================
    # LIQUIDATION RISK
    # =====================================================

    liquidation = liquidation_risk(
        df,
        signal,
        atr
    )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = min(
        95,
        50 +
        abs(score) * 5
    )

    # =====================================================
    # RESULT
    # =====================================================

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "timeframe":
            "5m",

        "signal":
            signal,

        "confidence":
            round(
                confidence,
                2
            ),

        "score":
            score,

        "current_price":
            round(
                price,
                8
            ),

        "trend":
            structure,

        "trendline":
            trendline,

        "support":
            round(
                levels[
                    "support"
                ],
                8
            ),

        "resistance":
            round(
                levels[
                    "resistance"
                ],
                8
            ),

        "pivot":
            round(
                levels[
                    "pivot"
                ],
                8
            ),

        "breakout":
            breakout,

        "gap":
            gap,

        "trap":
            trap,

        "liquidation":
            liquidation,

        "entry":
            round(
                entry,
                8
            ),

        "stop_loss":
            (
                round(
                    stop_loss,
                    8
                )
                if stop_loss
                is not None
                else None
            ),

        "take_profit_1":
            (
                round(
                    take_profit_1,
                    8
                )
                if take_profit_1
                is not None
                else None
            ),

        "take_profit_2":
            (
                round(
                    take_profit_2,
                    8
                )
                if take_profit_2
                is not None
                else None
            ),

        "risk_reward":
            round(
                risk_reward,
                2
            ),

        "money_management":
            money,

        "rsi":
            round(
                rsi,
                2
            ),

        "ema20":
            round(
                float(
                    last[
                        "EMA20"
                    ]
                ),
                8
            ),

        "ema50":
            round(
                float(
                    last[
                        "EMA50"
                    ]
                ),
                8
            ),

        "ema200":
            round(
                float(
                    last[
                        "EMA200"
                    ]
                ),
                8
            ),

        "macd":
            round(
                float(
                    last[
                        "MACD"
                    ]
                ),
                8
            ),

        "macd_signal":
            round(
                float(
                    last[
                        "MACD_SIGNAL"
                    ]
                ),
                8
            ),

        "momentum_5m_percent":
            round(
                float(
                    last[
                        "MOMENTUM_5"
                    ]
                ),
                4
            ),

        "momentum_15m_percent":
            round(
                float(
                    last[
                        "MOMENTUM_15"
                    ]
                ),
                4
            ),

        "volume_ratio":
            round(
                volume_ratio,
                3
            ),

        "volume_confirmed":
            bool(
                volume_confirmed
            ),

        "reasons":
            reasons
    }


# =========================================================
# 14. LIVE ANALYSIS
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "bot":
            "GM AI Trading Bot",

        "version":
            "All-In-One v6",

        "message":
            "Trading analysis system is running.",

        "example":
            "/analysis/BTCUSDT"

    })


@app.route("/analysis/<symbol>")
def analysis(symbol):

    try:

        symbol = normalize_symbol(
            symbol
        )

        account_size = float(
            request.args.get(
                "account_size",
                1000
            )
        )

        risk_percent = float(
            request.args.get(
                "risk_percent",
                1
            )
        )

        df = get_data(
            symbol,
            "5m",
            500
        )

        df = calculate_indicators(
            df
        )

        result = generate_analysis(
            df,
            symbol,
            account_size,
            risk_percent
        )

        # MTF confirmation
        try:

            mtf = get_mtf_confirmation(
                symbol
            )

            result[
                "multi_timeframe"
            ] = mtf

        except Exception as e:

            result[
                "multi_timeframe"
            ] = {
                "status":
                    "unavailable",
                "message":
                    str(e)
            }

        result[
            "timestamp"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        result[
            "warning"
        ] = (
            "Technical analysis only. "
            "No guaranteed prediction. "
            "Liquidation risk is estimated "
            "because direct exchange liquidation "
            "data is not included."
        )

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                symbol,

            "message":
                str(e),

            "hint":
                "Use a valid MEXC spot symbol, "
                "for example BTCUSDT."

        }), 400


# =========================================================
# 15. BACKTEST
# =========================================================

@app.route("/backtest/<symbol>")
def backtest(symbol):

    try:

        symbol = normalize_symbol(
            symbol
        )

        limit = int(
            request.args.get(
                "limit",
                500
            )
        )

        limit = max(
            250,
            min(
                limit,
                1000
            )
        )

        df = get_data(
            symbol,
            "5m",
            limit
        )

        df = calculate_indicators(
            df
        )

        correct = 0

        wrong = 0

        total_signals = 0

        neutral = 0

        # Start after indicators
        for i in range(
            200,
            len(df) - 1
        ):

            historical = (
                df.iloc[
                    :i + 1
                ].copy()
            )

            result = generate_analysis(
                historical,
                symbol,
                1000,
                1
            )

            signal = result[
                "signal"
            ]

            current_price = float(
                df.iloc[
                    i
                ][
                    "close"
                ]
            )

            next_price = float(
                df.iloc[
                    i + 1
                ][
                    "close"
                ]
            )

            if signal in [
                "BUY",
                "STRONG BUY"
            ]:

                total_signals += 1

                if next_price > current_price:

                    correct += 1

                else:

                    wrong += 1

            elif signal in [
                "SELL",
                "STRONG SELL"
            ]:

                total_signals += 1

                if next_price < current_price:

                    correct += 1

                else:

                    wrong += 1

            else:

                neutral += 1

        if total_signals > 0:

            win_rate = (
                correct /
                total_signals
            ) * 100

        else:

            win_rate = 0

        return jsonify({

            "status":
                "success",

            "symbol":
                symbol,

            "timeframe":
                "5m",

            "candles_tested":
                len(df) - 200,

            "total_signals":
                total_signals,

            "correct_signals":
                correct,

            "wrong_signals":
                wrong,

            "neutral_signals":
                neutral,

            "win_rate_percent":
                round(
                    win_rate,
                    2
                ),

            "note":
                "This is a simple historical "
                "one-candle backtest. It does not "
                "include fees, slippage, funding, "
                "or real liquidation data. "
                "Past performance does not guarantee "
                "future results."

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                symbol,

            "message":
                str(e)

        }), 400


# =========================================================
# 16. RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8080
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
