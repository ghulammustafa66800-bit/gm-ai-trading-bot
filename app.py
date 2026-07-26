from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime, timezone

app = Flask(__name__)

# =========================================================
# GM AI TRADING BOT v7
# FULL MARKET ANALYSIS ENGINE
# =========================================================

MEXC_BASE = "https://api.mexc.com"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 500

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MIN_CANDLES = 250
DEFAULT_ACCOUNT_SIZE = 1000.0
DEFAULT_RISK_PERCENT = 1.0

# Minimum score required before taking a trade
MIN_TRADE_SCORE = 7

# Trading fee estimate per side
DEFAULT_FEE_RATE = 0.001


# =========================================================
# 1. SYMBOL
# =========================================================

def normalize_symbol(symbol):
    symbol = str(symbol).upper().strip()
    symbol = symbol.replace("/", "")
    symbol = symbol.replace("-", "")
    symbol = symbol.replace("_", "")
    return symbol


# =========================================================
# 2. GET MARKET DATA
# =========================================================

def get_data(symbol, interval="5m", limit=500):

    symbol = normalize_symbol(symbol)

    try:
        limit = int(limit)
    except Exception:
        limit = 500

    limit = max(MIN_CANDLES, min(limit, 1000))

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
        raise Exception(f"MEXC connection error: {str(e)}")

    if response.status_code != 200:
        raise Exception(
            f"MEXC API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    if not isinstance(data, list):
        raise Exception(f"Unexpected MEXC response: {data}")

    if len(data) < MIN_CANDLES:
        raise Exception(
            f"Not enough candles. Received {len(data)}, "
            f"required at least {MIN_CANDLES}."
        )

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

    df = df.dropna().reset_index(drop=True)

    return df


# =========================================================
# 3. INDICATORS
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

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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

    df["MACD"] = ema12 - ema26

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
    ).max(axis=1)

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
        .pct_change(5) * 100
    )

    df["MOMENTUM_15"] = (
        df["close"]
        .pct_change(15) * 100
    )

    # Candle anatomy
    df["BODY"] = abs(
        df["close"] -
        df["open"]
    )

    df["UPPER_WICK"] = (
        df["high"] -
        df[
            ["open", "close"]
        ].max(axis=1)
    )

    df["LOWER_WICK"] = (
        df[
            ["open", "close"]
        ].min(axis=1)
        -
        df["low"]
    )

    # Candle range
    df["RANGE"] = (
        df["high"] -
        df["low"]
    )

    # Bull / Bear candle
    df["BULL_CANDLE"] = (
        df["close"] >
        df["open"]
    )

    df["BEAR_CANDLE"] = (
        df["close"] <
        df["open"]
    )

    return df


# =========================================================
# 4. SUPPORT / RESISTANCE
# IMPORTANT: EXCLUDE CURRENT CANDLE
# =========================================================

def support_resistance(df, lookback=100):

    if len(df) < lookback + 2:
        lookback = len(df) - 2

    historical = df.iloc[
        -(lookback + 1):-1
    ]

    support = float(
        historical["low"].min()
    )

    resistance = float(
        historical["high"].max()
    )

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
        "support": support,
        "resistance": resistance,
        "pivot": float(pivot),
        "r1": float(r1),
        "s1": float(s1)
    }


# =========================================================
# 5. MARKET STRUCTURE
# =========================================================

def market_structure(df):

    recent = df.tail(40)

    if len(recent) < 20:
        return "SIDEWAYS"

    first = recent.iloc[:20]
    second = recent.iloc[20:]

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

    recent = df.tail(60)

    if len(recent) < 20:
        return {
            "trendline": "UNKNOWN",
            "slope_percent": 0
        }

    x = np.arange(len(recent))

    y = recent[
        "close"
    ].values

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    avg_price = np.mean(y)

    if avg_price == 0:
        return {
            "trendline": "UNKNOWN",
            "slope_percent": 0
        }

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
            float(slope_percent),
            4
        )
    }


# =========================================================
# 7. BREAKOUT / BREAKDOWN
# =========================================================

def breakout_analysis(df, levels):

    last = df.iloc[-1]
    previous = df.iloc[-2]

    support = levels["support"]
    resistance = levels["resistance"]

    breakout = (
        last["close"] > resistance
        and
        previous["close"] <= resistance
    )

    breakdown = (
        last["close"] < support
        and
        previous["close"] >= support
    )

    if breakout:
        return "BREAKOUT"

    if breakdown:
        return "BREAKDOWN"

    return "NO_BREAKOUT"


# =========================================================
# 8. LIQUIDITY SWEEP
# =========================================================

def liquidity_analysis(df, levels):

    last = df.iloc[-1]

    support = levels["support"]
    resistance = levels["resistance"]

    volume_ratio = float(
        last["VOLUME_RATIO"]
    )

    if math.isnan(volume_ratio):
        volume_ratio = 1.0

    sweep_high = (
        last["high"] > resistance
        and
        last["close"] < resistance
    )

    sweep_low = (
        last["low"] < support
        and
        last["close"] > support
    )

    if sweep_high:
        direction = "BEARISH_LIQUIDITY_SWEEP"

    elif sweep_low:
        direction = "BULLISH_LIQUIDITY_SWEEP"

    else:
        direction = "NONE"

    # Trap risk
    if (
        sweep_high
        or
        sweep_low
    ):

        if volume_ratio < 1.0:
            trap_risk = "HIGH"

        else:
            trap_risk = "MEDIUM"

    else:
        trap_risk = "LOW"

    return {
        "liquidity_sweep": direction,
        "trap_risk": trap_risk,
        "sweep_high": bool(sweep_high),
        "sweep_low": bool(sweep_low)
    }


# =========================================================
# 9. CANDLE CONFIRMATION
# =========================================================

def candle_analysis(df):

    last = df.iloc[-1]

    body = float(
        last["BODY"]
    )

    upper = float(
        last["UPPER_WICK"]
    )

    lower = float(
        last["LOWER_WICK"]
    )

    candle_range = float(
        last["RANGE"]
    )

    if candle_range <= 0:
        return {
            "pattern": "UNKNOWN",
            "bullish": False,
            "bearish": False
        }

    bullish_rejection = (
        lower > body * 1.5
        and
        last["close"] > last["open"]
    )

    bearish_rejection = (
        upper > body * 1.5
        and
        last["close"] < last["open"]
    )

    strong_bull = (
        last["close"] > last["open"]
        and
        body / candle_range > 0.6
    )

    strong_bear = (
        last["close"] < last["open"]
        and
        body / candle_range > 0.6
    )

    if bullish_rejection:
        pattern = "BULLISH_REJECTION"

    elif bearish_rejection:
        pattern = "BEARISH_REJECTION"

    elif strong_bull:
        pattern = "STRONG_BULLISH_CANDLE"

    elif strong_bear:
        pattern = "STRONG_BEARISH_CANDLE"

    else:
        pattern = "NEUTRAL_CANDLE"

    return {
        "pattern": pattern,
        "bullish": bool(
            bullish_rejection
            or
            strong_bull
        ),
        "bearish": bool(
            bearish_rejection
            or
            strong_bear
        )
    }


# =========================================================
# 10. MTF TREND
# =========================================================

def timeframe_trend(df):

    last = df.iloc[-1]

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):
        return "BULLISH"

    if (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):
        return "BEARISH"

    if (
        last["EMA20"] >
        last["EMA50"]
    ):
        return "WEAK_BULLISH"

    if (
        last["EMA20"] <
        last["EMA50"]
    ):
        return "WEAK_BEARISH"

    return "SIDEWAYS"


def get_mtf_analysis(symbol):

    intervals = [
        "5m",
        "15m",
        "1h",
        "4h"
    ]

    results = {}

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

        except Exception as e:

            results[
                interval
            ] = "UNAVAILABLE"

    bullish = 0
    bearish = 0

    for value in results.values():

        if value in [
            "BULLISH",
            "WEAK_BULLISH"
        ]:
            bullish += 1

        elif value in [
            "BEARISH",
            "WEAK_BEARISH"
        ]:
            bearish += 1

    if bullish >= 3:
        confirmation = "BULLISH"

    elif bearish >= 3:
        confirmation = "BEARISH"

    else:
        confirmation = "MIXED"

    return {
        "timeframes": results,
        "confirmation": confirmation,
        "bullish_count": bullish,
        "bearish_count": bearish
    }


# =========================================================
# 11. MONEY MANAGEMENT
# =========================================================

def money_management(
    entry,
    stop_loss,
    account_size=1000,
    risk_percent=1
):

    if (
        stop_loss is None
        or
        entry <= 0
    ):

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
        entry -
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
        entry
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
# 12. SIGNAL ENGINE
# =========================================================

def generate_analysis(
    df,
    symbol,
    mtf=None,
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

    liquidity = liquidity_analysis(
        df,
        levels
    )

    candle = candle_analysis(
        df
    )

    price = float(
        last["close"]
    )

    atr = float(
        last["ATR"]
    )

    rsi = float(
        last["RSI"]
    )

    macd = float(
        last["MACD"]
    )

    macd_signal = float(
        last["MACD_SIGNAL"]
    )

    volume_ratio = float(
        last["VOLUME_RATIO"]
    )

    if math.isnan(volume_ratio):
        volume_ratio = 1.0

    score = 0

    reasons = []

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 50 < rsi < 68:

        score += 1

        reasons.append(
            "Bullish RSI momentum"
        )

    elif 32 < rsi < 50:

        score -= 1

        reasons.append(
            "Bearish RSI momentum"
        )

    elif rsi >= 70:

        reasons.append(
            "Overbought RSI warning"
        )

    elif rsi <= 30:

        reasons.append(
            "Oversold RSI warning"
        )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if macd > macd_signal:

        score += 2

        reasons.append(
            "MACD bullish"
        )

    else:

        score -= 2

        reasons.append(
            "MACD bearish"
        )

    # -----------------------------------------------------
    # STRUCTURE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # TRENDLINE
    # -----------------------------------------------------

    if trendline[
        "trendline"
    ] == "UPTREND":

        score += 1

        reasons.append(
            "Rising trendline"
        )

    elif trendline[
        "trendline"
    ] == "DOWNTREND":

        score -= 1

        reasons.append(
            "Falling trendline"
        )

    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    if breakout == "BREAKOUT":

        score += 2

        reasons.append(
            "Resistance breakout"
        )

    elif breakout == "BREAKDOWN":

        score -= 2

        reasons.append(
            "Support breakdown"
        )

    # -----------------------------------------------------
    # LIQUIDITY SWEEP
    # -----------------------------------------------------

    if (
        liquidity[
            "liquidity_sweep"
        ]
        ==
        "BULLISH_LIQUIDITY_SWEEP"
    ):

        score += 2

        reasons.append(
            "Bullish liquidity sweep"
        )

    elif (
        liquidity[
            "liquidity_sweep"
        ]
        ==
        "BEARISH_LIQUIDITY_SWEEP"
    ):

        score -= 2

        reasons.append(
            "Bearish liquidity sweep"
        )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if volume_ratio >= 1.2:

        if score > 0:
            score += 1

        elif score < 0:
            score -= 1

        reasons.append(
            "Volume confirmation"
        )

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    momentum_5 = float(
        last["MOMENTUM_5"]
    )

    momentum_15 = float(
        last["MOMENTUM_15"]
    )

    if (
        momentum_5 > 0
        and
        momentum_15 > 0
    ):

        score += 1

        reasons.append(
            "Positive momentum"
        )

    elif (
        momentum_5 < 0
        and
        momentum_15 < 0
    ):

        score -= 1

        reasons.append(
            "Negative momentum"
        )

    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

    if candle["bullish"]:

        score += 1

        reasons.append(
            f"Bullish candle: "
            f"{candle['pattern']}"
        )

    elif candle["bearish"]:

        score -= 1

        reasons.append(
            f"Bearish candle: "
            f"{candle['pattern']}"
        )

    # -----------------------------------------------------
    # MTF CONFIRMATION
    # -----------------------------------------------------

    mtf_confirmation = "MIXED"

    if mtf:

        mtf_confirmation = mtf[
            "confirmation"
        ]

        if mtf_confirmation == "BULLISH":

            score += 2

            reasons.append(
                "Multi-timeframe bullish confirmation"
            )

        elif mtf_confirmation == "BEARISH":

            score -= 2

            reasons.append(
                "Multi-timeframe bearish confirmation"
            )

    # -----------------------------------------------------
    # TRAP FILTER
    # -----------------------------------------------------

    if liquidity[
        "trap_risk"
    ] == "HIGH":

        score = int(
            score * 0.5
        )

        reasons.append(
            "High trap risk"
        )

    elif liquidity[
        "trap_risk"
    ] == "MEDIUM":

        reasons.append(
            "Medium trap risk"
        )

    # =====================================================
    # RAW DIRECTION
    # =====================================================

    if score >= MIN_TRADE_SCORE:

        direction = "LONG"

    elif score <= -MIN_TRADE_SCORE:

        direction = "SHORT"

    else:

        direction = "NO_TRADE"

    # =====================================================
    # FINAL STRICT FILTER
    # =====================================================

    if direction == "LONG":

        if mtf_confirmation == "BEARISH":

            direction = "NO_TRADE"

            reasons.append(
                "Long rejected: higher timeframe bearish"
            )

        if liquidity[
            "trap_risk"
        ] == "HIGH":

            direction = "NO_TRADE"

            reasons.append(
                "Long rejected: high trap risk"
            )

    elif direction == "SHORT":

        if mtf_confirmation == "BULLISH":

            direction = "NO_TRADE"

            reasons.append(
                "Short rejected: higher timeframe bullish"
            )

        if liquidity[
            "trap_risk"
        ] == "HIGH":

            direction = "NO_TRADE"

            reasons.append(
                "Short rejected: high trap risk"
            )

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    entry = price

    stop_loss = None

    tp1 = None
    tp2 = None
    tp3 = None

    if direction == "LONG":

        structure_sl = levels[
            "support"
        ]

        atr_sl = (
            price -
            atr * 1.5
        )

        stop_loss = min(
            structure_sl,
            atr_sl
        )

        risk = (
            entry -
            stop_loss
        )

        if risk > 0:

            tp1 = entry + (
                risk * 1.5
            )

            tp2 = entry + (
                risk * 2.0
            )

            tp3 = entry + (
                risk * 3.0
            )

    elif direction == "SHORT":

        structure_sl = levels[
            "resistance"
        ]

        atr_sl = (
            price +
            atr * 1.5
        )

        stop_loss = max(
            structure_sl,
            atr_sl
        )

        risk = (
            stop_loss -
            entry
        )

        if risk > 0:

            tp1 = entry - (
                risk * 1.5
            )

            tp2 = entry - (
                risk * 2.0
            )

            tp3 = entry - (
                risk * 3.0
            )

    # =====================================================
    # RISK REWARD
    # =====================================================

    if (
        stop_loss is not None
        and
        tp1 is not None
    ):

        risk = abs(
            entry -
            stop_loss
        )

        reward = abs(
            tp1 -
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
    # RISK/REWARD FILTER
    # =====================================================

    if (
        direction != "NO_TRADE"
        and
        risk_reward < 1.5
    ):

        direction = "NO_TRADE"

        reasons.append(
            "Trade rejected: Risk/Reward below 1.5"
        )

        stop_loss = None
        tp1 = None
        tp2 = None
        tp3 = None

    # =====================================================
    # SIGNAL NAME
    # =====================================================

    if direction == "LONG":

        if score >= 10:
            signal = "STRONG BUY"
        else:
            signal = "BUY"

    elif direction == "SHORT":

        if score <= -10:
            signal = "STRONG SELL"
        else:
            signal = "SELL"

    else:

        signal = "NO TRADE"

    # =====================================================
    # CONFIDENCE
    # =====================================================

    max_score = 18

    confidence = (
        abs(score) /
        max_score
    ) * 100

    confidence = max(
        0,
        min(
            95,
            confidence
        )
    )

    if direction == "NO_TRADE":

        confidence = min(
            confidence,
            55
        )

    # =====================================================
    # MONEY MANAGEMENT
    # =====================================================

    money = money_management(
        entry,
        stop_loss,
        account_size,
        risk_percent
    )

    # =====================================================
    # RESULT
    # =====================================================

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "signal":
            signal,

        "direction":
            direction,

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

        "market_structure":
            structure,

        "trendline":
            trendline,

        "multi_timeframe":
            mtf,

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

        "liquidity":
            liquidity,

        "candle":
            candle,

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
                if stop_loss is not None
                else None
            ),

        "take_profit_1":
            (
                round(
                    tp1,
                    8
                )
                if tp1 is not None
                else None
            ),

        "take_profit_2":
            (
                round(
                    tp2,
                    8
                )
                if tp2 is not None
                else None
            ),

        "take_profit_3":
            (
                round(
                    tp3,
                    8
                )
                if tp3 is not None
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
                    last["EMA20"]
                ),
                8
            ),

        "ema50":
            round(
                float(
                    last["EMA50"]
                ),
                8
            ),

        "ema200":
            round(
                float(
                    last["EMA200"]
                ),
                8
            ),

        "macd":
            round(
                macd,
                8
            ),

        "macd_signal":
            round(
                macd_signal,
                8
            ),

        "momentum_5m_percent":
            round(
                momentum_5,
                4
            ),

        "momentum_15m_percent":
            round(
                momentum_15,
                4
            ),

        "volume_ratio":
            round(
                volume_ratio,
                3
            ),

        "reasons":
            reasons
    }


# =========================================================
# 13. LIVE ANALYSIS
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "bot":
            "GM AI Trading Bot",

        "version":
            "v7 Full Chart Analysis",

        "message":
            "Bot is running.",

        "analysis":
            "/analysis/BTCUSDT",

        "backtest":
            "/backtest/BTCUSDT?limit=1000"

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
                DEFAULT_ACCOUNT_SIZE
            )
        )

        risk_percent = float(
            request.args.get(
                "risk_percent",
                DEFAULT_RISK_PERCENT
            )
        )

        # Main timeframe
        df = get_data(
            symbol,
            "5m",
            500
        )

        df = calculate_indicators(
            df
        )

        # MTF
        mtf = get_mtf_analysis(
            symbol
        )

        result = generate_analysis(
            df,
            symbol,
            mtf,
            account_size,
            risk_percent
        )

        result[
            "timeframe"
        ] = "5m"

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
            "Liquidation data is estimated. "
            "Do not risk money you cannot afford to lose."
        )

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                normalize_symbol(
                    symbol
                ),

            "message":
                str(e)

        }), 400


# =========================================================
# 14. REAL TP/SL BACKTEST
# =========================================================

def evaluate_trade(
    df,
    start_index,
    direction,
    entry,
    stop_loss,
    tp1,
    max_bars=30
):

    end_index = min(
        len(df),
        start_index +
        max_bars +
        1
    )

    for j in range(
        start_index + 1,
        end_index
    ):

        candle = df.iloc[j]

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        if direction == "LONG":

            sl_hit = (
                low <= stop_loss
            )

            tp_hit = (
                high >= tp1
            )

            # Conservative assumption:
            # if both happen in same candle,
            # assume SL first.
            if sl_hit and tp_hit:
                return "LOSS"

            if sl_hit:
                return "LOSS"

            if tp_hit:
                return "WIN"

        elif direction == "SHORT":

            sl_hit = (
                high >= stop_loss
            )

            tp_hit = (
                low <= tp1
            )

            if sl_hit and tp_hit:
                return "LOSS"

            if sl_hit:
                return "LOSS"

            if tp_hit:
                return "WIN"

    return "TIMEOUT"


@app.route("/backtest/<symbol>")
def backtest(symbol):

    try:

        symbol = normalize_symbol(
            symbol
        )

        limit = int(
            request.args.get(
                "limit",
                1000
            )
        )

        limit = max(
            400,
            min(
                limit,
                1000
            )
        )

        max_bars = int(
            request.args.get(
                "max_bars",
                30
            )
        )

        account_size = float(
            request.args.get(
                "account_size",
                DEFAULT_ACCOUNT_SIZE
            )
        )

        risk_percent = float(
            request.args.get(
                "risk_percent",
                DEFAULT_RISK_PERCENT
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

        wins = 0
        losses = 0
        timeouts = 0
        signals = 0
        no_trades = 0

        total_r = 0.0

        start = 220

        # Keep enough future candles
        last_index = (
            len(df) -
            max_bars -
            1
        )

        for i in range(
            start,
            last_index
        ):

            historical = df.iloc[
                :i + 1
            ].copy()

            # Historical analysis
            # MTF intentionally skipped in backtest
            # to avoid huge API calls.
            result = generate_analysis(
                historical,
                symbol,
                mtf=None,
                account_size=account_size,
                risk_percent=risk_percent
            )

            direction = result[
                "direction"
            ]

            if direction == "NO_TRADE":

                no_trades += 1

                continue

            entry = result[
                "entry"
            ]

            stop_loss = result[
                "stop_loss"
            ]

            tp1 = result[
                "take_profit_1"
            ]

            if (
                stop_loss is None
                or
                tp1 is None
            ):

                no_trades += 1

                continue

            signals += 1

            outcome = evaluate_trade(
                df,
                i,
                direction,
                entry,
                stop_loss,
                tp1,
                max_bars
            )

            if outcome == "WIN":

                wins += 1

                total_r += 1.5

            elif outcome == "LOSS":

                losses += 1

                total_r -= 1.0

            else:

                timeouts += 1

        if signals > 0:

            win_rate = (
                wins /
                signals
            ) * 100

        else:

            win_rate = 0

        profit_factor = (
            (
                wins * 1.5
            )
            /
            (
                losses * 1.0
            )
            if losses > 0
            else 0
        )

        return jsonify({

            "status":
                "success",

            "symbol":
                symbol,

            "timeframe":
                "5m",

            "candles_used":
                len(df),

            "trade_signals":
                signals,

            "no_trade_setups":
                no_trades,

            "wins":
                wins,

            "losses":
                losses,

            "timeouts":
                timeouts,

            "win_rate_percent":
                round(
                    win_rate,
                    2
                ),

            "profit_factor":
                round(
                    profit_factor,
                    2
                ),

            "total_R":
                round(
                    total_r,
                    2
                ),

            "risk_per_trade_percent":
                risk_percent,

            "tp1_r_multiple":
                1.5,

            "max_bars_per_trade":
                max_bars,

            "note":
                (
                    "Backtest uses historical "
                    "Entry -> TP1/SL logic. "
                    "If TP and SL occur in the "
                    "same candle, SL is assumed first "
                    "for conservative testing. "
                    "Fees, slippage and funding are "
                    "not fully modeled."
                )

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                normalize_symbol(
                    symbol
                ),

            "message":
                str(e)

        }), 400


# =========================================================
# 15. SERVER
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
