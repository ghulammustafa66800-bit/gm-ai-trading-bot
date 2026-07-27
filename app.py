from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

app = Flask(__name__)

MEXC_BASE = "https://api.mexc.com"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LIMIT = 1000
MIN_CANDLES = 250

# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_symbol(symbol):
    return (
        str(symbol)
        .upper()
        .strip()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except:
        return default


# =========================================================
# MEXC MARKET DATA
# =========================================================

def get_data(symbol, interval="5m", limit=1000):

    symbol = normalize_symbol(symbol)

    limit = max(
        MIN_CANDLES,
        min(int(limit), 1000)
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
    except Exception as e:
        raise Exception(
            f"MEXC connection error: {str(e)}"
        )

    if response.status_code != 200:
        raise Exception(
            f"MEXC API error {response.status_code}: "
            f"{response.text[:300]}"
        )

    data = response.json()

    if not isinstance(data, list):
        raise Exception(
            f"Unexpected MEXC response: {data}"
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

    if len(df) < MIN_CANDLES:
        raise Exception(
            "Not enough market data"
        )

    return df


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # MACD
    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

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
    hl = (
        df["high"] -
        df["low"]
    )

    hc = (
        df["high"] -
        df["close"].shift()
    ).abs()

    lc = (
        df["low"] -
        df["close"].shift()
    ).abs()

    tr = pd.concat(
        [
            hl,
            hc,
            lc
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        tr
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
    df["MOM_5"] = (
        df["close"]
        .pct_change(5)
        * 100
    )

    df["MOM_15"] = (
        df["close"]
        .pct_change(15)
        * 100
    )

    # Candle
    df["BODY"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["RANGE"] = (
        df["high"] -
        df["low"]
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

    return df


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def get_levels(df):

    # Exclude current candle
    # to avoid look-ahead bias

    history = df.iloc[
        :-1
    ].tail(120)

    support = float(
        history["low"].min()
    )

    resistance = float(
        history["high"].max()
    )

    last = df.iloc[-1]

    pivot = (
        last["high"] +
        last["low"] +
        last["close"]
    ) / 3

    return {
        "support": support,
        "resistance": resistance,
        "pivot": float(pivot)
    }


# =========================================================
# MARKET STRUCTURE
# =========================================================

def get_market_structure(df):

    data = df.tail(60)

    first = data.iloc[:30]

    second = data.iloc[30:]

    high1 = first["high"].max()
    high2 = second["high"].max()

    low1 = first["low"].min()
    low2 = second["low"].min()

    if (
        high2 > high1
        and
        low2 > low1
    ):

        return "BULLISH"

    if (
        high2 < high1
        and
        low2 < low1
    ):

        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# TREND
# =========================================================

def get_trend(df):

    last = df.iloc[-1]

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):

        return "STRONG_BULLISH"

    if (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):

        return "STRONG_BEARISH"

    if (
        last["EMA20"] >
        last["EMA50"]
    ):

        return "BULLISH"

    if (
        last["EMA20"] <
        last["EMA50"]
    ):

        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# LIQUIDITY SWEEP / TRAP
# =========================================================

def liquidity_analysis(
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

    volume_ratio = safe_float(
        last["VOLUME_RATIO"],
        1
    )

    bullish_sweep = (
        last["low"] <
        support
        and
        last["close"] >
        support
    )

    bearish_sweep = (
        last["high"] >
        resistance
        and
        last["close"] <
        resistance
    )

    if bullish_sweep:

        sweep = (
            "BULLISH_LIQUIDITY_SWEEP"
        )

    elif bearish_sweep:

        sweep = (
            "BEARISH_LIQUIDITY_SWEEP"
        )

    else:

        sweep = "NONE"

    if sweep == "NONE":

        trap = "LOW"

    elif volume_ratio < 1:

        trap = "HIGH"

    else:

        trap = "MEDIUM"

    return {
        "sweep": sweep,
        "trap_risk": trap
    }


# =========================================================
# CANDLE CONFIRMATION
# =========================================================

def candle_analysis(df):

    last = df.iloc[-1]

    body = safe_float(
        last["BODY"]
    )

    candle_range = safe_float(
        last["RANGE"]
    )

    if candle_range <= 0:

        return "NEUTRAL"

    bullish_rejection = (
        last["LOWER_WICK"] >
        body * 1.5
        and
        last["close"] >
        last["open"]
    )

    bearish_rejection = (
        last["UPPER_WICK"] >
        body * 1.5
        and
        last["close"] <
        last["open"]
    )

    strong_bull = (
        last["close"] >
        last["open"]
        and
        body /
        candle_range >
        0.60
    )

    strong_bear = (
        last["close"] <
        last["open"]
        and
        body /
        candle_range >
        0.60
    )

    if bullish_rejection:

        return "BULLISH_REJECTION"

    if bearish_rejection:

        return "BEARISH_REJECTION"

    if strong_bull:

        return "STRONG_BULLISH"

    if strong_bear:

        return "STRONG_BEARISH"

    return "NEUTRAL"


# =========================================================
# TIMEFRAME ANALYSIS
# =========================================================

def timeframe_direction(
    df
):

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

    return "NEUTRAL"


def multi_timeframe_analysis(
    symbol
):

    intervals = [
        "5m",
        "15m",
        "1h",
        "4h"
    ]

    result = {}

    for interval in intervals:

        try:

            data = get_data(
                symbol,
                interval,
                300
            )

            data = add_indicators(
                data
            )

            result[
                interval
            ] = timeframe_direction(
                data
            )

        except Exception as e:

            result[
                interval
            ] = "UNAVAILABLE"

    bullish = sum(
        1
        for value
        in result.values()
        if value == "BULLISH"
    )

    bearish = sum(
        1
        for value
        in result.values()
        if value == "BEARISH"
    )

    if bullish >= 3:

        confirmation = "BULLISH"

    elif bearish >= 3:

        confirmation = "BEARISH"

    else:

        confirmation = "MIXED"

    return {
        "timeframes": result,
        "confirmation": confirmation,
        "bullish_count": bullish,
        "bearish_count": bearish
    }


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(
    df,
    symbol,
    mtf,
    account_size,
    risk_percent
):

    last = df.iloc[-1]

    levels = get_levels(
        df
    )

    trend = get_trend(
        df
    )

    structure = get_market_structure(
        df
    )

    liquidity = liquidity_analysis(
        df,
        levels
    )

    candle = candle_analysis(
        df
    )

    price = safe_float(
        last["close"]
    )

    atr = safe_float(
        last["ATR"]
    )

    rsi = safe_float(
        last["RSI"],
        50
    )

    volume_ratio = safe_float(
        last["VOLUME_RATIO"],
        1
    )

    mom5 = safe_float(
        last["MOM_5"]
    )

    mom15 = safe_float(
        last["MOM_15"]
    )

    score = 0

    reasons = []

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if trend == "STRONG_BULLISH":

        score += 3

        reasons.append(
            "Strong bullish EMA trend"
        )

    elif trend == "STRONG_BEARISH":

        score -= 3

        reasons.append(
            "Strong bearish EMA trend"
        )

    elif trend == "BULLISH":

        score += 1

    elif trend == "BEARISH":

        score -= 1

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
    # RSI
    # -----------------------------------------------------

    if (
        50 <
        rsi <
        68
    ):

        score += 1

        reasons.append(
            "Healthy bullish RSI"
        )

    elif (
        32 <
        rsi <
        50
    ):

        score -= 1

        reasons.append(
            "Bearish RSI momentum"
        )

    # Avoid buying extreme overbought
    if rsi >= 75:

        score -= 2

        reasons.append(
            "Extreme RSI overbought"
        )

    if rsi <= 25:

        score += 2

        reasons.append(
            "Extreme RSI oversold"
        )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    if (
        mom5 > 0
        and
        mom15 > 0
    ):

        score += 1

        reasons.append(
            "Positive momentum"
        )

    elif (
        mom5 < 0
        and
        mom15 < 0
    ):

        score -= 1

        reasons.append(
            "Negative momentum"
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
    # CANDLE
    # -----------------------------------------------------

    if candle in [
        "BULLISH_REJECTION",
        "STRONG_BULLISH"
    ]:

        score += 1

        reasons.append(
            "Bullish candle confirmation"
        )

    elif candle in [
        "BEARISH_REJECTION",
        "STRONG_BEARISH"
    ]:

        score -= 1

        reasons.append(
            "Bearish candle confirmation"
        )

    # -----------------------------------------------------
    # MTF
    # -----------------------------------------------------

    if mtf["confirmation"] == "BULLISH":

        score += 2

        reasons.append(
            "Multi-timeframe bullish confirmation"
        )

    elif mtf["confirmation"] == "BEARISH":

        score -= 2

        reasons.append(
            "Multi-timeframe bearish confirmation"
        )

    # -----------------------------------------------------
    # LIQUIDITY SWEEP
    # -----------------------------------------------------

    if (
        liquidity["sweep"] ==
        "BULLISH_LIQUIDITY_SWEEP"
    ):

        score += 2

        reasons.append(
            "Bullish liquidity sweep"
        )

    elif (
        liquidity["sweep"] ==
        "BEARISH_LIQUIDITY_SWEEP"
    ):

        score -= 2

        reasons.append(
            "Bearish liquidity sweep"
        )

    # -----------------------------------------------------
    # FINAL DIRECTION
    # -----------------------------------------------------

    if score >= 7:

        direction = "LONG"

    elif score <= -7:

        direction = "SHORT"

    else:

        direction = "NO_TRADE"

    # Strong conflict filter

    if (
        direction == "LONG"
        and
        mtf["confirmation"] ==
        "BEARISH"
    ):

        direction = "NO_TRADE"

        reasons.append(
            "Long rejected by higher timeframe"
        )

    if (
        direction == "SHORT"
        and
        mtf["confirmation"] ==
        "BULLISH"
    ):

        direction = "NO_TRADE"

        reasons.append(
            "Short rejected by higher timeframe"
        )

    # High trap risk filter

    if (
        liquidity["trap_risk"] ==
        "HIGH"
    ):

        direction = "NO_TRADE"

        reasons.append(
            "Trade rejected because trap risk is high"
        )

    # -----------------------------------------------------
    # ENTRY / SL / TP
    # -----------------------------------------------------

    entry = price

    stop_loss = None

    tp1 = None

    tp2 = None

    tp3 = None

    risk_reward = 0

    if direction == "LONG":

        structural_sl = levels[
            "support"
        ]

        atr_sl = (
            entry -
            atr * 1.5
        )

        stop_loss = min(
            structural_sl,
            atr_sl
        )

        risk = (
            entry -
            stop_loss
        )

        if risk > 0:

            tp1 = (
                entry +
                risk * 1.5
            )

            tp2 = (
                entry +
                risk * 2
            )

            tp3 = (
                entry +
                risk * 3
            )

            risk_reward = 1.5

    elif direction == "SHORT":

        structural_sl = levels[
            "resistance"
        ]

        atr_sl = (
            entry +
            atr * 1.5
        )

        stop_loss = max(
            structural_sl,
            atr_sl
        )

        risk = (
            stop_loss -
            entry
        )

        if risk > 0:

            tp1 = (
                entry -
                risk * 1.5
            )

            tp2 = (
                entry -
                risk * 2
            )

            tp3 = (
                entry -
                risk * 3
            )

            risk_reward = 1.5

    # -----------------------------------------------------
    # MONEY MANAGEMENT
    # -----------------------------------------------------

    if stop_loss is not None:

        risk_amount = (
            account_size *
            risk_percent /
            100
        )

        stop_distance = abs(
            entry -
            stop_loss
        )

        position_size = (
            risk_amount /
            stop_distance
            if stop_distance > 0
            else 0
        )

        position_value = (
            position_size *
            entry
        )

    else:

        risk_amount = 0

        position_size = 0

        position_value = 0

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = min(
        95,
        50 +
        abs(score) * 4
    )

    if direction == "NO_TRADE":

        confidence = min(
            confidence,
            55
        )

    if direction == "LONG":

        signal = (
            "STRONG BUY"
            if score >= 10
            else "BUY"
        )

    elif direction == "SHORT":

        signal = (
            "STRONG SELL"
            if score <= -10
            else "SELL"
        )

    else:

        signal = "NO TRADE"

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

        "price":
            round(
                price,
                8
            ),

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
                    tp1,
                    8
                )
                if tp1
                is not None
                else None
            ),

        "take_profit_2":
            (
                round(
                    tp2,
                    8
                )
                if tp2
                is not None
                else None
            ),

        "take_profit_3":
            (
                round(
                    tp3,
                    8
                )
                if tp3
                is not None
                else None
            ),

        "risk_reward":
            risk_reward,

        "trend":
            trend,

        "market_structure":
            structure,

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

        "liquidity":
            liquidity,

        "candle":
            candle,

        "rsi":
            round(
                rsi,
                2
            ),

        "volume_ratio":
            round(
                volume_ratio,
                3
            ),

        "momentum_5m":
            round(
                mom5,
                4
            ),

        "momentum_15m":
            round(
                mom15,
                4
            ),

        "money_management":
            {
                "account_size":
                    account_size,

                "risk_percent":
                    risk_percent,

                "risk_amount":
                    round(
                        risk_amount,
                        2
                    ),

                "position_size":
                    round(
                        position_size,
                        8
                    ),

                "position_value":
                    round(
                        position_value,
                        2
                    )
            },

        "reasons":
            reasons
    }


# =========================================================
# LIVE ANALYSIS API
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "bot":
            "GM AI Trading Analysis Bot",

        "version":
            "FINAL V1",

        "message":
            "Bot is running",

        "example":
            "/analysis/BTCUSDT"

    })


@app.route(
    "/analysis/<symbol>"
)
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

        # Main chart
        df = get_data(
            symbol,
            "5m",
            1000
        )

        df = add_indicators(
            df
        )

        # Multi timeframe
        mtf = multi_timeframe_analysis(
            symbol
        )

        result = generate_signal(
            df,
            symbol,
            mtf,
            account_size,
            risk_percent
        )

        result[
            "timestamp"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        result[
            "warning"
        ] = (
            "Technical analysis only. "
            "No guaranteed prediction or "
            "guaranteed win rate."
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
# SIMPLE HEALTH CHECK
# =========================================================

@app.route(
    "/health"
)
def health():

    return jsonify({

        "status":
            "healthy",

        "time":
            datetime.now(
                timezone.utc
            ).isoformat()

    })


# =========================================================
# RUN
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
