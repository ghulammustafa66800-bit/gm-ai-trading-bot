from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

app = Flask(__name__)

# =========================================================
# GM SMART SCALPER V3
# VIDEO-INSPIRED SYSTEM
#
# FLOW:
# MARKET BIAS
# -> KEY LEVEL
# -> LIQUIDITY SWEEP / BREAKOUT
# -> RETEST
# -> VOLUME CONFIRMATION
# -> ENTRY
# -> SL
# -> TP1 / TP2 / TP3
#
# MEXC SPOT KLINE DATA
# =========================================================

MEXC_BASE = "https://api.mexc.com"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 1000

MIN_CANDLES = 250

# Signal quality
MIN_SCORE = 8

# Risk management
SL_ATR = 1.5
TP1_RR = 1.5
TP2_RR = 2.5
TP3_RR = 3.5

# Backtest
MAX_HOLD_CANDLES = 20


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

def get_data(
    symbol,
    interval="5m",
    limit=1000
):

    symbol = normalize_symbol(symbol)

    try:

        limit = int(limit)

    except:

        limit = 1000

    limit = max(
        MIN_CANDLES,
        min(limit, 1000)
    )

    url = (
        f"{MEXC_BASE}/api/v3/klines"
    )

    params = {

        "symbol":
            symbol,

        "interval":
            interval,

        "limit":
            limit
    }

    response = requests.get(

        url,

        params=params,

        timeout=20
    )

    if response.status_code != 200:

        raise Exception(

            f"MEXC API error "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    if not isinstance(data, list):

        raise Exception(

            f"Unexpected MEXC response: "
            f"{data}"
        )

    rows = []

    for candle in data:

        if len(candle) < 6:

            continue

        rows.append(

            [

                candle[0],

                candle[1],

                candle[2],

                candle[3],

                candle[4],

                candle[5]

            ]
        )

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

    df = (

        df

        .dropna()

        .reset_index(drop=True)
    )

    if len(df) < MIN_CANDLES:

        raise Exception(

            "Not enough candles "
            "returned by MEXC"
        )

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    high_low = (

        df["high"] -

        df["low"]
    )

    high_close = (

        df["high"] -

        df["close"].shift()

    ).abs()

    low_close = (

        df["low"] -

        df["close"].shift()

    ).abs()

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

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CANDLE STRUCTURE
    # -----------------------------------------------------

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
# MARKET BIAS
# =========================================================

def get_market_bias(df):

    last = df.iloc[-1]

    score = 0

    if (

        last["EMA20"]

        >

        last["EMA50"]

        >

        last["EMA200"]

    ):

        score += 3

    elif (

        last["EMA20"]

        <

        last["EMA50"]

        <

        last["EMA200"]

    ):

        score -= 3

    elif (

        last["EMA20"]

        >

        last["EMA50"]

    ):

        score += 1

    elif (

        last["EMA20"]

        <

        last["EMA50"]

    ):

        score -= 1

    if safe_float(
        last["MACD"]
    ) > safe_float(
        last["MACD_SIGNAL"]
    ):

        score += 1

    else:

        score -= 1

    if safe_float(
        last["MOM_15"]
    ) > 0:

        score += 1

    elif safe_float(
        last["MOM_15"]
    ) < 0:

        score -= 1

    if score >= 3:

        bias = "BULLISH"

    elif score <= -3:

        bias = "BEARISH"

    else:

        bias = "NEUTRAL"

    return {

        "bias":
            bias,

        "score":
            score
    }


# =========================================================
# MARKET STRUCTURE
# =========================================================

def get_market_structure(df):

    data = df.tail(60)

    first = data.iloc[:30]

    second = data.iloc[30:]

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
# KEY LEVELS
# =========================================================

def get_key_levels(df):

    # IMPORTANT:
    # Exclude current candle.
    # This prevents using current candle's future
    # information in historical backtesting.

    history = (

        df.iloc[:-1]

        .tail(100)
    )

    support = safe_float(

        history["low"].min()
    )

    resistance = safe_float(

        history["high"].max()
    )

    return {

        "support":
            support,

        "resistance":
            resistance
    }


# =========================================================
# LIQUIDITY SWEEP
# =========================================================

def detect_liquidity_sweep(
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

    bullish_sweep = (

        last["low"]

        <

        support

        and

        last["close"]

        >

        support

        and

        last["close"]

        >

        last["open"]
    )

    bearish_sweep = (

        last["high"]

        >

        resistance

        and

        last["close"]

        <

        resistance

        and

        last["close"]

        <

        last["open"]
    )

    if bullish_sweep:

        return "BULLISH_SWEEP"

    if bearish_sweep:

        return "BEARISH_SWEEP"

    return "NONE"


# =========================================================
# BREAKOUT DETECTION
# =========================================================

def detect_breakout(
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

    bullish_breakout = (

        last["close"]

        >

        resistance

        and

        last["open"]

        <=

        resistance
    )

    bearish_breakdown = (

        last["close"]

        <

        support

        and

        last["open"]

        >=

        support
    )

    if bullish_breakout:

        return "BULLISH_BREAKOUT"

    if bearish_breakdown:

        return "BEARISH_BREAKDOWN"

    return "NONE"


# =========================================================
# RETEST CONFIRMATION
# =========================================================

def detect_retest(
    df,
    levels
):

    if len(df) < 4:

        return "NONE"

    current = df.iloc[-1]

    previous = df.iloc[-2]

    support = levels[
        "support"
    ]

    resistance = levels[
        "resistance"
    ]

    # Bullish reclaim/retest
    bullish_retest = (

        previous["close"]

        >

        resistance

        and

        current["low"]

        <=

        resistance * 1.001

        and

        current["close"]

        >

        resistance
    )

    # Bearish retest
    bearish_retest = (

        previous["close"]

        <

        support

        and

        current["high"]

        >=

        support * 0.999

        and

        current["close"]

        <

        support
    )

    if bullish_retest:

        return "BULLISH_RETEST"

    if bearish_retest:

        return "BEARISH_RETEST"

    return "NONE"


# =========================================================
# VOLUME CONFIRMATION
# =========================================================

def volume_confirmation(df):

    last = df.iloc[-1]

    volume_ratio = safe_float(

        last["VOLUME_RATIO"],

        1
    )

    if volume_ratio >= 1.2:

        return {

            "confirmed":
                True,

            "ratio":
                round(
                    volume_ratio,
                    3
                )
        }

    return {

        "confirmed":
            False,

        "ratio":
            round(
                volume_ratio,
                3
            )
    }


# =========================================================
# CANDLE CONFIRMATION
# =========================================================

def candle_confirmation(df):

    last = df.iloc[-1]

    body = safe_float(
        last["BODY"]
    )

    candle_range = safe_float(
        last["RANGE"]
    )

    if candle_range <= 0:

        return "NONE"

    if (

        last["close"]

        >

        last["open"]

        and

        last["LOWER_WICK"]

        >

        body * 1.2

    ):

        return "BULLISH_REJECTION"

    if (

        last["close"]

        <

        last["open"]

        and

        last["UPPER_WICK"]

        >

        body * 1.2

    ):

        return "BEARISH_REJECTION"

    if (

        last["close"]

        >

        last["open"]

        and

        body / candle_range

        >

        0.60

    ):

        return "STRONG_BULLISH"

    if (

        last["close"]

        <

        last["open"]

        and

        body / candle_range

        >

        0.60

    ):

        return "STRONG_BEARISH"

    return "NONE"


# =========================================================
# HIGHER TIMEFRAME CONFIRMATION
# =========================================================

def timeframe_bias(
    df
):

    last = df.iloc[-1]

    if (

        last["EMA20"]

        >

        last["EMA50"]

        >

        last["EMA200"]

    ):

        return "BULLISH"

    if (

        last["EMA20"]

        <

        last["EMA50"]

        <

        last["EMA200"]

    ):

        return "BEARISH"

    return "NEUTRAL"


def get_mtf_confirmation(
    symbol
):

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

            data = add_indicators(
                data
            )

            results[
                interval
            ] = timeframe_bias(
                data
            )

        except Exception:

            results[
                interval
            ] = "UNAVAILABLE"

    bullish = list(

        results.values()

    ).count(

        "BULLISH"
    )

    bearish = list(

        results.values()

    ).count(

        "BEARISH"
    )

    if bullish >= 3:

        confirmation = "BULLISH"

    elif bearish >= 3:

        confirmation = "BEARISH"

    else:

        confirmation = "MIXED"

    return {

        "timeframes":
            results,

        "confirmation":
            confirmation,

        "bullish_count":
            bullish,

        "bearish_count":
            bearish
    }


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(
    df,
    symbol,
    mtf
):

    last = df.iloc[-1]

    bias = get_market_bias(
        df
    )

    structure = get_market_structure(
        df
    )

    levels = get_key_levels(
        df
    )

    sweep = detect_liquidity_sweep(

        df,

        levels
    )

    breakout = detect_breakout(

        df,

        levels
    )

    retest = detect_retest(

        df,

        levels
    )

    volume = volume_confirmation(
        df
    )

    candle = candle_confirmation(
        df
    )

    rsi = safe_float(

        last["RSI"],

        50
    )

    atr = safe_float(

        last["ATR"]
    )

    price = safe_float(

        last["close"]
    )

    score = 0

    reasons = []

    # -----------------------------------------------------
    # MARKET BIAS
    # -----------------------------------------------------

    if bias["bias"] == "BULLISH":

        score += 2

        reasons.append(

            "Market bias is bullish"
        )

    elif bias["bias"] == "BEARISH":

        score -= 2

        reasons.append(

            "Market bias is bearish"
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
    # LIQUIDITY
    # -----------------------------------------------------

    if sweep == "BULLISH_SWEEP":

        score += 3

        reasons.append(

            "Bullish liquidity sweep"
        )

    elif sweep == "BEARISH_SWEEP":

        score -= 3

        reasons.append(

            "Bearish liquidity sweep"
        )

    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    if breakout == "BULLISH_BREAKOUT":

        score += 3

        reasons.append(

            "Bullish key-level breakout"
        )

    elif breakout == "BEARISH_BREAKDOWN":

        score -= 3

        reasons.append(

            "Bearish key-level breakdown"
        )

    # -----------------------------------------------------
    # RETEST
    # -----------------------------------------------------

    if retest == "BULLISH_RETEST":

        score += 3

        reasons.append(

            "Bullish breakout retest confirmed"
        )

    elif retest == "BEARISH_RETEST":

        score -= 3

        reasons.append(

            "Bearish breakdown retest confirmed"
        )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if volume["confirmed"]:

        if score > 0:

            score += 2

            reasons.append(

                "Volume confirms bullish pressure"
            )

        elif score < 0:

            score -= 2

            reasons.append(

                "Volume confirms bearish pressure"
            )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 50 < rsi < 70:

        score += 1

        reasons.append(

            "RSI supports bullish momentum"
        )

    elif 30 < rsi < 50:

        score -= 1

        reasons.append(

            "RSI supports bearish momentum"
        )

    elif rsi >= 75:

        score -= 2

        reasons.append(

            "Overbought warning"
        )

    elif rsi <= 25:

        score += 2

        reasons.append(

            "Oversold recovery possibility"
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

        score += 3

        reasons.append(

            "Higher timeframes support LONG"
        )

    elif mtf["confirmation"] == "BEARISH":

        score -= 3

        reasons.append(

            "Higher timeframes support SHORT"
        )

    # =====================================================
    # FINAL DIRECTION
    # =====================================================

    direction = "NO_TRADE"

    # We require a stronger setup than simple indicator
    # agreement.

    if score >= MIN_SCORE:

        direction = "LONG"

    elif score <= -MIN_SCORE:

        direction = "SHORT"

    # Reject higher timeframe conflict

    if (

        direction == "LONG"

        and

        mtf["confirmation"] == "BEARISH"

    ):

        direction = "NO_TRADE"

        reasons.append(

            "LONG rejected by higher timeframe conflict"
        )

    if (

        direction == "SHORT"

        and

        mtf["confirmation"] == "BULLISH"

    ):

        direction = "NO_TRADE"

        reasons.append(

            "SHORT rejected by higher timeframe conflict"
        )

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    entry = price

    stop_loss = None

    tp1 = None

    tp2 = None

    tp3 = None

    risk_reward = 0

    if direction == "LONG":

        atr_stop = (

            entry -

            atr *

            SL_ATR
        )

        # Put SL below key support or ATR distance

        stop_loss = min(

            atr_stop,

            levels["support"]
        )

        risk = (

            entry -

            stop_loss
        )

        if risk > 0:

            tp1 = (

                entry +

                risk *

                TP1_RR
            )

            tp2 = (

                entry +

                risk *

                TP2_RR
            )

            tp3 = (

                entry +

                risk *

                TP3_RR
            )

            risk_reward = TP1_RR

    elif direction == "SHORT":

        atr_stop = (

            entry +

            atr *

            SL_ATR
        )

        stop_loss = max(

            atr_stop,

            levels["resistance"]
        )

        risk = (

            stop_loss -

            entry
        )

        if risk > 0:

            tp1 = (

                entry -

                risk *

                TP1_RR
            )

            tp2 = (

                entry -

                risk *

                TP2_RR
            )

            tp3 = (

                entry -

                risk *

                TP3_RR
            )

            risk_reward = TP1_RR

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = min(

        95,

        50 +

        abs(score) * 3
    )

    if direction == "NO_TRADE":

        confidence = min(

            confidence,

            55
        )

    # =====================================================
    # SIGNAL NAME
    # =====================================================

    if direction == "LONG":

        signal = (

            "STRONG BUY"

            if score >= 12

            else

            "BUY"
        )

    elif direction == "SHORT":

        signal = (

            "STRONG SELL"

            if score <= -12

            else

            "SELL"
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

        "current_price":
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

            risk_reward,

        "market_bias":
            bias,

        "market_structure":
            structure,

        "liquidity_sweep":
            sweep,

        "breakout":
            breakout,

        "retest":
            retest,

        "volume":
            volume,

        "candle":
            candle,

        "rsi":
            round(

                rsi,

                2
            ),

        "atr":
            round(

                atr,

                8
            ),

        "support":
            round(

                levels["support"],

                8
            ),

        "resistance":
            round(

                levels["resistance"],

                8
            ),

        "multi_timeframe":
            mtf,

        "reasons":
            reasons
    }


# =========================================================
# MONEY MANAGEMENT
# =========================================================

def calculate_position_size(

    entry,

    stop_loss,

    account_size,

    risk_percent

):

    if stop_loss is None:

        return {

            "risk_amount":
                0,

            "position_size":
                0,

            "position_value":
                0
        }

    risk_amount = (

        account_size *

        risk_percent /

        100
    )

    distance = abs(

        entry -

        stop_loss
    )

    if distance <= 0:

        return {

            "risk_amount":
                0,

            "position_size":
                0,

            "position_value":
                0
        }

    position_size = (

        risk_amount /

        distance
    )

    position_value = (

        position_size *

        entry
    )

    return {

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
    }


# =========================================================
# LIVE ANALYSIS
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "bot":
            "GM Smart Scalper V3",

        "version":
            "3.0",

        "message":
            "Video-inspired smart trading analysis bot is running.",

        "endpoints":

            [

                "/health",

                "/analysis/BTCUSDT",

                "/backtest/BTCUSDT"

            ]

    })


@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "bot":
            "GM Smart Scalper V3",

        "timestamp":

            datetime.now(

                timezone.utc

            ).isoformat()

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

        df = get_data(

            symbol,

            "5m",

            1000
        )

        df = add_indicators(
            df
        )

        mtf = get_mtf_confirmation(
            symbol
        )

        result = generate_signal(

            df,

            symbol,

            mtf
        )

        money = calculate_position_size(

            result["entry"],

            result["stop_loss"],

            account_size,

            risk_percent
        )

        result[
            "money_management"
        ] = {

            "account_size":
                account_size,

            "risk_percent":
                risk_percent,

            **money
        }

        result[
            "timestamp"
        ] = (

            datetime.now(

                timezone.utc

            ).isoformat()
        )

        result[
            "warning"
        ] = (

            "Technical analysis only. "
            "No guaranteed win rate or profit. "
            "This bot uses MEXC spot candle data. "
            "Paper trade before using real money."
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
# BACKTEST
# =========================================================

def run_backtest(

    df,

    symbol,

    fee_percent=0.1,

    slippage_percent=0.02

):

    df = add_indicators(
        df
    )

    wins = 0

    losses = 0

    total_trades = 0

    skipped = 0

    net_return = 0

    trades = []

    # Start after enough candles
    start = 220

    for i in range(

        start,

        len(df) -

        MAX_HOLD_CANDLES -

        1

    ):

        # Historical data only
        historical = (

            df.iloc[

                :i + 1

            ]

            .copy()
        )

        # We don't call live MTF here because that would
        # make the backtest extremely slow and could mix
        # current market data.
        #
        # Instead we use historical 5m bias.

        fake_mtf = {

            "confirmation":
                get_market_bias(

                    historical

                )["bias"],

            "timeframes":
                {},

            "bullish_count":
                0,

            "bearish_count":
                0

        }

        result = generate_signal(

            historical,

            symbol,

            fake_mtf
        )

        direction = result[
            "direction"
        ]

        if direction == "NO_TRADE":

            skipped += 1

            continue

        entry = safe_float(

            result["entry"]
        )

        sl = result[
            "stop_loss"
        ]

        tp = result[
            "take_profit_1"
        ]

        if (

            sl is None

            or

            tp is None

        ):

            skipped += 1

            continue

        total_trades += 1

        outcome = "OPEN"

        exit_price = None

        exit_index = None

        for j in range(

            i + 1,

            min(

                i +

                1 +

                MAX_HOLD_CANDLES,

                len(df)
            )

        ):

            high = safe_float(

                df.iloc[j][
                    "high"
                ]
            )

            low = safe_float(

                df.iloc[j][
                    "low"
                ]
            )

            # Conservative:
            # SL first if both touched
            # in the same candle.

            if direction == "LONG":

                if low <= sl:

                    outcome = "LOSS"

                    exit_price = sl

                    exit_index = j

                    break

                if high >= tp:

                    outcome = "WIN"

                    exit_price = tp

                    exit_index = j

                    break

            elif direction == "SHORT":

                if high >= sl:

                    outcome = "LOSS"

                    exit_price = sl

                    exit_index = j

                    break

                if low <= tp:

                    outcome = "WIN"

                    exit_price = tp

                    exit_index = j

                    break

        if outcome == "OPEN":

            # If neither TP nor SL hit,
            # use final candle close as exit.

            exit_index = min(

                i +

                MAX_HOLD_CANDLES,

                len(df) - 1
            )

            exit_price = safe_float(

                df.iloc[
                    exit_index
                ][
                    "close"
                ]
            )

            if direction == "LONG":

                gross = (

                    exit_price -

                    entry

                ) / entry * 100

            else:

                gross = (

                    entry -

                    exit_price

                ) / entry * 100

            total_cost = (

                fee_percent * 2

                +

                slippage_percent
            )

            net = (

                gross -

                total_cost
            )

            if net > 0:

                outcome = "WIN"

                wins += 1

            else:

                outcome = "LOSS"

                losses += 1

            net_return += net

        else:

            gross_distance = (

                abs(

                    exit_price -

                    entry

                ) / entry

            ) * 100

            total_cost = (

                fee_percent * 2

                +

                slippage_percent
            )

            if outcome == "WIN":

                net = (

                    gross_distance -

                    total_cost
                )

                wins += 1

            else:

                net = (

                    -gross_distance -

                    total_cost
                )

                losses += 1

            net_return += net

        trades.append({

            "direction":
                direction,

            "entry":
                round(

                    entry,

                    8
                ),

            "exit":
                round(

                    exit_price,

                    8
                ),

            "result":
                outcome,

            "net_return_percent":
                round(

                    net,

                    4
                )
        })

    if total_trades > 0:

        win_rate = (

            wins /

            total_trades

        ) * 100

    else:

        win_rate = 0

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "strategy":
            "GM Smart Scalper V3",

        "candles_tested":
            len(df) - start,

        "total_trades":
            total_trades,

        "wins":
            wins,

        "losses":
            losses,

        "skipped_no_trade":
            skipped,

        "win_rate_percent":
            round(

                win_rate,

                2
            ),

        "net_return_percent":
            round(

                net_return,

                4
            ),

        "fee_percent_per_side":
            fee_percent,

        "slippage_percent":
            slippage_percent,

        "max_hold_candles":
            MAX_HOLD_CANDLES,

        "note":

            (

                "Historical backtest. "
                "TP1 and SL are checked on future "
                "candles. If both TP and SL are "
                "touched in one candle, SL is counted "
                "first as a conservative assumption. "
                "Past performance does not guarantee "
                "future results."
            ),

        "recent_trades":
            trades[-20:]
    }


@app.route(
    "/backtest/<symbol>"
)
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

        fee = float(

            request.args.get(

                "fee",

                0.1
            )
        )

        slippage = float(

            request.args.get(

                "slippage",

                0.02
            )
        )

        df = get_data(

            symbol,

            "5m",

            limit
        )

        result = run_backtest(

            df,

            symbol,

            fee,

            slippage
        )

        result[
            "timestamp"
        ] = (

            datetime.now(

                timezone.utc

            ).isoformat()
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
# RUN SERVER
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
