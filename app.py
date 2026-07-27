from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

app = Flask(__name__)

MEXC_BASE = "https://api.mexc.com"

# =========================================================
# SETTINGS
# =========================================================

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LIMIT = 1000
MIN_CANDLES = 250

# Signal threshold
SIGNAL_THRESHOLD = 7

# ATR settings
SL_ATR_MULTIPLIER = 1.5
TP1_RR = 1.5
TP2_RR = 2.0
TP3_RR = 3.0


# =========================================================
# HELPERS
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
# MEXC DATA
# =========================================================

def get_data(
    symbol,
    interval="5m",
    limit=1000
):

    symbol = normalize_symbol(symbol)

    limit = max(
        MIN_CANDLES,
        min(int(limit), 1000)
    )

    url = (
        f"{MEXC_BASE}/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
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
            "Not enough candles"
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
# MARKET STRUCTURE
# =========================================================

def market_structure(df):

    data = df.tail(60)

    first = data.iloc[:30]

    second = data.iloc[30:]

    h1 = first["high"].max()
    h2 = second["high"].max()

    l1 = first["low"].min()
    l2 = second["low"].min()

    if h2 > h1 and l2 > l1:

        return "BULLISH"

    if h2 < h1 and l2 < l1:

        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def get_levels(df):

    history = (
        df.iloc[:-1]
        .tail(120)
    )

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

    if last["EMA20"] > last["EMA50"]:

        return "BULLISH"

    if last["EMA20"] < last["EMA50"]:

        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# LIQUIDITY / TRAP
# =========================================================

def liquidity_analysis(
    df,
    levels
):

    last = df.iloc[-1]

    support = levels["support"]

    resistance = levels["resistance"]

    volume_ratio = safe_float(
        last["VOLUME_RATIO"],
        1
    )

    bullish_sweep = (
        last["low"] < support
        and
        last["close"] > support
    )

    bearish_sweep = (
        last["high"] > resistance
        and
        last["close"] < resistance
    )

    if bullish_sweep:

        sweep = "BULLISH_SWEEP"

    elif bearish_sweep:

        sweep = "BEARISH_SWEEP"

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
# CANDLE
# =========================================================

def candle_signal(df):

    last = df.iloc[-1]

    body = safe_float(
        last["BODY"]
    )

    candle_range = safe_float(
        last["RANGE"]
    )

    if candle_range <= 0:

        return "NEUTRAL"

    if (
        last["LOWER_WICK"] >
        body * 1.5
        and
        last["close"] >
        last["open"]
    ):

        return "BULLISH_REJECTION"

    if (
        last["UPPER_WICK"] >
        body * 1.5
        and
        last["close"] <
        last["open"]
    ):

        return "BEARISH_REJECTION"

    if (
        last["close"] >
        last["open"]
        and
        body / candle_range >
        0.60
    ):

        return "STRONG_BULLISH"

    if (
        last["close"] <
        last["open"]
        and
        body / candle_range >
        0.60
    ):

        return "STRONG_BEARISH"

    return "NEUTRAL"


# =========================================================
# MTF
# =========================================================

def timeframe_direction(df):

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


def get_mtf(symbol):

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
            ] = timeframe_direction(
                data
            )

        except:

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
        "timeframes": results,
        "confirmation": confirmation,
        "bullish_count": bullish,
        "bearish_count": bearish
    }


# =========================================================
# SIGNAL ENGINE
# =========================================================

def analyze_market(
    df,
    symbol,
    mtf,
    account_size=1000,
    risk_percent=1
):

    last = df.iloc[-1]

    levels = get_levels(
        df
    )

    trend = get_trend(
        df
    )

    structure = market_structure(
        df
    )

    liquidity = liquidity_analysis(
        df,
        levels
    )

    candle = candle_signal(
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

    volume = safe_float(
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

    # -------------------------
    # TREND
    # -------------------------

    if trend == "STRONG_BULLISH":

        score += 3

        reasons.append(
            "Strong bullish EMA alignment"
        )

    elif trend == "STRONG_BEARISH":

        score -= 3

        reasons.append(
            "Strong bearish EMA alignment"
        )

    elif trend == "BULLISH":

        score += 1

    elif trend == "BEARISH":

        score -= 1

    # -------------------------
    # STRUCTURE
    # -------------------------

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

    # -------------------------
    # RSI
    # -------------------------

    if 50 < rsi < 68:

        score += 1

        reasons.append(
            "Healthy bullish RSI"
        )

    elif 32 < rsi < 50:

        score -= 1

        reasons.append(
            "Bearish RSI"
        )

    elif rsi >= 75:

        score -= 2

        reasons.append(
            "Extreme overbought"
        )

    elif rsi <= 25:

        score += 2

        reasons.append(
            "Extreme oversold"
        )

    # -------------------------
    # MACD
    # -------------------------

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

    # -------------------------
    # MOMENTUM
    # -------------------------

    if mom5 > 0 and mom15 > 0:

        score += 1

        reasons.append(
            "Positive momentum"
        )

    elif mom5 < 0 and mom15 < 0:

        score -= 1

        reasons.append(
            "Negative momentum"
        )

    # -------------------------
    # VOLUME
    # -------------------------

    if volume >= 1.2:

        if score > 0:

            score += 1

        elif score < 0:

            score -= 1

        reasons.append(
            "Volume confirmation"
        )

    # -------------------------
    # CANDLE
    # -------------------------

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

    # -------------------------
    # MTF
    # -------------------------

    if mtf["confirmation"] == "BULLISH":

        score += 2

        reasons.append(
            "Higher timeframe bullish"
        )

    elif mtf["confirmation"] == "BEARISH":

        score -= 2

        reasons.append(
            "Higher timeframe bearish"
        )

    # -------------------------
    # LIQUIDITY
    # -------------------------

    if (
        liquidity["sweep"]
        ==
        "BULLISH_SWEEP"
    ):

        score += 2

        reasons.append(
            "Bullish liquidity sweep"
        )

    elif (
        liquidity["sweep"]
        ==
        "BEARISH_SWEEP"
    ):

        score -= 2

        reasons.append(
            "Bearish liquidity sweep"
        )

    # -------------------------
    # DIRECTION
    # -------------------------

    if score >= SIGNAL_THRESHOLD:

        direction = "LONG"

    elif score <= -SIGNAL_THRESHOLD:

        direction = "SHORT"

    else:

        direction = "NO_TRADE"

    # Higher timeframe conflict
    if (
        direction == "LONG"
        and
        mtf["confirmation"] ==
        "BEARISH"
    ):

        direction = "NO_TRADE"

        reasons.append(
            "LONG rejected by higher timeframe"
        )

    if (
        direction == "SHORT"
        and
        mtf["confirmation"] ==
        "BULLISH"
    ):

        direction = "NO_TRADE"

        reasons.append(
            "SHORT rejected by higher timeframe"
        )

    # Trap filter
    if (
        liquidity["trap_risk"]
        ==
        "HIGH"
    ):

        direction = "NO_TRADE"

        reasons.append(
            "Trade rejected due to high trap risk"
        )

    # -------------------------
    # ENTRY SL TP
    # -------------------------

    entry = price

    stop_loss = None

    tp1 = None

    tp2 = None

    tp3 = None

    risk_reward = 0

    if direction == "LONG":

        atr_sl = (
            entry -
            atr *
            SL_ATR_MULTIPLIER
        )

        stop_loss = min(
            levels["support"],
            atr_sl
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

        atr_sl = (
            entry +
            atr *
            SL_ATR_MULTIPLIER
        )

        stop_loss = max(
            levels["resistance"],
            atr_sl
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

    # -------------------------
    # MONEY MANAGEMENT
    # -------------------------

    if stop_loss is not None:

        risk_amount = (
            account_size *
            risk_percent /
            100
        )

        distance = abs(
            entry -
            stop_loss
        )

        position_size = (
            risk_amount /
            distance
            if distance > 0
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

    # -------------------------
    # SIGNAL NAME
    # -------------------------

    if direction == "LONG":

        signal = (
            "STRONG BUY"
            if score >= 10
            else
            "BUY"
        )

    elif direction == "SHORT":

        signal = (
            "STRONG SELL"
            if score <= -10
            else
            "SELL"
        )

    else:

        signal = "NO TRADE"

    confidence = min(
        95,
        50 +
        abs(score) *
        4
    )

    if direction == "NO_TRADE":

        confidence = min(
            confidence,
            55
        )

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
                levels["support"],
                8
            ),

        "resistance":
            round(
                levels["resistance"],
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
                volume,
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
# BACKTEST ENGINE
# =========================================================

def backtest_strategy(
    df,
    symbol,
    fee_percent=0.1,
    slippage_percent=0.02
):

    df = add_indicators(
        df
    )

    trades = []

    wins = 0

    losses = 0

    skipped = 0

    start = 220

    # We use historical candles only.
    # No future data is used to create the signal.

    for i in range(
        start,
        len(df) - 20
    ):

        historical = (
            df.iloc[
                :i + 1
            ]
            .copy()
        )

        last = historical.iloc[-1]

        # Simple historical signal
        # using current historical information

        trend = get_trend(
            historical
        )

        structure = market_structure(
            historical
        )

        rsi = safe_float(
            last["RSI"],
            50
        )

        macd_bull = (
            last["MACD"] >
            last["MACD_SIGNAL"]
        )

        mom5 = safe_float(
            last["MOM_5"]
        )

        mom15 = safe_float(
            last["MOM_15"]
        )

        score = 0

        if trend == "STRONG_BULLISH":
            score += 3

        elif trend == "STRONG_BEARISH":
            score -= 3

        elif trend == "BULLISH":
            score += 1

        elif trend == "BEARISH":
            score -= 1

        if structure == "BULLISH":
            score += 2

        elif structure == "BEARISH":
            score -= 2

        if 50 < rsi < 68:
            score += 1

        elif 32 < rsi < 50:
            score -= 1

        if rsi >= 75:
            score -= 2

        elif rsi <= 25:
            score += 2

        if macd_bull:
            score += 2

        else:
            score -= 2

        if mom5 > 0 and mom15 > 0:
            score += 1

        elif mom5 < 0 and mom15 < 0:
            score -= 1

        if score >= SIGNAL_THRESHOLD:

            direction = "LONG"

        elif score <= -SIGNAL_THRESHOLD:

            direction = "SHORT"

        else:

            skipped += 1

            continue

        entry = safe_float(
            df.iloc[i]["close"]
        )

        atr = safe_float(
            df.iloc[i]["ATR"]
        )

        if atr <= 0:

            continue

        if direction == "LONG":

            sl = (
                entry -
                atr *
                SL_ATR_MULTIPLIER
            )

            tp = (
                entry +
                (
                    entry -
                    sl
                ) *
                TP1_RR
            )

        else:

            sl = (
                entry +
                atr *
                SL_ATR_MULTIPLIER
            )

            tp = (
                entry -
                (
                    sl -
                    entry
                ) *
                TP1_RR
            )

        result = "OPEN"

        exit_price = None

        exit_index = None

        # Check future candles
        for j in range(
            i + 1,
            min(
                i + 21,
                len(df)
            )
        ):

            high = safe_float(
                df.iloc[j]["high"]
            )

            low = safe_float(
                df.iloc[j]["low"]
            )

            if direction == "LONG":

                hit_sl = (
                    low <= sl
                )

                hit_tp = (
                    high >= tp
                )

                # Conservative rule:
                # if both hit same candle,
                # count SL first.

                if hit_sl:

                    result = "LOSS"

                    exit_price = sl

                    exit_index = j

                    break

                if hit_tp:

                    result = "WIN"

                    exit_price = tp

                    exit_index = j

                    break

            else:

                hit_sl = (
                    high >= sl
                )

                hit_tp = (
                    low <= tp
                )

                if hit_sl:

                    result = "LOSS"

                    exit_price = sl

                    exit_index = j

                    break

                if hit_tp:

                    result = "WIN"

                    exit_price = tp

                    exit_index = j

                    break

        if result == "OPEN":

            continue

        # Approximate trading costs
        gross_return = abs(
            exit_price -
            entry
        ) / entry * 100

        total_cost = (
            fee_percent * 2
            +
            slippage_percent
        )

        if result == "WIN":

            net_return = (
                gross_return -
                total_cost
            )

            wins += 1

        else:

            net_return = (
                -gross_return -
                total_cost
            )

            losses += 1

        trades.append(
            {
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
                    result,

                "net_return_percent":
                    round(
                        net_return,
                        4
                    )
            }
        )

    total = wins + losses

    if total > 0:

        win_rate = (
            wins /
            total
        ) * 100

    else:

        win_rate = 0

    net_profit = sum(
        t[
            "net_return_percent"
        ]
        for t in trades
    )

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "candles_tested":
            len(df) - start,

        "total_trades":
            total,

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
                net_profit,
                4
            ),

        "fee_percent_per_side":
            fee_percent,

        "slippage_percent":
            slippage_percent,

        "note":
            (
                "Backtest checks whether "
                "TP1 or SL was reached "
                "within the next 20 candles. "
                "If both TP and SL occur in "
                "the same candle, SL is counted "
                "first as a conservative assumption."
            ),

        "recent_trades":
            trades[-20:]
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "bot":
            "GM AI Trading Bot",

        "version":
            "Professional V2",

        "message":
            "Trading analysis bot is running.",

        "endpoints":
            [
                "/health",
                "/analysis/BTCUSDT",
                "/backtest/BTCUSDT"
            ]

    })


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/health"
)
def health():

    return jsonify({

        "status":
            "healthy",

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat()

    })


# =========================================================
# LIVE ANALYSIS
# =========================================================

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

        mtf = get_mtf(
            symbol
        )

        result = analyze_market(
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
            "No guaranteed profit or "
            "win rate. Test with paper "
            "trading before risking money."
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

        result = backtest_strategy(
            df,
            symbol,
            fee,
            slippage
        )

        result[
            "timestamp"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

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
# SERVER
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
