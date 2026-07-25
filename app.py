from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
import math
import time

app = Flask(__name__)

# =========================================================
# CONFIG
# =========================================================

MEXC_URL = "https://api.mexc.com/api/v3/klines"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 500

# Risk management
DEFAULT_ACCOUNT_SIZE = 1000.0
DEFAULT_RISK_PERCENT = 1.0

# Supported MEXC intervals
VALID_INTERVALS = {
    "1m",
    "5m",
    "15m",
    "30m",
    "60m",
    "4h",
    "1d",
    "1W",
    "1M"
}

# User-friendly aliases
INTERVAL_ALIASES = {
    "1h": "60m",
    "60min": "60m",
    "1hour": "60m",
    "4hour": "4h",
    "1day": "1d",
    "daily": "1d"
}


# =========================================================
# 1. INTERVAL VALIDATION
# =========================================================

def normalize_interval(interval):

    interval = str(interval).strip()

    interval = INTERVAL_ALIASES.get(
        interval.lower(),
        interval
    )

    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Use: {sorted(list(VALID_INTERVALS))}"
        )

    return interval


# =========================================================
# 2. GET MARKET DATA FROM MEXC
# =========================================================

def get_data(
    symbol=DEFAULT_SYMBOL,
    interval=DEFAULT_INTERVAL,
    limit=500
):

    symbol = symbol.upper().strip()

    interval = normalize_interval(interval)

    limit = int(limit)

    limit = max(100, min(limit, 1000))

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:

        response = requests.get(
            MEXC_URL,
            params=params,
            timeout=20
        )

        if response.status_code != 200:

            raise Exception(
                f"MEXC API Error {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()

    except requests.RequestException as e:

        raise Exception(
            f"Market data connection error: {str(e)}"
        )

    if not isinstance(data, list):

        raise Exception(
            f"Unexpected MEXC response: {data}"
        )

    if len(data) < 100:

        raise Exception(
            "Not enough market data received."
        )

    # MEXC response usually contains:
    # time, open, high, low, close, volume,
    # close_time, quote_volume

    df = pd.DataFrame(
        data,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume"
        ]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    for col in numeric_columns:

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
    # RSI - Wilder style
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

    # -----------------------------------------------------
    # Volume
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
    # Momentum
    # -----------------------------------------------------

    df["MOMENTUM"] = (
        df["close"]
        .pct_change(5) *
        100
    )

    # -----------------------------------------------------
    # Trendline slope
    # -----------------------------------------------------

    window = 30

    def calculate_slope(series):

        if len(series) < 5:

            return 0.0

        x = np.arange(
            len(series)
        )

        y = np.array(
            series
        )

        try:

            slope = np.polyfit(
                x,
                y,
                1
            )[0]

            return float(
                slope
            )

        except:

            return 0.0

    df["TRENDLINE_SLOPE"] = (
        df["close"]
        .rolling(window)
        .apply(
            calculate_slope,
            raw=False
        )
    )

    return df


# =========================================================
# 4. SUPPORT / RESISTANCE
# =========================================================

def support_resistance(df):

    recent = df.tail(50)

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    return support, resistance


# =========================================================
# 5. TRENDLINE ANALYSIS
# =========================================================

def trendline_analysis(df):

    recent = df.tail(30)

    x = np.arange(
        len(recent)
    )

    close_values = (
        recent["close"]
        .values
    )

    try:

        slope = np.polyfit(
            x,
            close_values,
            1
        )[0]

    except:

        slope = 0

    current_price = float(
        df.iloc[-1]["close"]
    )

    average_price = float(
        recent["close"].mean()
    )

    # Normalize slope
    slope_percent = (
        slope /
        average_price
    ) * 100

    if slope_percent > 0.03:

        trendline = "UPTREND"

    elif slope_percent < -0.03:

        trendline = "DOWNTREND"

    else:

        trendline = "FLAT"

    return {
        "trendline": trendline,
        "slope_percent": round(
            float(
                slope_percent
            ),
            4
        )
    }


# =========================================================
# 6. MARKET STRUCTURE
# =========================================================

def market_structure(df):

    recent = df.tail(20)

    first_half = recent.iloc[:10]

    second_half = recent.iloc[10:]

    first_high = float(
        first_half["high"].max()
    )

    second_high = float(
        second_half["high"].max()
    )

    first_low = float(
        first_half["low"].min()
    )

    second_low = float(
        second_half["low"].min()
    )

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
# 7. BREAKOUT / BREAKDOWN
# =========================================================

def breakout_analysis(
    df,
    support,
    resistance
):

    last = df.iloc[-1]

    previous = df.iloc[-2]

    if (
        last["close"] > resistance
        and
        previous["close"] <= resistance
    ):

        return "BREAKOUT"

    if (
        last["close"] < support
        and
        previous["close"] >= support
    ):

        return "BREAKDOWN"

    return "NO_BREAKOUT"


# =========================================================
# 8. GAP UP / GAP DOWN
# =========================================================

def gap_analysis(df):

    last = df.iloc[-1]

    previous = df.iloc[-2]

    if previous["close"] == 0:

        return "NO_GAP"

    gap_percent = (
        (
            last["open"] -
            previous["close"]
        )
        /
        previous["close"]
    ) * 100

    if gap_percent > 0.30:

        return "GAP_UP"

    if gap_percent < -0.30:

        return "GAP_DOWN"

    return "NO_GAP"


# =========================================================
# 9. TRAP / FAKE BREAKOUT
# =========================================================

def trap_analysis(
    df,
    support,
    resistance
):

    last = df.iloc[-1]

    volume_confirmed = (
        last["volume"] >
        last["VOL_AVG"]
    )

    fake_breakout = (
        last["high"] > resistance
        and
        last["close"] < resistance
    )

    fake_breakdown = (
        last["low"] < support
        and
        last["close"] > support
    )

    if fake_breakout:

        if volume_confirmed:

            return "MEDIUM"

        return "HIGH"

    if fake_breakdown:

        if volume_confirmed:

            return "MEDIUM"

        return "HIGH"

    return "LOW"


# =========================================================
# 10. ESTIMATED LIQUIDATION RISK
# =========================================================

def liquidation_risk_analysis(
    df,
    signal,
    atr
):

    last = df.iloc[-1]

    price = float(
        last["close"]
    )

    volume_ratio = float(
        last["VOLUME_RATIO"]
        if pd.notna(
            last["VOLUME_RATIO"]
        )
        else 1
    )

    momentum = float(
        last["MOMENTUM"]
        if pd.notna(
            last["MOMENTUM"]
        )
        else 0
    )

    atr_percent = (
        atr /
        price
    ) * 100

    risk_score = 0

    # High volatility
    if atr_percent > 1.5:

        risk_score += 3

    elif atr_percent > 0.8:

        risk_score += 2

    # Extreme momentum
    if abs(momentum) > 2:

        risk_score += 2

    elif abs(momentum) > 1:

        risk_score += 1

    # Low volume during strong signal
    if (
        abs(momentum) > 1
        and
        volume_ratio < 0.8
    ):

        risk_score += 2

    if risk_score >= 5:

        risk = "HIGH"

    elif risk_score >= 3:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "risk": risk,
        "risk_score": risk_score,
        "atr_percent": round(
            atr_percent,
            4
        ),
        "note":
            "Estimated liquidation risk. "
            "Direct liquidation data is not "
            "available from the public candle API."
    }


# =========================================================
# 11. MULTI-TIMEFRAME CONFIRMATION
# =========================================================

def get_timeframe_trend(
    symbol,
    interval
):

    try:

        df = get_data(
            symbol,
            interval,
            300
        )

        df = calculate_indicators(
            df
        )

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

        return "SIDEWAYS"

    except Exception:

        return "UNKNOWN"


def multi_timeframe_analysis(
    symbol
):

    # IMPORTANT:
    # MEXC uses 60m, not 1h

    return {

        "5m":
            get_timeframe_trend(
                symbol,
                "5m"
            ),

        "15m":
            get_timeframe_trend(
                symbol,
                "15m"
            ),

        "1h":
            get_timeframe_trend(
                symbol,
                "60m"
            )
    }


# =========================================================
# 12. RISK MANAGEMENT
# =========================================================

def risk_management(
    price,
    atr,
    signal,
    account_size,
    risk_percent
):

    account_size = float(
        account_size
    )

    risk_percent = float(
        risk_percent
    )

    risk_amount = (
        account_size *
        risk_percent /
        100
    )

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

        return {

            "entry": price,

            "stop_loss": None,

            "take_profit_1": None,

            "take_profit_2": None,

            "risk_amount": 0,

            "position_size": 0,

            "risk_reward": 0
        }

    stop_distance = abs(
        entry -
        stop_loss
    )

    if stop_distance <= 0:

        position_size = 0

    else:

        position_size = (
            risk_amount /
            stop_distance
        )

    reward = abs(
        take_profit_1 -
        entry
    )

    if stop_distance > 0:

        risk_reward = (
            reward /
            stop_distance
        )

    else:

        risk_reward = 0

    return {

        "entry": round(
            entry,
            6
        ),

        "stop_loss": round(
            stop_loss,
            6
        ),

        "take_profit_1": round(
            take_profit_1,
            6
        ),

        "take_profit_2": round(
            take_profit_2,
            6
        ),

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
            6
        ),

        "risk_reward": round(
            risk_reward,
            2
        )
    }


# =========================================================
# 13. MAIN SIGNAL ENGINE
# =========================================================

def generate_analysis(
    df,
    symbol=DEFAULT_SYMBOL,
    account_size=DEFAULT_ACCOUNT_SIZE,
    risk_percent=DEFAULT_RISK_PERCENT,
    multi_tf=None
):

    last = df.iloc[-1]

    support, resistance = (
        support_resistance(
            df
        )
    )

    structure = (
        market_structure(
            df
        )
    )

    breakout = (
        breakout_analysis(
            df,
            support,
            resistance
        )
    )

    gap = (
        gap_analysis(
            df
        )
    )

    trap_risk = (
        trap_analysis(
            df,
            support,
            resistance
        )
    )

    trendline = (
        trendline_analysis(
            df
        )
    )

    price = float(
        last["close"]
    )

    atr = float(
        last["ATR"]
    )

    score = 0

    reasons = []

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):

        score += 3

        reasons.append(
            "Strong bullish EMA trend"
        )

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):

        score -= 3

        reasons.append(
            "Strong bearish EMA trend"
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi = float(
        last["RSI"]
    )

    if (
        rsi > 50
        and
        rsi < 70
    ):

        score += 1

        reasons.append(
            "Bullish RSI momentum"
        )

    elif (
        rsi < 50
        and
        rsi > 30
    ):

        score -= 1

        reasons.append(
            "Bearish RSI momentum"
        )

    elif rsi >= 70:

        reasons.append(
            "RSI overbought - caution"
        )

    elif rsi <= 30:

        reasons.append(
            "RSI oversold - caution"
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
    # MARKET STRUCTURE
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
            "Trendline direction bullish"
        )

    elif trendline[
        "trendline"
    ] == "DOWNTREND":

        score -= 1

        reasons.append(
            "Trendline direction bearish"
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
    # VOLUME
    # -----------------------------------------------------

    volume_confirmed = (
        last["volume"] >
        last["VOL_AVG"]
    )

    if volume_confirmed:

        reasons.append(
            "Volume confirmation"
        )

        if score > 0:

            score += 1

        elif score < 0:

            score -= 1

    # -----------------------------------------------------
    # GAP
    # -----------------------------------------------------

    if gap == "GAP_UP":

        reasons.append(
            "Gap up detected"
        )

    elif gap == "GAP_DOWN":

        reasons.append(
            "Gap down detected"
        )

    # -----------------------------------------------------
    # TRAP FILTER
    # -----------------------------------------------------

    if trap_risk == "HIGH":

        score = int(
            score *
            0.5
        )

        reasons.append(
            "High fake-breakout trap risk"
        )

    elif trap_risk == "MEDIUM":

        reasons.append(
            "Medium trap risk"
        )

    # -----------------------------------------------------
    # MULTI-TIMEFRAME CONFIRMATION
    # -----------------------------------------------------

    mtf_bonus = 0

    if multi_tf:

        bullish_count = sum(
            1
            for value
            in multi_tf.values()
            if value == "BULLISH"
        )

        bearish_count = sum(
            1
            for value
            in multi_tf.values()
            if value == "BEARISH"
        )

        if bullish_count >= 2:

            mtf_bonus = 2

            score += 2

            reasons.append(
                "Multi-timeframe bullish confirmation"
            )

        elif bearish_count >= 2:

            mtf_bonus = -2

            score -= 2

            reasons.append(
                "Multi-timeframe bearish confirmation"
            )

        else:

            reasons.append(
                "Multi-timeframe confirmation mixed"
            )

    # -----------------------------------------------------
    # FINAL SIGNAL
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = min(
        95,
        50 +
        abs(score) * 5
    )

    # Reduce confidence for trap
    if trap_risk == "HIGH":

        confidence -= 15

    elif trap_risk == "MEDIUM":

        confidence -= 7

    confidence = max(
        5,
        confidence
    )

    # -----------------------------------------------------
    # LIQUIDATION RISK
    # -----------------------------------------------------

    liquidation = (
        liquidation_risk_analysis(
            df,
            signal,
            atr
        )
    )

    # -----------------------------------------------------
    # RISK MANAGEMENT
    # -----------------------------------------------------

    risk_plan = (
        risk_management(
            price,
            atr,
            signal,
            account_size,
            risk_percent
        )
    )

    return {

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
                6
            ),

        "trend":
            structure,

        "trendline":
            trendline,

        "support":
            round(
                support,
                6
            ),

        "resistance":
            round(
                resistance,
                6
            ),

        "breakout":
            breakout,

        "gap":
            gap,

        "trap_risk":
            trap_risk,

        "liquidation_risk":
            liquidation,

        "risk_management":
            risk_plan,

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
                6
            ),

        "ema50":
            round(
                float(
                    last["EMA50"]
                ),
                6
            ),

        "ema200":
            round(
                float(
                    last["EMA200"]
                ),
                6
            ),

        "macd":
            round(
                float(
                    last["MACD"]
                ),
                6
            ),

        "macd_signal":
            round(
                float(
                    last["MACD_SIGNAL"]
                ),
                6
            ),

        "atr":
            round(
                atr,
                6
            ),

        "momentum_percent":
            round(
                float(
                    last["MOMENTUM"]
                ),
                4
            ),

        "volume_ratio":
            round(
                float(
                    last["VOLUME_RATIO"]
                )
                if pd.notna(
                    last["VOLUME_RATIO"]
                )
                else 0,
                3
            ),

        "volume_confirmed":
            bool(
                volume_confirmed
            ),

        "multi_timeframe":
            multi_tf,

        "reasons":
            reasons
    }


# =========================================================
# 14. LIVE ANALYSIS API
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "message":
            "GM AI Trading Bot v6 "
            "All-In-One Analysis System is running.",

        "endpoints": [

            "/analysis/BTCUSDT",

            "/analysis/BTCUSDT?account_size=1000&risk_percent=1",

            "/backtest/BTCUSDT",

            "/backtest/BTCUSDT?limit=1000"
        ],

        "features": [

            "EMA20 EMA50 EMA200",

            "RSI",

            "MACD",

            "ATR",

            "Momentum",

            "Volume",

            "Support Resistance",

            "Market Structure",

            "Breakout Breakdown",

            "Gap Up Gap Down",

            "Trap Detection",

            "Estimated Liquidation Risk",

            "Trendline Analysis",

            "Multi Timeframe Confirmation",

            "Risk Management",

            "Position Sizing",

            "Backtesting"
        ]
    })


@app.route(
    "/analysis/<symbol>"
)
def analysis(symbol):

    try:

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

        if account_size <= 0:

            raise ValueError(
                "account_size must be greater than 0"
            )

        if (
            risk_percent <= 0
            or
            risk_percent > 10
        ):

            raise ValueError(
                "risk_percent must be between 0 and 10"
            )

        symbol = symbol.upper()

        # Main 5m data
        df = get_data(
            symbol,
            "5m",
            500
        )

        df = calculate_indicators(
            df
        )

        # Multi-timeframe
        multi_tf = (
            multi_timeframe_analysis(
                symbol
            )
        )

        result = generate_analysis(
            df,
            symbol,
            account_size,
            risk_percent,
            multi_tf
        )

        result["status"] = (
            "success"
        )

        result["symbol"] = (
            symbol
        )

        result["timeframe"] = (
            "5m"
        )

        result["warning"] = (
            "Technical analysis only. "
            "No guaranteed prediction. "
            "Liquidation risk is estimated "
            "because direct liquidation data "
            "is not included."
        )

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 15. BACKTEST
# =========================================================

@app.route(
    "/backtest/<symbol>"
)
def backtest(symbol):

    try:

        limit = int(
            request.args.get(
                "limit",
                500
            )
        )

        limit = max(
            300,
            min(
                limit,
                1000
            )
        )

        df = get_data(
            symbol.upper(),
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

        starting_balance = 1000.0

        balance = (
            starting_balance
        )

        wins = 0

        losses = 0

        # Backtest from 200th candle
        for i in range(
            200,
            len(df) - 1
        ):

            historical = (
                df.iloc[
                    :i + 1
                ].copy()
            )

            analysis_result = (
                generate_analysis(
                    historical,
                    symbol.upper(),
                    1000,
                    1,
                    None
                )
            )

            signal = (
                analysis_result[
                    "signal"
                ]
            )

            current_price = float(
                df.iloc[i][
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

            # -------------------------------------------------
            # BUY
            # -------------------------------------------------

            if signal in [
                "BUY",
                "STRONG BUY"
            ]:

                total_signals += 1

                if (
                    next_price >
                    current_price
                ):

                    correct += 1

                    wins += 1

                    balance += 10

                else:

                    wrong += 1

                    losses += 1

                    balance -= 10

            # -------------------------------------------------
            # SELL
            # -------------------------------------------------

            elif signal in [
                "SELL",
                "STRONG SELL"
            ]:

                total_signals += 1

                if (
                    next_price <
                    current_price
                ):

                    correct += 1

                    wins += 1

                    balance += 10

                else:

                    wrong += 1

                    losses += 1

                    balance -= 10

            else:

                neutral += 1

        if total_signals > 0:

            win_rate = (
                correct /
                total_signals
            ) * 100

        else:

            win_rate = 0

        profit_loss = (
            balance -
            starting_balance
        )

        return jsonify({

            "status":
                "success",

            "symbol":
                symbol.upper(),

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

            "wins":
                wins,

            "losses":
                losses,

            "win_rate_percent":
                round(
                    win_rate,
                    2
                ),

            "starting_balance":
                starting_balance,

            "ending_balance":
                round(
                    balance,
                    2
                ),

            "estimated_profit_loss":
                round(
                    profit_loss,
                    2
                ),

            "note":
                "Simplified historical backtest. "
                "It does not model fees, slippage, "
                "funding, liquidation or real order execution. "
                "Past performance does not guarantee future results."

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 16. RUN APP
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
