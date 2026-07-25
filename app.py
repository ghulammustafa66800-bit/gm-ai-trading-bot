from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"


# =========================================================
# 1. GET MARKET DATA
# =========================================================

def get_data(symbol="BTCUSDT", interval="5m", limit=500):

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(int(limit), 1000)
    }

    response = requests.get(
        MEXC_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 100:
        raise Exception("Not enough market data")

    df = pd.DataFrame(data, columns=[
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume"
    ])

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(df[col])

    return df


# =========================================================
# 2. TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    # EMA Trend
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

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
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
        .ewm(span=9, adjust=False)
        .mean()
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
        .rolling(14)
        .mean()
    )

    # Volume
    df["VOL_AVG"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # Momentum
    df["MOMENTUM"] = (
        df["close"]
        .pct_change(5) * 100
    )

    return df


# =========================================================
# 3. SUPPORT / RESISTANCE
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
# 4. MARKET STRUCTURE
# =========================================================

def market_structure(df):

    recent = df.tail(10)

    first_high = recent["high"].iloc[:5].max()
    second_high = recent["high"].iloc[5:].max()

    first_low = recent["low"].iloc[:5].min()
    second_low = recent["low"].iloc[5:].min()

    if (
        second_high > first_high
        and second_low > first_low
    ):
        return "BULLISH"

    if (
        second_high < first_high
        and second_low < first_low
    ):
        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# 5. BREAKOUT / BREAKDOWN
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
        and previous["close"] <= resistance
    ):
        return "BREAKOUT"

    if (
        last["close"] < support
        and previous["close"] >= support
    ):
        return "BREAKDOWN"

    return "NO_BREAKOUT"


# =========================================================
# 6. GAP DETECTION
# =========================================================

def gap_analysis(df):

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

        return "GAP_UP"

    if gap_percent < -0.3:

        return "GAP_DOWN"

    return "NO_GAP"


# =========================================================
# 7. TRAP / FAKE BREAKOUT RISK
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

    # Price moved above resistance
    # but closed back below it
    fake_breakout = (
        last["high"] > resistance
        and last["close"] < resistance
    )

    # Price moved below support
    # but closed back above it
    fake_breakdown = (
        last["low"] < support
        and last["close"] > support
    )

    if fake_breakout or fake_breakdown:

        if not volume_confirmed:

            return "HIGH"

        return "MEDIUM"

    return "LOW"


# =========================================================
# 8. COMBINED SIGNAL ENGINE
# =========================================================

def generate_analysis(df):

    last = df.iloc[-1]

    support, resistance = (
        support_resistance(df)
    )

    structure = market_structure(df)

    breakout = breakout_analysis(
        df,
        support,
        resistance
    )

    gap = gap_analysis(df)

    trap_risk = trap_analysis(
        df,
        support,
        resistance
    )

    score = 0

    reasons = []

    # -------------------------
    # EMA TREND
    # -------------------------

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

    # -------------------------
    # RSI
    # -------------------------

    if (
        last["RSI"] > 50
        and last["RSI"] < 70
    ):

        score += 1

        reasons.append(
            "Bullish RSI momentum"
        )

    elif (
        last["RSI"] < 50
        and last["RSI"] > 30
    ):

        score -= 1

        reasons.append(
            "Bearish RSI momentum"
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
    # MARKET STRUCTURE
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
    # BREAKOUT
    # -------------------------

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

    # -------------------------
    # VOLUME
    # -------------------------

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

    # -------------------------
    # TRAP FILTER
    # -------------------------

    if trap_risk == "HIGH":

        score = int(
            score * 0.5
        )

        reasons.append(
            "High fake-breakout trap risk"
        )

    elif trap_risk == "MEDIUM":

        reasons.append(
            "Medium trap risk"
        )

    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    if score >= 6:

        signal = "STRONG BUY"

    elif score >= 3:

        signal = "BUY"

    elif score <= -6:

        signal = "STRONG SELL"

    elif score <= -3:

        signal = "SELL"

    else:

        signal = "NEUTRAL"

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = min(
        95,
        50 + abs(score) * 5
    )

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    price = float(
        last["close"]
    )

    atr = float(
        last["ATR"]
    )

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        entry = price

        stop_loss = (
            price -
            (atr * 1.5)
        )

        take_profit_1 = (
            price +
            (atr * 2)
        )

        take_profit_2 = (
            price +
            (atr * 3)
        )

    elif signal in [
        "SELL",
        "STRONG SELL"
    ]:

        entry = price

        stop_loss = (
            price +
            (atr * 1.5)
        )

        take_profit_1 = (
            price -
            (atr * 2)
        )

        take_profit_2 = (
            price -
            (atr * 3)
        )

    else:

        entry = price

        stop_loss = None

        take_profit_1 = None

        take_profit_2 = None

    return {

        "signal": signal,

        "confidence": round(
            confidence,
            2
        ),

        "score": score,

        "current_price": round(
            price,
            6
        ),

        "trend": structure,

        "support": round(
            support,
            6
        ),

        "resistance": round(
            resistance,
            6
        ),

        "breakout": breakout,

        "gap": gap,

        "trap_risk": trap_risk,

        "entry": round(
            entry,
            6
        ),

        "stop_loss": (
            round(
                stop_loss,
                6
            )
            if stop_loss
            else None
        ),

        "take_profit_1": (
            round(
                take_profit_1,
                6
            )
            if take_profit_1
            else None
        ),

        "take_profit_2": (
            round(
                take_profit_2,
                6
            )
            if take_profit_2
            else None
        ),

        "rsi": round(
            float(last["RSI"]),
            2
        ),

        "ema20": round(
            float(last["EMA20"]),
            6
        ),

        "ema50": round(
            float(last["EMA50"]),
            6
        ),

        "ema200": round(
            float(last["EMA200"]),
            6
        ),

        "macd": round(
            float(last["MACD"]),
            6
        ),

        "atr": round(
            atr,
            6
        ),

        "momentum_percent": round(
            float(last["MOMENTUM"]),
            4
        ),

        "volume_confirmed":
            bool(volume_confirmed),

        "reasons":
            reasons
    }


# =========================================================
# 9. LIVE ANALYSIS API
# =========================================================

@app.route("/")
def home():

    return (
        "GM AI Trading Bot v5 "
        "All-In-One Analysis System is running!"
    )


@app.route("/analysis/<symbol>")
def analysis(symbol):

    try:

        symbol = symbol.upper()

        df = get_data(
            symbol,
            "5m",
            500
        )

        df = calculate_indicators(
            df
        )

        result = generate_analysis(
            df
        )

        result["status"] = "success"

        result["symbol"] = symbol

        result["timeframe"] = "5m"

        result["warning"] = (
            "Technical analysis only. "
            "No guaranteed prediction. "
            "Trap and liquidation risk are "
            "estimated unless direct liquidation "
            "data is available."
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
# 10. BACKTEST
# =========================================================

@app.route("/backtest/<symbol>")
def backtest(symbol):

    try:

        limit = int(
            request.args.get(
                "limit",
                500
            )
        )

        limit = min(
            limit,
            1000
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

        # One next candle outcome
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
                    historical
                )
            )

            signal = (
                analysis_result[
                    "signal"
                ]
            )

            current_price = float(
                df.iloc[i]["close"]
            )

            next_price = float(
                df.iloc[
                    i + 1
                ]["close"]
            )

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

                else:

                    wrong += 1

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

            "win_rate_percent":
                round(
                    win_rate,
                    2
                ),

            "note":
                "Historical backtest only. "
                "It does not guarantee future results."

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 11. RUN APP
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
