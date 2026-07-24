from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"


def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for price in values[period:]:
        result = (price - result) * multiplier + result

    return result


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def macd(values):
    ema12 = ema(values, 12)
    ema26 = ema(values, 26)

    if ema12 is None or ema26 is None:
        return None

    return ema12 - ema26


def generate_signal(closes):

    if len(closes) < 30:
        return "NEUTRAL"

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    rsi_value = rsi(closes, 14)
    macd_value = macd(closes)

    score = 0

    if ema9 > ema21:
        score += 1
    else:
        score -= 1

    if rsi_value > 55:
        score += 1
    elif rsi_value < 45:
        score -= 1

    if macd_value > 0:
        score += 1
    else:
        score -= 1

    if score >= 2:
        return "UP"

    if score <= -2:
        return "DOWN"

    return "NEUTRAL"


@app.route("/")
def home():
    return "GM AI Trading Bot is running!"


@app.route("/analysis/<symbol>")
def analysis(symbol):

    try:

        symbol = symbol.upper() + "USDT"

        response = requests.get(
            MEXC_URL,
            params={
                "symbol": symbol,
                "interval": "5m",
                "limit": 100
            },
            timeout=15
        )

        data = response.json()

        if response.status_code != 200:

            return jsonify({
                "status": "error",
                "mexc_response": data
            }), response.status_code

        closes = [
            float(candle[4])
            for candle in data
        ]

        signal = generate_signal(closes)

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "timeframe": "5m",
            "current_price": closes[-1],
            "signal": signal
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/backtest/<symbol>")
def backtest(symbol):

    try:

        symbol = symbol.upper() + "USDT"

        response = requests.get(
            MEXC_URL,
            params={
                "symbol": symbol,
                "interval": "5m",
                "limit": 1000
            },
            timeout=20
        )

        data = response.json()

        if response.status_code != 200:

            return jsonify({
                "status": "error",
                "mexc_response": data
            }), response.status_code

        closes = [
            float(candle[4])
            for candle in data
        ]

        total_signals = 0
        correct = 0
        wrong = 0
        neutral = 0

        results = []

        # Test each historical candle
        for i in range(30, len(closes) - 1):

            historical_data = closes[:i]

            signal = generate_signal(
                historical_data
            )

            current_price = closes[i]
            next_price = closes[i + 1]

            if signal == "NEUTRAL":

                neutral += 1
                continue

            total_signals += 1

            actual_direction = (
                "UP"
                if next_price > current_price
                else "DOWN"
            )

            if signal == actual_direction:

                correct += 1

                result = "CORRECT"

            else:

                wrong += 1

                result = "WRONG"

            results.append({
                "signal": signal,
                "actual": actual_direction,
                "result": result
            })

        if total_signals > 0:

            win_rate = (
                correct / total_signals
            ) * 100

        else:

            win_rate = 0

        return jsonify({

            "status": "success",

            "symbol": symbol,

            "timeframe": "5m",

            "candles_tested": len(closes),

            "total_signals": total_signals,

            "correct_signals": correct,

            "wrong_signals": wrong,

            "neutral_signals": neutral,

            "win_rate_percent": round(
                win_rate,
                2
            ),

            "note":
            "Backtest results are historical and do not guarantee future performance."

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


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
