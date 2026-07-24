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

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

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

        if not data:
            return jsonify({
                "status": "error",
                "message": "No market data received"
            }), 400

        closes = [float(candle[4]) for candle in data]
        volumes = [float(candle[5]) for candle in data]

        current_price = closes[-1]

        # Indicators
        ema9 = ema(closes, 9)
        ema21 = ema(closes, 21)
        rsi_value = rsi(closes, 14)
        macd_value = macd(closes)

        # Volume confirmation
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]

        score = 0
        reasons = []

        # EMA trend
        if ema9 > ema21:
            score += 1
            reasons.append("EMA bullish")
        else:
            score -= 1
            reasons.append("EMA bearish")

        # RSI
        if rsi_value > 55:
            score += 1
            reasons.append("RSI bullish")
        elif rsi_value < 45:
            score -= 1
            reasons.append("RSI bearish")
        else:
            reasons.append("RSI neutral")

        # MACD
        if macd_value > 0:
            score += 1
            reasons.append("MACD bullish")
        else:
            score -= 1
            reasons.append("MACD bearish")

        # Volume
        if current_volume > avg_volume:
            reasons.append("Volume confirmation")
        else:
            reasons.append("Low volume")

        # Final signal
        if score >= 2:
            signal = "UP"
        elif score <= -2:
            signal = "DOWN"
        else:
            signal = "NEUTRAL"

        # Confidence
        confidence = round(abs(score) / 3 * 100, 1)

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "timeframe": "5m",
            "current_price": current_price,
            "EMA9": round(ema9, 4),
            "EMA21": round(ema21, 4),
            "RSI": round(rsi_value, 2),
            "MACD": round(macd_value, 6),
            "volume": round(current_volume, 2),
            "average_volume": round(avg_volume, 2),
            "signal": signal,
            "confidence": confidence,
            "reasons": reasons,
            "warning": "Technical analysis only. No guaranteed prediction."
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )
