from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"


@app.route("/")
def home():
    return "GM AI Trading Bot is running!"


@app.route("/analysis/<symbol>")
def analysis(symbol):
    try:
        symbol = symbol.upper() + "USDT"

        # Get 5-minute candles from MEXC
        response = requests.get(
            MEXC_URL,
            params={
                "symbol": symbol,
                "interval": "5m",
                "limit": 20
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
                "message": "No candle data received"
            }), 400

        # Closing prices
        closes = [float(candle[4]) for candle in data]

        current_price = closes[-1]

        # Simple moving averages
        sma5 = sum(closes[-5:]) / 5
        sma10 = sum(closes[-10:]) / 10

        # Simple trend analysis
        if sma5 > sma10 and current_price > sma5:
            signal = "UP"
        elif sma5 < sma10 and current_price < sma5:
            signal = "DOWN"
        else:
            signal = "NEUTRAL"

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "timeframe": "5m",
            "current_price": current_price,
            "SMA5": round(sma5, 4),
            "SMA10": round(sma10, 4),
            "signal": signal,
            "warning": "This is technical analysis, not guaranteed financial advice."
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
