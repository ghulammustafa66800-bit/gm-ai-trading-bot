from flask import Flask, jsonify, request
import requests
import pandas as pd
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"


def get_data(symbol="BTCUSDT", interval="5m", limit=200):
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }

    response = requests.get(MEXC_URL, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 50:
        raise Exception("Not enough market data")

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume"
    ])

    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])

    return df


def calculate_indicators(df):
    # EMA
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, 0.000001)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Volume average
    df["VOL_AVG"] = df["volume"].rolling(20).mean()

    return df


def generate_signal(df):
    last = df.iloc[-1]

    score = 0

    # EMA Trend
    if last["EMA20"] > last["EMA50"]:
        score += 2
    elif last["EMA20"] < last["EMA50"]:
        score -= 2

    # RSI
    if 50 < last["RSI"] < 70:
        score += 1
    elif 30 < last["RSI"] < 50:
        score -= 1

    # MACD
    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 2
    else:
        score -= 2

    # Volume
    if last["volume"] > last["VOL_AVG"]:
        if score > 0:
            score += 1
        elif score < 0:
            score -= 1

    # Final Signal
    if score >= 4:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    confidence = min(95, 50 + abs(score) * 8)

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "score": score
    }


@app.route("/")
def home():
    return "GM AI Trading Bot v2 is running!"


@app.route("/signal/<symbol>")
def signal(symbol):
    try:
        symbol = symbol.upper()

        df = get_data(symbol, "5m", 200)
        df = calculate_indicators(df)

        result = generate_signal(df)
        last = df.iloc[-1]

        return jsonify({
            "status": "success",
            "symbol": symbol,
            "timeframe": "5m",
            "current_price": round(float(last["close"]), 4),
            "EMA20": round(float(last["EMA20"]), 4),
            "EMA50": round(float(last["EMA50"]), 4),
            "RSI": round(float(last["RSI"]), 2),
            "MACD": round(float(last["MACD"]), 6),
            "MACD_SIGNAL": round(float(last["MACD_SIGNAL"]), 6),
            "signal": result["signal"],
            "confidence": result["confidence"],
            "score": result["score"],
            "warning": "Technical analysis only. No guaranteed prediction."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
