from flask import Flask, jsonify, request
import requests
import pandas as pd
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"


def get_data(symbol="BTCUSDT", interval="5m", limit=500):
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }

    response = requests.get(MEXC_URL, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 60:
        raise Exception("Not enough market data")

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume"
    ])

    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])

    return df


def calculate_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, 0.000001)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["VOL_AVG"] = df["volume"].rolling(20).mean()

    return df


def get_signal(row):
    score = 0

    if row["EMA20"] > row["EMA50"]:
        score += 2
    elif row["EMA20"] < row["EMA50"]:
        score -= 2

    if 50 < row["RSI"] < 70:
        score += 1
    elif 30 < row["RSI"] < 50:
        score -= 1

    if row["MACD"] > row["MACD_SIGNAL"]:
        score += 2
    else:
        score -= 2

    if row["volume"] > row["VOL_AVG"]:
        if score > 0:
            score += 1
        elif score < 0:
            score -= 1

    if score >= 4:
        return "BUY"

    if score <= -4:
        return "SELL"

    return "NEUTRAL"


@app.route("/")
def home():
    return "GM AI Trading Bot v3 is running!"


@app.route("/signal/<symbol>")
def signal(symbol):
    try:
        df = get_data(symbol, "5m", 200)
        df = calculate_indicators(df)

        last = df.iloc[-1]
        signal_result = get_signal(last)

        return jsonify({
            "status": "success",
            "symbol": symbol.upper(),
            "timeframe": "5m",
            "current_price": round(float(last["close"]), 4),
            "RSI": round(float(last["RSI"]), 2),
            "EMA20": round(float(last["EMA20"]), 4),
            "EMA50": round(float(last["EMA50"]), 4),
            "MACD": round(float(last["MACD"]), 6),
            "MACD_SIGNAL": round(float(last["MACD_SIGNAL"]), 6),
            "signal": signal_result,
            "warning": "Technical analysis only. No guaranteed prediction."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/backtest/<symbol>")
def backtest(symbol):
    try:
        limit = int(request.args.get("limit", 500))

        if limit > 1000:
            limit = 1000

        df = get_data(symbol, "5m", limit)
        df = calculate_indicators(df)

        correct = 0
        wrong = 0
        neutral = 0
        total_signals = 0

        # Test each candle against the NEXT candle
        for i in range(60, len(df) - 1):

            current = df.iloc[i]
            next_candle = df.iloc[i + 1]

            signal_result = get_signal(current)

            current_price = float(current["close"])
            next_price = float(next_candle["close"])

            if signal_result == "BUY":
                total_signals += 1

                if next_price > current_price:
                    correct += 1
                else:
                    wrong += 1

            elif signal_result == "SELL":
                total_signals += 1

                if next_price < current_price:
                    correct += 1
                else:
                    wrong += 1

            else:
                neutral += 1

        if total_signals > 0:
            win_rate = (correct / total_signals) * 100
        else:
            win_rate = 0

        return jsonify({
            "status": "success",
            "symbol": symbol.upper(),
            "timeframe": "5m",
            "candles_tested": len(df) - 60,
            "total_signals": total_signals,
            "correct_signals": correct,
            "wrong_signals": wrong,
            "neutral_signals": neutral,
            "win_rate_percent": round(win_rate, 2),
            "note": "Backtest results are historical and do not guarantee future performance."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
