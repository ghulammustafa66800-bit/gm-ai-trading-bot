from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "GM AI Trading Bot is running!"

@app.route("/price/<symbol>")
def price(symbol):
    try:
        symbol = symbol.upper()
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        response = requests.get(url, timeout=10)
        data = response.json()

        return jsonify({
            "symbol": symbol + "USDT",
            "price": data.get("price"),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
