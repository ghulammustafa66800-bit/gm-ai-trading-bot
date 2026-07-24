from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)


@app.route("/")
def home():
    return "GM AI Trading Bot is running!"


@app.route("/price/<symbol>")
def price(symbol):
    try:
        # MEXC Spot symbol format: BTCUSDT
        symbol = symbol.upper() + "USDT"

        url = "https://api.mexc.com/api/v3/ticker/price"

        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=15
        )

        data = response.json()

        # Check MEXC API response
        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "symbol": symbol,
                "mexc_response": data,
                "http_status": response.status_code
            }), response.status_code

        return jsonify({
            "status": "success",
            "symbol": data.get("symbol"),
            "price": data.get("price")
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
