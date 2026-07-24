from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "GM AI Trading Bot is running!"

@app.route("/price/<symbol>")
def price(symbol):
    try:
        symbol = symbol.upper() + "USDT"

        url = "https://api.binance.com/api/v3/ticker/price"
        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=15
        )

        data = response.json()

        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "symbol": symbol,
                "binance_response": data
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
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
