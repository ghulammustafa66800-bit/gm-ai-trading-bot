from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "GM AI Trading Bot is running!"

@app.route("/test")
def test_binance():
    try:
        url = "https://fapi.binance.com/fapi/v1/time"
        response = requests.get(url, timeout=15)

        return jsonify({
            "status_code": response.status_code,
            "response": response.json()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
