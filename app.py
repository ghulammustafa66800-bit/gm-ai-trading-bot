from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"

DEFAULT_ACCOUNT = 1000.0
DEFAULT_RISK = 1.0
FEE_PER_SIDE = 0.0005


# =========================================================
# 1. GET MEXC DATA
# =========================================================

def get_data(symbol="BTCUSDT", interval="5m", limit=500):

    response = requests.get(
        MEXC_URL,
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(int(limit), 1000)
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 100:
        raise Exception(
            f"Not enough data for {symbol} {interval}"
        )

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
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(
        drop=True
    )

    return df


# =========================================================
# 2. INDICATORS
# =========================================================

def indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["RSI"] = (
        100 -
        100 /
        (1 + rs)
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

    df["MACD"] = (
        ema12 -
        ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # ATR
    hl = (
        df["high"] -
        df["low"]
    )

    hc = abs(
        df["high"] -
        df["close"].shift()
    )

    lc = abs(
        df["low"] -
        df["close"].shift()
    )

    tr = pd.concat(
        [
            hl,
            hc,
            lc
        ],
        axis=1
    ).max(
        axis=1
    )

    df["ATR"] = tr.rolling(
        14
    ).mean()

    # Volume
    df["VOL_AVG"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # Momentum
    df["MOMENTUM"] = (
        df["close"]
        .pct_change(5)
        * 100
    )

    # Candle range/body
    df["RANGE"] = (
        df["high"] -
        df["low"]
    )

    df["BODY"] = abs(
        df["close"] -
        df["open"]
    )

    df["BODY_RATIO"] = (
        df["BODY"] /
        df["RANGE"].replace(
            0,
            np.nan
        )
    )

    return df


# =========================================================
# 3. SINGLE TIMEFRAME ANALYSIS
# =========================================================

def timeframe_analysis(df):

    last = df.iloc[-1]

    bullish = 0
    bearish = 0

    # EMA
    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):
        bullish += 3

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):
        bearish += 3

    # RSI
    if (
        last["RSI"] > 50
        and
        last["RSI"] < 70
    ):
        bullish += 1

    elif (
        last["RSI"] < 50
        and
        last["RSI"] > 30
    ):
        bearish += 1

    # MACD
    if (
        last["MACD"] >
        last["MACD_SIGNAL"]
    ):
        bullish += 2

    else:
        bearish += 2

    # Momentum
    if last["MOMENTUM"] > 0.15:
        bullish += 1

    elif last["MOMENTUM"] < -0.15:
        bearish += 1

    # Volume
    volume_confirmed = (
        last["volume"] >
        last["VOL_AVG"]
    )

    if bullish > bearish:
        direction = "BULLISH"

    elif bearish > bullish:
        direction = "BEARISH"

    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "bullish_score": bullish,
        "bearish_score": bearish,
        "rsi": round(
            float(last["RSI"]),
            2
        ),
        "momentum": round(
            float(last["MOMENTUM"]),
            4
        ),
        "volume_confirmed":
            bool(volume_confirmed)
    }


# =========================================================
# 4. SUPPORT / RESISTANCE
# =========================================================

def support_resistance(df):

    recent = df.iloc[
        -51:-1
    ]

    return (
        float(
            recent["low"].min()
        ),
        float(
            recent["high"].max()
        )
    )


# =========================================================
# 5. MARKET STRUCTURE
# =========================================================

def market_structure(df):

    recent = df.tail(
        20
    )

    a = recent.iloc[
        :10
    ]

    b = recent.iloc[
        10:
    ]

    high_a = a["high"].max()
    high_b = b["high"].max()

    low_a = a["low"].min()
    low_b = b["low"].min()

    if (
        high_b > high_a
        and
        low_b > low_a
    ):
        return "BULLISH"

    if (
        high_b < high_a
        and
        low_b < low_a
    ):
        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# 6. TRENDLINE ESTIMATE
# =========================================================

def trendline(df):

    recent = df.tail(
        30
    )

    first = float(
        recent["close"].iloc[0]
    )

    last = float(
        recent["close"].iloc[-1]
    )

    change = (
        (
            last - first
        )
        /
        first
    ) * 100

    if change > 0.5:
        return "UPTREND"

    if change < -0.5:
        return "DOWNTREND"

    return "FLAT"


# =========================================================
# 7. BREAKOUT / TRAP
# =========================================================

def breakout_trap(
    df,
    support,
    resistance
):

    last = df.iloc[-1]
    previous = df.iloc[-2]

    volume_ok = (
        last["volume"] >
        last["VOL_AVG"]
    )

    breakout = "NONE"

    trap = "LOW"

    # Breakout
    if (
        last["close"] >
        resistance
        and
        previous["close"] <=
        resistance
    ):
        breakout = "BREAKOUT"

    # Breakdown
    elif (
        last["close"] <
        support
        and
        previous["close"] >=
        support
    ):
        breakout = "BREAKDOWN"

    # Fake breakout
    fake_up = (
        last["high"] >
        resistance
        and
        last["close"] <
        resistance
    )

    fake_down = (
        last["low"] <
        support
        and
        last["close"] >
        support
    )

    weak_candle = (
        last["BODY_RATIO"] <
        0.35
    )

    if fake_up or fake_down:

        if not volume_ok:
            trap = "HIGH"

        else:
            trap = "MEDIUM"

    elif weak_candle:

        trap = "MEDIUM"

    return breakout, trap


# =========================================================
# 8. GAP
# =========================================================

def gap_analysis(df):

    last = df.iloc[-1]
    previous = df.iloc[-2]

    gap = (
        (
            last["open"] -
            previous["close"]
        )
        /
        previous["close"]
    ) * 100

    if gap > 0.3:
        return "GAP_UP"

    if gap < -0.3:
        return "GAP_DOWN"

    return "NO_GAP"


# =========================================================
# 9. MONEY MANAGEMENT
# =========================================================

def money_management(
    entry,
    stop,
    account,
    risk_percent
):

    risk_amount = (
        account *
        risk_percent /
        100
    )

    distance = abs(
        entry -
        stop
    )

    if distance <= 0:

        return {
            "risk_amount": 0,
            "position_size": 0,
            "position_value": 0
        }

    size = (
        risk_amount /
        distance
    )

    value = (
        size *
        entry
    )

    return {

        "risk_amount":
            round(
                risk_amount,
                4
            ),

        "position_size":
            round(
                size,
                8
            ),

        "position_value":
            round(
                value,
                4
            )
    }


# =========================================================
# 10. COMPLETE MULTI-TIMEFRAME ANALYSIS
# =========================================================

def complete_analysis(
    symbol,
    account=1000,
    risk_percent=1
):

    # Get all timeframes
    df5 = indicators(
        get_data(
            symbol,
            "5m",
            500
        )
    )

    df15 = indicators(
        get_data(
            symbol,
            "15m",
            500
        )
    )

    df1h = indicators(
        get_data(
            symbol,
            "1h",
            500
        )
    )

    # Timeframe direction
    tf5 = timeframe_analysis(
        df5
    )

    tf15 = timeframe_analysis(
        df15
    )

    tf1h = timeframe_analysis(
        df1h
    )

    # 5m details
    last = df5.iloc[-1]

    support, resistance = (
        support_resistance(
            df5
        )
    )

    structure = (
        market_structure(
            df5
        )
    )

    trend = trendline(
        df5
    )

    breakout, trap = (
        breakout_trap(
            df5,
            support,
            resistance
        )
    )

    gap = gap_analysis(
        df5
    )

    # =====================================================
    # MULTI-TIMEFRAME SCORE
    # =====================================================

    score = 0

    reasons = []

    if tf1h["direction"] == "BULLISH":
        score += 3
        reasons.append(
            "1H bullish confirmation"
        )

    elif tf1h["direction"] == "BEARISH":
        score -= 3
        reasons.append(
            "1H bearish confirmation"
        )

    if tf15["direction"] == "BULLISH":
        score += 2
        reasons.append(
            "15M bullish confirmation"
        )

    elif tf15["direction"] == "BEARISH":
        score -= 2
        reasons.append(
            "15M bearish confirmation"
        )

    if tf5["direction"] == "BULLISH":
        score += 2
        reasons.append(
            "5M bullish confirmation"
        )

    elif tf5["direction"] == "BEARISH":
        score -= 2
        reasons.append(
            "5M bearish confirmation"
        )

    # Structure
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

    # Trendline
    if trend == "UPTREND":
        score += 1

    elif trend == "DOWNTREND":
        score -= 1

    # Breakout
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

    # Trap filter
    if trap == "HIGH":

        score = int(
            score * 0.5
        )

        reasons.append(
            "High fake-breakout trap risk"
        )

    elif trap == "MEDIUM":

        reasons.append(
            "Medium trap risk"
        )

    # =====================================================
    # STRICT MTF FILTER
    # =====================================================

    bullish_alignment = (
        tf1h["direction"] ==
        "BULLISH"
        and
        tf15["direction"] ==
        "BULLISH"
        and
        tf5["direction"] ==
        "BULLISH"
    )

    bearish_alignment = (
        tf1h["direction"] ==
        "BEARISH"
        and
        tf15["direction"] ==
        "BEARISH"
        and
        tf5["direction"] ==
        "BEARISH"
    )

    # =====================================================
    # SIGNAL
    # =====================================================

    if (
        bullish_alignment
        and
        score >= 7
        and
        trap != "HIGH"
    ):

        signal = "STRONG BUY"

    elif (
        bullish_alignment
        and
        score >= 5
        and
        trap != "HIGH"
    ):

        signal = "BUY"

    elif (
        bearish_alignment
        and
        score <= -7
        and
        trap != "HIGH"
    ):

        signal = "STRONG SELL"

    elif (
        bearish_alignment
        and
        score <= -5
        and
        trap != "HIGH"
    ):

        signal = "SELL"

    else:

        signal = "NO TRADE"

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    price = float(
        last["close"]
    )

    atr = float(
        last["ATR"]
    )

    if (
        not np.isfinite(
            atr
        )
        or
        atr <= 0
    ):

        atr = price * 0.005

    entry = price

    stop = None

    tp1 = None

    tp2 = None

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        stop = min(
            price -
            atr * 1.5,
            support
        )

        if stop >= price:

            stop = (
                price -
                atr * 1.5
            )

        tp1 = (
            price +
            atr * 2.25
        )

        tp2 = (
            price +
            atr * 3.5
        )

    elif signal in [
        "SELL",
        "STRONG SELL"
    ]:

        stop = max(
            price +
            atr * 1.5,
            resistance
        )

        if stop <= price:

            stop = (
                price +
                atr * 1.5
            )

        tp1 = (
            price -
            atr * 2.25
        )

        tp2 = (
            price -
            atr * 3.5
        )

    # =====================================================
    # RISK MANAGEMENT
    # =====================================================

    if stop is not None:

        risk_data = (
            money_management(
                entry,
                stop,
                account,
                risk_percent
            )
        )

        risk_distance = abs(
            entry -
            stop
        )

        reward_distance = abs(
            tp1 -
            entry
        )

        rr = (
            reward_distance /
            risk_distance
            if risk_distance > 0
            else 0
        )

    else:

        risk_data = {
            "risk_amount": 0,
            "position_size": 0,
            "position_value": 0
        }

        rr = 0

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = min(
        95,
        50 +
        abs(score) * 5
    )

    if signal == "NO TRADE":

        confidence = min(
            confidence,
            55
        )

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "timeframe":
            "5m + 15m + 1h",

        "signal":
            signal,

        "confidence":
            round(
                confidence,
                2
            ),

        "score":
            score,

        "current_price":
            round(
                price,
                6
            ),

        "market_structure":
            structure,

        "trendline":
            trend,

        "support":
            round(
                support,
                6
            ),

        "resistance":
            round(
                resistance,
                6
            ),

        "breakout":
            breakout,

        "gap":
            gap,

        "liquidity_trap_risk_estimate":
            trap,

        "5m":
            tf5,

        "15m":
            tf15,

        "1h":
            tf1h,

        "entry":
            (
                round(
                    entry,
                    6
                )
                if signal !=
                "NO TRADE"
                else None
            ),

        "stop_loss":
            (
                round(
                    stop,
                    6
                )
                if stop is not None
                else None
            ),

        "take_profit_1":
            (
                round(
                    tp1,
                    6
                )
                if tp1 is not None
                else None
            ),

        "take_profit_2":
            (
                round(
                    tp2,
                    6
                )
                if tp2 is not None
                else None
            ),

        "risk_reward":
            round(
                rr,
                2
            ),

        "risk_amount":
            risk_data[
                "risk_amount"
            ],

        "position_size":
            risk_data[
                "position_size"
            ],

        "position_value":
            risk_data[
                "position_value"
            ],

        "rsi":
            round(
                float(
                    last["RSI"]
                ),
                2
            ),

        "macd":
            round(
                float(
                    last["MACD"]
                ),
                6
            ),

        "momentum_percent":
            round(
                float(
                    last["MOMENTUM"]
                ),
                4
            ),

        "volume_confirmed":
            bool(
                last["volume"] >
                last["VOL_AVG"]
            ),

        "reasons":
            reasons,

        "warning":
            "Technical analysis only. "
            "No guaranteed prediction. "
            "Liquidation/trap risk is estimated "
            "because direct liquidation data is "
            "not included."
    }


# =========================================================
# 11. LIVE API
# =========================================================

@app.route("/")
def home():

    return (
        "GM AI Trading Bot V2 "
        "Multi-Timeframe System is running!"
    )


@app.route("/analysis/<symbol>")
def analysis(symbol):

    try:

        account = float(
            request.args.get(
                "account",
                DEFAULT_ACCOUNT
            )
        )

        risk = float(
            request.args.get(
                "risk",
                DEFAULT_RISK
            )
        )

        if account <= 0:
            account = DEFAULT_ACCOUNT

        if risk <= 0 or risk > 2:
            risk = DEFAULT_RISK

        return jsonify(
            complete_analysis(
                symbol.upper(),
                account,
                risk
            )
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 12. REALISTIC BACKTEST
# =========================================================

@app.route("/backtest/<symbol>")
def backtest(symbol):

    try:

        limit = int(
            request.args.get(
                "limit",
                1000
            )
        )

        account = float(
            request.args.get(
                "account",
                DEFAULT_ACCOUNT
            )
        )

        risk_percent = float(
            request.args.get(
                "risk",
                DEFAULT_RISK
            )
        )

        limit = min(
            limit,
            1000
        )

        # For a complete MTF backtest we need
        # 5m candles and higher timeframe data.
        df5 = indicators(
            get_data(
                symbol.upper(),
                "5m",
                limit
            )
        )

        # Simplified historical simulation:
        # signal is generated using the 5m data
        # plus resampled 15m and 1h data.

        df15 = (
            df5.set_index(
                pd.to_datetime(
                    df5["time"],
                    unit="ms"
                )
            )
            .resample("15min")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            })
            .dropna()
            .reset_index(
                drop=True
            )
        )

        df1h = (
            df5.set_index(
                pd.to_datetime(
                    df5["time"],
                    unit="ms"
                )
            )
            .resample("1h")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            })
            .dropna()
            .reset_index(
                drop=True
            )
        )

        df15 = indicators(
            df15
        )

        df1h = indicators(
            df1h
        )

        wins = 0
        losses = 0
        trades = 0
        neutral = 0

        equity = account
        peak = account
        max_dd = 0

        total_pnl = 0

        # Need enough data for indicators
        for i in range(
            250,
            len(df5) - 10
        ):

            current = df5.iloc[
                i
            ]

            current_time = pd.to_datetime(
                current["time"],
                unit="ms"
            )

            # Last completed 15m candle
            hist15 = df15[
                df15.index <=
                len(df15)
            ]

            # Find approximate higher TF
            # using current 5m position
            five_minutes = i

            idx15 = min(
                int(
                    five_minutes /
                    3
                ),
                len(df15) - 1
            )

            idx1h = min(
                int(
                    five_minutes /
                    12
                ),
                len(df1h) - 1
            )

            if (
                idx15 < 200
                or
                idx1h < 200
            ):
                continue

            h5 = df5.iloc[
                :i + 1
            ].copy()

            h15 = df15.iloc[
                :idx15 + 1
            ].copy()

            h1h = df1h.iloc[
                :idx1h + 1
            ].copy()

            a5 = timeframe_analysis(
                h5
            )

            a15 = timeframe_analysis(
                h15
            )

            a1h = timeframe_analysis(
                h1h
            )

            bullish = (
                a5["direction"] ==
                "BULLISH"
                and
                a15["direction"] ==
                "BULLISH"
                and
                a1h["direction"] ==
                "BULLISH"
            )

            bearish = (
                a5["direction"] ==
                "BEARISH"
                and
                a15["direction"] ==
                "BEARISH"
                and
                a1h["direction"] ==
                "BEARISH"
            )

            if not bullish and not bearish:

                neutral += 1

                continue

            price = float(
                current["close"]
            )

            atr = float(
                current["ATR"]
            )

            if (
                not np.isfinite(
                    atr
                )
                or atr <= 0
            ):
                continue

            if bullish:

                stop = (
                    price -
                    atr * 1.5
                )

                target = (
                    price +
                    atr * 2.25
                )

                direction = "LONG"

            else:

                stop = (
                    price +
                    atr * 1.5
                )

                target = (
                    price -
                    atr * 2.25
                )

                direction = "SHORT"

            outcome = None
            exit_price = None

            # Check next 10 candles
            for j in range(
                i + 1,
                min(
                    i + 11,
                    len(df5)
                )
            ):

                high = float(
                    df5.iloc[j]["high"]
                )

                low = float(
                    df5.iloc[j]["low"]
                )

                if direction == "LONG":

                    # Conservative:
                    # if both touched same candle,
                    # assume SL first.
                    if low <= stop:

                        outcome = "LOSS"
                        exit_price = stop
                        break

                    if high >= target:

                        outcome = "WIN"
                        exit_price = target
                        break

                else:

                    if high >= stop:

                        outcome = "LOSS"
                        exit_price = stop
                        break

                    if low <= target:

                        outcome = "WIN"
                        exit_price = target
                        break

            if outcome is None:

                exit_price = float(
                    df5.iloc[
                        min(
                            i + 10,
                            len(df5) - 1
                        )
                    ]["close"]
                )

                if direction == "LONG":

                    outcome = (
                        "WIN"
                        if exit_price >
                        price
                        else
                        "LOSS"
                    )

                else:

                    outcome = (
                        "WIN"
                        if exit_price <
                        price
                        else
                        "LOSS"
                    )

            risk_amount = (
                equity *
                risk_percent /
                100
            )

            price_risk = abs(
                price -
                stop
            )

            position_size = (
                risk_amount /
                price_risk
                if price_risk > 0
                else 0
            )

            if direction == "LONG":

                raw_pnl = (
                    exit_price -
                    price
                ) * position_size

            else:

                raw_pnl = (
                    price -
                    exit_price
                ) * position_size

            fees = (
                (
                    price +
                    exit_price
                )
                *
                position_size
                *
                FEE_PER_SIDE
            )

            trade_pnl = (
                raw_pnl -
                fees
            )

            equity += trade_pnl

            total_pnl += trade_pnl

            trades += 1

            if outcome == "WIN":
                wins += 1
            else:
                losses += 1

            peak = max(
                peak,
                equity
            )

            drawdown = (
                (
                    peak -
                    equity
                )
                /
                peak
            ) * 100

            max_dd = max(
                max_dd,
                drawdown
            )

        win_rate = (
            wins /
            trades *
            100
            if trades > 0
            else 0
        )

        return jsonify({

            "status":
                "success",

            "symbol":
                symbol.upper(),

            "timeframe":
                "5m with 15m + 1h confirmation",

            "candles_tested":
                len(df5),

            "total_trades":
                trades,

            "wins":
                wins,

            "losses":
                losses,

            "neutral":
                neutral,

            "win_rate_percent":
                round(
                    win_rate,
                    2
                ),

            "starting_balance":
                round(
                    account,
                    2
                ),

            "estimated_pnl":
                round(
                    total_pnl,
                    2
                ),

            "ending_balance":
                round(
                    equity,
                    2
                ),

            "max_drawdown_percent":
                round(
                    max_dd,
                    2
                ),

            "risk_per_trade_percent":
                risk_percent,

            "note":
                "Historical simulation only. "
                "Fees are estimated. "
                "This does not guarantee future results."

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 13. RUN
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
