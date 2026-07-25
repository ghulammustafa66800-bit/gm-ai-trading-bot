from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MEXC_URL = "https://api.mexc.com/api/v3/klines"


# =========================================================
# SETTINGS
# =========================================================

DEFAULT_RISK_PERCENT = 1.0
DEFAULT_ACCOUNT_SIZE = 1000.0
FEE_PERCENT_PER_SIDE = 0.05


# =========================================================
# 1. GET MARKET DATA
# =========================================================

def get_data(symbol="BTCUSDT", interval="5m", limit=500):

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(int(limit), 1000)
    }

    response = requests.get(
        MEXC_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 100:
        raise Exception("Not enough market data")

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

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(
        drop=True
    )

    return df


# =========================================================
# 2. TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
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

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # MACD
    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # ATR
    high_low = (
        df["high"] -
        df["low"]
    )

    high_close = abs(
        df["high"] -
        df["close"].shift()
    )

    low_close = abs(
        df["low"] -
        df["close"].shift()
    )

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(
        axis=1
    )

    df["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

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

    # Candle body
    df["BODY"] = abs(
        df["close"] -
        df["open"]
    )

    # Candle range
    df["RANGE"] = (
        df["high"] -
        df["low"]
    )

    # Body strength
    df["BODY_RATIO"] = (
        df["BODY"] /
        df["RANGE"].replace(
            0,
            np.nan
        )
    )

    return df


# =========================================================
# 3. SUPPORT / RESISTANCE
# =========================================================

def support_resistance(df):

    recent = df.iloc[
        -51:-1
    ]

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    return support, resistance


# =========================================================
# 4. MARKET STRUCTURE
# =========================================================

def market_structure(df):

    recent = df.tail(
        20
    )

    first_half = recent.iloc[
        :10
    ]

    second_half = recent.iloc[
        10:
    ]

    first_high = (
        first_half["high"]
        .max()
    )

    second_high = (
        second_half["high"]
        .max()
    )

    first_low = (
        first_half["low"]
        .min()
    )

    second_low = (
        second_half["low"]
        .min()
    )

    if (
        second_high >
        first_high
        and
        second_low >
        first_low
    ):
        return "BULLISH"

    if (
        second_high <
        first_high
        and
        second_low <
        first_low
    ):
        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# 5. TRENDLINE ESTIMATE
# =========================================================

def trendline_analysis(df):

    recent = df.tail(
        20
    )

    first_price = float(
        recent["close"].iloc[
            0
        ]
    )

    last_price = float(
        recent["close"].iloc[
            -1
        ]
    )

    change = (
        (
            last_price -
            first_price
        )
        /
        first_price
    ) * 100

    if change > 0.5:

        return "UPTREND"

    if change < -0.5:

        return "DOWNTREND"

    return "FLAT"


# =========================================================
# 6. BREAKOUT / BREAKDOWN
# =========================================================

def breakout_analysis(
    df,
    support,
    resistance
):

    last = df.iloc[
        -1
    ]

    previous = df.iloc[
        -2
    ]

    if (
        last["close"] >
        resistance
        and
        previous["close"] <=
        resistance
    ):

        return "BREAKOUT"

    if (
        last["close"] <
        support
        and
        previous["close"] >=
        support
    ):

        return "BREAKDOWN"

    return "NO_BREAKOUT"


# =========================================================
# 7. GAP DETECTION
# =========================================================

def gap_analysis(df):

    last = df.iloc[
        -1
    ]

    previous = df.iloc[
        -2
    ]

    gap_percent = (
        (
            last["open"] -
            previous["close"]
        )
        /
        previous["close"]
    ) * 100

    if gap_percent > 0.3:

        return "GAP_UP"

    if gap_percent < -0.3:

        return "GAP_DOWN"

    return "NO_GAP"


# =========================================================
# 8. FAKE BREAKOUT / LIQUIDITY TRAP ESTIMATE
# =========================================================

def trap_analysis(
    df,
    support,
    resistance
):

    last = df.iloc[
        -1
    ]

    volume_confirmed = (
        last["volume"] >
        last["VOL_AVG"]
    )

    fake_breakout = (
        last["high"] >
        resistance
        and
        last["close"] <
        resistance
    )

    fake_breakdown = (
        last["low"] <
        support
        and
        last["close"] >
        support
    )

    long_wick = (
        last["RANGE"] >
        0
        and
        last["BODY_RATIO"] <
        0.35
    )

    if (
        fake_breakout
        or
        fake_breakdown
    ):

        if not volume_confirmed:

            return "HIGH"

        return "MEDIUM"

    if long_wick:

        return "MEDIUM"

    return "LOW"


# =========================================================
# 9. MONEY MANAGEMENT
# =========================================================

def money_management(
    entry,
    stop_loss,
    account_size,
    risk_percent
):

    risk_amount = (
        account_size *
        (
            risk_percent /
            100
        )
    )

    price_risk = abs(
        entry -
        stop_loss
    )

    if price_risk <= 0:

        return {
            "risk_amount": 0,
            "position_size": 0,
            "position_value": 0
        }

    position_size = (
        risk_amount /
        price_risk
    )

    position_value = (
        position_size *
        entry
    )

    return {

        "risk_amount": round(
            risk_amount,
            4
        ),

        "position_size": round(
            position_size,
            8
        ),

        "position_value": round(
            position_value,
            4
        )
    }


# =========================================================
# 10. COMBINED ANALYSIS
# =========================================================

def generate_analysis(
    df,
    account_size=1000,
    risk_percent=1
):

    last = df.iloc[
        -1
    ]

    support, resistance = (
        support_resistance(
            df
        )
    )

    structure = (
        market_structure(
            df
        )
    )

    trendline = (
        trendline_analysis(
            df
        )
    )

    breakout = (
        breakout_analysis(
            df,
            support,
            resistance
        )
    )

    gap = (
        gap_analysis(
            df
        )
    )

    trap_risk = (
        trap_analysis(
            df,
            support,
            resistance
        )
    )

    score = 0

    reasons = []

    # =====================================================
    # EMA TREND
    # =====================================================

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):

        score += 3

        reasons.append(
            "EMA trend bullish"
        )

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):

        score -= 3

        reasons.append(
            "EMA trend bearish"
        )

    # =====================================================
    # RSI
    # =====================================================

    if (
        last["RSI"] >
        50
        and
        last["RSI"] <
        70
    ):

        score += 1

        reasons.append(
            "RSI bullish zone"
        )

    elif (
        last["RSI"] <
        50
        and
        last["RSI"] >
        30
    ):

        score -= 1

        reasons.append(
            "RSI bearish zone"
        )

    elif (
        last["RSI"] >=
        70
    ):

        reasons.append(
            "RSI overbought"
        )

    elif (
        last["RSI"] <=
        30
    ):

        reasons.append(
            "RSI oversold"
        )

    # =====================================================
    # MACD
    # =====================================================

    if (
        last["MACD"] >
        last["MACD_SIGNAL"]
    ):

        score += 2

        reasons.append(
            "MACD bullish"
        )

    else:

        score -= 2

        reasons.append(
            "MACD bearish"
        )

    # =====================================================
    # MOMENTUM
    # =====================================================

    if (
        last["MOMENTUM"] >
        0.2
    ):

        score += 1

        reasons.append(
            "Positive momentum"
        )

    elif (
        last["MOMENTUM"] <
        -0.2
    ):

        score -= 1

        reasons.append(
            "Negative momentum"
        )

    # =====================================================
    # MARKET STRUCTURE
    # =====================================================

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

    # =====================================================
    # TRENDLINE
    # =====================================================

    if trendline == "UPTREND":

        score += 1

        reasons.append(
            "Price trendline rising"
        )

    elif trendline == "DOWNTREND":

        score -= 1

        reasons.append(
            "Price trendline falling"
        )

    # =====================================================
    # BREAKOUT
    # =====================================================

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

    # =====================================================
    # VOLUME
    # =====================================================

    volume_confirmed = (
        last["volume"] >
        last["VOL_AVG"]
    )

    if volume_confirmed:

        reasons.append(
            "Volume confirmation"
        )

        if score > 0:

            score += 1

        elif score < 0:

            score -= 1

    # =====================================================
    # TRAP FILTER
    # =====================================================

    if trap_risk == "HIGH":

        score = int(
            score *
            0.5
        )

        reasons.append(
            "High trap risk"
        )

    elif trap_risk == "MEDIUM":

        reasons.append(
            "Medium trap risk"
        )

    # =====================================================
    # GAP
    # =====================================================

    if gap == "GAP_UP":

        reasons.append(
            "Gap up detected"
        )

    elif gap == "GAP_DOWN":

        reasons.append(
            "Gap down detected"
        )

    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    if score >= 7:

        signal = "STRONG BUY"

    elif score >= 4:

        signal = "BUY"

    elif score <= -7:

        signal = "STRONG SELL"

    elif score <= -4:

        signal = "SELL"

    else:

        signal = "WAIT"

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = min(
        95,
        50 +
        abs(score) *
        4
    )

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    price = float(
        last["close"]
    )

    atr = float(
        last["ATR"]
    )

    if not np.isfinite(
        atr
    ) or atr <= 0:

        atr = (
            price *
            0.005
        )

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        entry = price

        stop_loss = max(
            price -
            (
                atr *
                1.5
            ),
            support
        )

        if stop_loss >= price:

            stop_loss = (
                price -
                (
                    atr *
                    1.5
                )
            )

        take_profit_1 = (
            price +
            (
                atr *
                2
            )
        )

        take_profit_2 = (
            price +
            (
                atr *
                3
            )
        )

    elif signal in [
        "SELL",
        "STRONG SELL"
    ]:

        entry = price

        stop_loss = min(
            price +
            (
                atr *
                1.5
            ),
            resistance
        )

        if stop_loss <= price:

            stop_loss = (
                price +
                (
                    atr *
                    1.5
                )
            )

        take_profit_1 = (
            price -
            (
                atr *
                2
            )
        )

        take_profit_2 = (
            price -
            (
                atr *
                3
            )
        )

    else:

        entry = price

        stop_loss = None

        take_profit_1 = None

        take_profit_2 = None

    # =====================================================
    # MONEY MANAGEMENT
    # =====================================================

    if stop_loss is not None:

        risk_data = (
            money_management(
                entry,
                stop_loss,
                account_size,
                risk_percent
            )
        )

        risk_distance = abs(
            entry -
            stop_loss
        )

        reward_distance = abs(
            take_profit_1 -
            entry
        )

        if risk_distance > 0:

            risk_reward = (
                reward_distance /
                risk_distance
            )

        else:

            risk_reward = 0

    else:

        risk_data = {
            "risk_amount": 0,
            "position_size": 0,
            "position_value": 0
        }

        risk_reward = 0

    return {

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
            trendline,

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

        "trap_risk_estimate":
            trap_risk,

        "entry":
            round(
                entry,
                6
            ),

        "stop_loss":
            (
                round(
                    stop_loss,
                    6
                )
                if stop_loss
                else None
            ),

        "take_profit_1":
            (
                round(
                    take_profit_1,
                    6
                )
                if take_profit_1
                else None
            ),

        "take_profit_2":
            (
                round(
                    take_profit_2,
                    6
                )
                if take_profit_2
                else None
            ),

        "risk_reward":
            round(
                risk_reward,
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

        "ema20":
            round(
                float(
                    last["EMA20"]
                ),
                6
            ),

        "ema50":
            round(
                float(
                    last["EMA50"]
                ),
                6
            ),

        "ema200":
            round(
                float(
                    last["EMA200"]
                ),
                6
            ),

        "macd":
            round(
                float(
                    last["MACD"]
                ),
                6
            ),

        "atr":
            round(
                atr,
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
                volume_confirmed
            ),

        "reasons":
            reasons
    }


# =========================================================
# 11. LIVE ANALYSIS
# =========================================================

@app.route("/")
def home():

    return (
        "GM AI Trading Bot v6 "
        "All-In-One System is running!"
    )


@app.route("/analysis/<symbol>")
def analysis(symbol):

    try:

        account_size = float(
            request.args.get(
                "account",
                DEFAULT_ACCOUNT_SIZE
            )
        )

        risk_percent = float(
            request.args.get(
                "risk",
                DEFAULT_RISK_PERCENT
            )
        )

        if account_size <= 0:

            account_size = (
                DEFAULT_ACCOUNT_SIZE
            )

        if (
            risk_percent <= 0
            or
            risk_percent > 5
        ):

            risk_percent = (
                DEFAULT_RISK_PERCENT
            )

        df = get_data(
            symbol.upper(),
            "5m",
            500
        )

        df = calculate_indicators(
            df
        )

        result = generate_analysis(
            df,
            account_size,
            risk_percent
        )

        result["status"] = (
            "success"
        )

        result["symbol"] = (
            symbol.upper()
        )

        result["timeframe"] = (
            "5m"
        )

        result["account_size"] = (
            account_size
        )

        result["risk_percent"] = (
            risk_percent
        )

        result["warning"] = (
            "Analysis only. "
            "No guaranteed prediction. "
            "Trap/liquidation risk is estimated "
            "from price and volume because direct "
            "liquidation data is not included."
        )

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 12. IMPROVED BACKTEST
# =========================================================

@app.route("/backtest/<symbol>")
def backtest(symbol):

    try:

        limit = int(
            request.args.get(
                "limit",
                500
            )
        )

        account_size = float(
            request.args.get(
                "account",
                DEFAULT_ACCOUNT_SIZE
            )
        )

        risk_percent = float(
            request.args.get(
                "risk",
                DEFAULT_RISK_PERCENT
            )
        )

        limit = min(
            limit,
            1000
        )

        df = get_data(
            symbol.upper(),
            "5m",
            limit
        )

        df = calculate_indicators(
            df
        )

        wins = 0

        losses = 0

        total_trades = 0

        neutral = 0

        pnl = 0.0

        equity = account_size

        peak_equity = (
            account_size
        )

        max_drawdown = 0.0

        results = []

        for i in range(
            200,
            len(df) - 1
        ):

            historical = (
                df.iloc[
                    :i + 1
                ].copy()
            )

            analysis_result = (
                generate_analysis(
                    historical,
                    account_size,
                    risk_percent
                )
            )

            signal = (
                analysis_result[
                    "signal"
                ]
            )

            if signal not in [
                "BUY",
                "STRONG BUY",
                "SELL",
                "STRONG SELL"
            ]:

                neutral += 1

                continue

            entry = float(
                df.iloc[i]["close"]
            )

            atr = float(
                df.iloc[i]["ATR"]
            )

            if (
                not np.isfinite(
                    atr
                )
                or
                atr <= 0
            ):

                continue

            if signal in [
                "BUY",
                "STRONG BUY"
            ]:

                stop = (
                    entry -
                    (
                        atr *
                        1.5
                    )
                )

                target = (
                    entry +
                    (
                        atr *
                        2
                    )
                )

                direction = (
                    "LONG"
                )

            else:

                stop = (
                    entry +
                    (
                        atr *
                        1.5
                    )
                )

                target = (
                    entry -
                    (
                        atr *
                        2
                    )
                )

                direction = (
                    "SHORT"
                )

            outcome = (
                "OPEN"
            )

            exit_price = None

            # Check next 3 candles
            end_index = min(
                i + 4,
                len(df)
            )

            for j in range(
                i + 1,
                end_index
            ):

                candle_high = float(
                    df.iloc[j]["high"]
                )

                candle_low = float(
                    df.iloc[j]["low"]
                )

                if direction == "LONG":

                    if candle_low <= stop:

                        outcome = (
                            "LOSS"
                        )

                        exit_price = (
                            stop
                        )

                        break

                    if candle_high >= target:

                        outcome = (
                            "WIN"
                        )

                        exit_price = (
                            target
                        )

                        break

                else:

                    if candle_high >= stop:

                        outcome = (
                            "LOSS"
                        )

                        exit_price = (
                            stop
                        )

                        break

                    if candle_low <= target:

                        outcome = (
                            "WIN"
                        )

                        exit_price = (
                            target
                        )

                        break

            if outcome == "OPEN":

                exit_price = float(
                    df.iloc[
                        end_index - 1
                    ]["close"]
                )

                if direction == "LONG":

                    outcome = (
                        "WIN"
                        if exit_price >
                        entry
                        else
                        "LOSS"
                    )

                else:

                    outcome = (
                        "WIN"
                        if exit_price <
                        entry
                        else
                        "LOSS"
                    )

            total_trades += 1

            risk_amount = (
                equity *
                (
                    risk_percent /
                    100
                )
            )

            if direction == "LONG":

                price_change = (
                    exit_price -
                    entry
                )

            else:

                price_change = (
                    entry -
                    exit_price
                )

            price_risk = abs(
                entry -
                stop
            )

            if price_risk > 0:

                position_size = (
                    risk_amount /
                    price_risk
                )

            else:

                position_size = 0

            trade_pnl = (
                price_change *
                position_size
            )

            # Approximate trading fees
            fee = (
                (
                    entry +
                    exit_price
                )
                *
                position_size
                *
                (
                    FEE_PERCENT_PER_SIDE /
                    100
                )
            )

            trade_pnl -= fee

            pnl += trade_pnl

            equity += trade_pnl

            if outcome == "WIN":

                wins += 1

            else:

                losses += 1

            peak_equity = max(
                peak_equity,
                equity
            )

            drawdown = (
                (
                    peak_equity -
                    equity
                )
                /
                peak_equity
            ) * 100

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

            results.append({

                "signal":
                    signal,

                "direction":
                    direction,

                "outcome":
                    outcome,

                "pnl":
                    round(
                        trade_pnl,
                        4
                    )

            })

        if total_trades > 0:

            win_rate = (
                wins /
                total_trades
            ) * 100

        else:

            win_rate = 0

        return jsonify({

            "status":
                "success",

            "symbol":
                symbol.upper(),

            "timeframe":
                "5m",

            "candles_used":
                len(df),

            "total_trades":
                total_trades,

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
                    account_size,
                    2
                ),

            "estimated_pnl":
                round(
                    pnl,
                    2
                ),

            "ending_balance":
                round(
                    equity,
                    2
                ),

            "max_drawdown_percent":
                round(
                    max_drawdown,
                    2
                ),

            "risk_per_trade_percent":
                risk_percent,

            "note":
                "Historical simulation only. "
                "Fees are approximate. "
                "Results do not guarantee future performance."

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
