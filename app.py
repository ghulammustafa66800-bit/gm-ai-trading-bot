from flask import Flask, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

app = Flask(__name__)

# =========================================================
# GM SMART SCALPER V4
# MEXC FUTURES + TECHNICAL + ORDER FLOW
#
# ANALYSIS ONLY
# NO REAL ORDER EXECUTION
#
# FLOW:
# HTF CONTEXT
# -> MARKET BIAS
# -> MARKET STRUCTURE / BOS
# -> KEY LEVELS
# -> LIQUIDITY SWEEP
# -> BREAKOUT / RETEST
# -> ORDER BOOK
# -> BUYER / SELLER PRESSURE
# -> EXECUTED TRADE FLOW
# -> DELTA
# -> AGGRESSION
# -> ABSORPTION / FAKE MOVE FILTER
# -> RISK / R:R
# -> FINAL SIGNAL
# =========================================================

MEXC_BASE = "https://contract.mexc.com"

DEFAULT_SYMBOL = "BTC_USDT"
DEFAULT_INTERVAL = "Min5"

MIN_CANDLES = 250

MIN_SCORE = 10

SL_ATR = 1.5

TP1_RR = 1.5
TP2_RR = 2.5
TP3_RR = 3.5

MAX_HOLD_CANDLES = 20

ORDERBOOK_LEVELS = 20


# =========================================================
# HELPERS
# =========================================================

def normalize_symbol(symbol):

    symbol = (
        str(symbol)
        .upper()
        .strip()
        .replace("/", "_")
        .replace("-", "_")
    )

    if symbol.endswith("USDT") and "_" not in symbol:

        symbol = symbol[:-4] + "_USDT"

    return symbol


def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isnan(value) or np.isinf(value):

            return default

        return value

    except Exception:

        return default


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# =========================================================
# MEXC FUTURES REQUEST
# =========================================================

def mexc_get(
    path,
    params=None
):

    url = (
        MEXC_BASE +
        path
    )

    response = requests.get(

        url,

        params=params,

        timeout=20
    )

    if response.status_code != 200:

        raise Exception(

            f"MEXC HTTP error "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    if not data.get(
        "success",
        False
    ):

        raise Exception(

            f"MEXC API error: "
            f"{data}"
        )

    return data.get(
        "data"
    )


# =========================================================
# CONTRACT INFO
# =========================================================

def get_contract_info(symbol):

    symbol = normalize_symbol(
        symbol
    )

    data = mexc_get(

        "/api/v1/contract/detail",

        {
            "symbol":
                symbol
        }

    )

    if isinstance(
        data,
        list
    ):

        if len(data) == 0:

            raise Exception(
                "Contract not found"
            )

        return data[0]

    return data


# =========================================================
# KLINE DATA
# =========================================================

def interval_to_minutes(interval):

    mapping = {

        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,

        "Min1": 1,
        "Min5": 5,
        "Min15": 15,
        "Min30": 30,

        "Min60": 60,
        "Hour1": 60,

        "Hour4": 240
    }

    return mapping.get(
        interval,
        5
    )


def normalize_interval(interval):

    mapping = {

        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "30m": "Min30",

        "1h": "Min60",
        "60m": "Min60",

        "4h": "Hour4"
    }

    return mapping.get(

        interval,

        interval
    )


def get_data(

    symbol,

    interval="5m",

    limit=1000

):

    symbol = normalize_symbol(
        symbol
    )

    interval = normalize_interval(
        interval
    )

    minutes = interval_to_minutes(
        interval
    )

    try:

        limit = int(
            limit
        )

    except:

        limit = 1000

    limit = max(

        MIN_CANDLES,

        min(
            limit,
            1000
        )
    )

    # Request enough historical time
    # for approximately "limit" candles.

    end = int(

        datetime.now(
            timezone.utc
        ).timestamp()
    )

    start = (

        end -

        (
            limit *

            minutes *

            60
        )
    )

    data = mexc_get(

        f"/api/v1/contract/kline/{symbol}",

        {

            "interval":
                interval,

            "start":
                start,

            "end":
                end

        }

    )

    if not isinstance(
        data,
        dict
    ):

        raise Exception(
            "Unexpected Kline response"
        )

    times = data.get(
        "time",
        []
    )

    opens = data.get(
        "open",
        []
    )

    highs = data.get(
        "high",
        []
    )

    lows = data.get(
        "low",
        []
    )

    closes = data.get(
        "close",
        []
    )

    volumes = data.get(
        "vol",
        []
    )

    if not volumes:

        volumes = data.get(
            "volume",
            []
        )

    n = min(

        len(times),

        len(opens),

        len(highs),

        len(lows),

        len(closes),

        len(volumes)

    )

    if n < MIN_CANDLES:

        raise Exception(

            f"Not enough candles. "
            f"Received {n}, "
            f"required {MIN_CANDLES}"
        )

    df = pd.DataFrame({

        "time":
            times[:n],

        "open":
            opens[:n],

        "high":
            highs[:n],

        "low":
            lows[:n],

        "close":
            closes[:n],

        "volume":
            volumes[:n]

    })

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

    df = (

        df

        .dropna()

        .sort_values(
            "time"
        )

        .drop_duplicates(
            "time"
        )

        .reset_index(
            drop=True
        )

    )

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

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

    delta = df[
        "close"
    ].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (

        gain

        .ewm(

            alpha=1 / 14,

            adjust=False

        )

        .mean()
    )

    avg_loss = (

        loss

        .ewm(

            alpha=1 / 14,

            adjust=False

        )

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

            (
                1 +

                rs
            )

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

    df["MACD_HIST"] = (

        df["MACD"] -

        df["MACD_SIGNAL"]
    )

    # ATR

    high_low = (

        df["high"] -

        df["low"]
    )

    high_close = (

        df["high"] -

        df["close"].shift()

    ).abs()

    low_close = (

        df["low"] -

        df["close"].shift()

    ).abs()

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

        .ewm(

            alpha=1 / 14,

            adjust=False

        )

        .mean()
    )

    # Volume

    df["VOL_AVG"] = (

        df["volume"]

        .rolling(
            20
        )

        .mean()
    )

    df["VOLUME_RATIO"] = (

        df["volume"] /

        df["VOL_AVG"].replace(

            0,

            np.nan
        )
    )

    # Momentum

    df["MOM_5"] = (

        df["close"]

        .pct_change(
            5
        )

        * 100
    )

    df["MOM_15"] = (

        df["close"]

        .pct_change(
            15
        )

        * 100
    )

    # Candle structure

    df["BODY"] = (

        df["close"] -

        df["open"]

    ).abs()

    df["RANGE"] = (

        df["high"] -

        df["low"]
    )

    df["UPPER_WICK"] = (

        df["high"] -

        df[
            [
                "open",
                "close"
            ]
        ].max(
            axis=1
        )
    )

    df["LOWER_WICK"] = (

        df[
            [
                "open",
                "close"
            ]
        ].min(
            axis=1
        )

        -

        df["low"]
    )

    return df


# =========================================================
# MARKET BIAS
# =========================================================

def get_market_bias(df):

    last = df.iloc[-1]

    score = 0

    if (

        last["EMA20"]

        >

        last["EMA50"]

        >

        last["EMA200"]

    ):

        score += 3

    elif (

        last["EMA20"]

        <

        last["EMA50"]

        <

        last["EMA200"]

    ):

        score -= 3

    elif (

        last["EMA20"]

        >

        last["EMA50"]

    ):

        score += 1

    elif (

        last["EMA20"]

        <

        last["EMA50"]

    ):

        score -= 1

    if (

        last["MACD"]

        >

        last["MACD_SIGNAL"]

    ):

        score += 1

    else:

        score -= 1

    if (

        safe_float(
            last["MOM_15"]
        )

        >

        0

    ):

        score += 1

    elif (

        safe_float(
            last["MOM_15"]
        )

        <

        0

    ):

        score -= 1

    if score >= 3:

        bias = "BULLISH"

    elif score <= -3:

        bias = "BEARISH"

    else:

        bias = "NEUTRAL"

    return {

        "bias":
            bias,

        "score":
            score
    }


# =========================================================
# MARKET STRUCTURE + BOS
# =========================================================

def get_market_structure(df):

    data = df.tail(
        60
    )

    first = data.iloc[
        :30
    ]

    second = data.iloc[
        30:
    ]

    first_high = first[
        "high"
    ].max()

    second_high = second[
        "high"
    ].max()

    first_low = first[
        "low"
    ].min()

    second_low = second[
        "low"
    ].min()

    if (

        second_high > first_high

        and

        second_low > first_low

    ):

        return "BULLISH"

    if (

        second_high < first_high

        and

        second_low < first_low

    ):

        return "BEARISH"

    return "SIDEWAYS"


def detect_bos(df):

    if len(df) < 25:

        return "NONE"

    previous = df.iloc[
        -21:-1
    ]

    last = df.iloc[
        -1
    ]

    swing_high = previous[
        "high"
    ].max()

    swing_low = previous[
        "low"
    ].min()

    if (

        last["close"]

        >

        swing_high

    ):

        return "BULLISH_BOS"

    if (

        last["close"]

        <

        swing_low

    ):

        return "BEARISH_BOS"

    return "NONE"


# =========================================================
# KEY LEVELS
# =========================================================

def get_key_levels(df):

    history = (

        df.iloc[
            :-1
        ]

        .tail(
            100
        )
    )

    return {

        "support":
            safe_float(

                history[
                    "low"
                ].min()

            ),

        "resistance":
            safe_float(

                history[
                    "high"
                ].max()

            )

    }


# =========================================================
# LIQUIDITY SWEEP
# =========================================================

def detect_liquidity_sweep(

    df,

    levels

):

    last = df.iloc[
        -1
    ]

    support = levels[
        "support"
    ]

    resistance = levels[
        "resistance"
    ]

    bullish = (

        last["low"]

        <

        support

        and

        last["close"]

        >

        support

        and

        last["close"]

        >

        last["open"]

    )

    bearish = (

        last["high"]

        >

        resistance

        and

        last["close"]

        <

        resistance

        and

        last["close"]

        <

        last["open"]

    )

    if bullish:

        return "BULLISH_SWEEP"

    if bearish:

        return "BEARISH_SWEEP"

    return "NONE"


# =========================================================
# BREAKOUT
# =========================================================

def detect_breakout(

    df,

    levels

):

    last = df.iloc[
        -1
    ]

    support = levels[
        "support"
    ]

    resistance = levels[
        "resistance"
    ]

    if (

        last["close"]

        >

        resistance

        and

        last["open"]

        <=

        resistance

    ):

        return "BULLISH_BREAKOUT"

    if (

        last["close"]

        <

        support

        and

        last["open"]

        >=

        support

    ):

        return "BEARISH_BREAKDOWN"

    return "NONE"


# =========================================================
# RETEST
# =========================================================

def detect_retest(

    df,

    levels

):

    if len(df) < 4:

        return "NONE"

    current = df.iloc[
        -1
    ]

    previous = df.iloc[
        -2
    ]

    support = levels[
        "support"
    ]

    resistance = levels[
        "resistance"
    ]

    bullish = (

        previous["close"]

        >

        resistance

        and

        current["low"]

        <=

        resistance * 1.001

        and

        current["close"]

        >

        resistance

    )

    bearish = (

        previous["close"]

        <

        support

        and

        current["high"]

        >=

        support * 0.999

        and

        current["close"]

        <

        support

    )

    if bullish:

        return "BULLISH_RETEST"

    if bearish:

        return "BEARISH_RETEST"

    return "NONE"


# =========================================================
# CANDLE CONFIRMATION
# =========================================================

def candle_confirmation(df):

    last = df.iloc[
        -1
    ]

    body = safe_float(
        last["BODY"]
    )

    candle_range = safe_float(
        last["RANGE"]
    )

    if candle_range <= 0:

        return "NONE"

    if (

        last["close"]

        >

        last["open"]

        and

        last["LOWER_WICK"]

        >

        body * 1.2

    ):

        return "BULLISH_REJECTION"

    if (

        last["close"]

        <

        last["open"]

        and

        last["UPPER_WICK"]

        >

        body * 1.2

    ):

        return "BEARISH_REJECTION"

    if (

        last["close"]

        >

        last["open"]

        and

        body / candle_range

        >

        0.60

    ):

        return "STRONG_BULLISH"

    if (

        last["close"]

        <

        last["open"]

        and

        body / candle_range

        >

        0.60

    ):

        return "STRONG_BEARISH"

    return "NONE"


# =========================================================
# MTF
# =========================================================

def timeframe_bias(df):

    last = df.iloc[
        -1
    ]

    if (

        last["EMA20"]

        >

        last["EMA50"]

        >

        last["EMA200"]

    ):

        return "BULLISH"

    if (

        last["EMA20"]

        <

        last["EMA50"]

        <

        last["EMA200"]

    ):

        return "BEARISH"

    return "NEUTRAL"


def get_mtf_confirmation(symbol):

    intervals = [

        "5m",
        "15m",
        "1h",
        "4h"

    ]

    results = {}

    for interval in intervals:

        try:

            data = get_data(

                symbol,

                interval,

                300
            )

            data = add_indicators(
                data
            )

            results[
                interval
            ] = timeframe_bias(
                data
            )

        except Exception as e:

            results[
                interval
            ] = "UNAVAILABLE"

    bullish = list(

        results.values()

    ).count(
        "BULLISH"
    )

    bearish = list(

        results.values()

    ).count(
        "BEARISH"
    )

    if bullish >= 3:

        confirmation = "BULLISH"

    elif bearish >= 3:

        confirmation = "BEARISH"

    else:

        confirmation = "MIXED"

    return {

        "timeframes":
            results,

        "confirmation":
            confirmation,

        "bullish_count":
            bullish,

        "bearish_count":
            bearish
    }


# =========================================================
# ORDER BOOK
# =========================================================

def get_orderbook(

    symbol,

    levels=20

):

    symbol = normalize_symbol(
        symbol
    )

    data = mexc_get(

        f"/api/v1/contract/depth/{symbol}",

        {

            "limit":
                levels

        }

    )

    if not isinstance(
        data,
        dict
    ):

        raise Exception(
            "Invalid order book response"
        )

    bids = data.get(
        "bids",
        []
    )

    asks = data.get(
        "asks",
        []
    )

    bids_clean = []

    asks_clean = []

    for row in bids:

        if len(row) >= 2:

            bids_clean.append({

                "price":
                    safe_float(
                        row[0]
                    ),

                "volume":
                    safe_float(
                        row[1]
                    ),

                "orders":
                    (
                        int(
                            row[2]
                        )

                        if len(row) >= 3

                        else None
                    )

            })

    for row in asks:

        if len(row) >= 2:

            asks_clean.append({

                "price":
                    safe_float(
                        row[0]
                    ),

                "volume":
                    safe_float(
                        row[1]
                    ),

                "orders":
                    (
                        int(
                            row[2]
                        )

                        if len(row) >= 3

                        else None
                    )

            })

    bid_volume = sum(

        x["volume"]

        for x in bids_clean
    )

    ask_volume = sum(

        x["volume"]

        for x in asks_clean
    )

    total = (

        bid_volume +

        ask_volume
    )

    if total > 0:

        imbalance = (

            (
                bid_volume -

                ask_volume
            )

            /

            total

        ) * 100

        buyer_pressure = (

            bid_volume /

            total

        ) * 100

        seller_pressure = (

            ask_volume /

            total

        ) * 100

    else:

        imbalance = 0

        buyer_pressure = 50

        seller_pressure = 50

    best_bid = (

        bids_clean[0]["price"]

        if bids_clean

        else None
    )

    best_ask = (

        asks_clean[0]["price"]

        if asks_clean

        else None
    )

    bid_wall = (

        max(

            bids_clean,

            key=lambda x:
            x["volume"]

        )

        if bids_clean

        else None
    )

    ask_wall = (

        max(

            asks_clean,

            key=lambda x:
            x["volume"]

        )

        if asks_clean

        else None
    )

    if imbalance >= 15:

        pressure = "BUYER_DOMINANT"

    elif imbalance <= -15:

        pressure = "SELLER_DOMINANT"

    else:

        pressure = "BALANCED"

    return {

        "status":
            "READY",

        "symbol":
            symbol,

        "best_bid":
            best_bid,

        "best_ask":
            best_ask,

        "bid_volume":
            round(
                bid_volume,
                4
            ),

        "ask_volume":
            round(
                ask_volume,
                4
            ),

        "imbalance_percent":
            round(
                imbalance,
                2
            ),

        "buyer_pressure_percent":
            round(
                buyer_pressure,
                2
            ),

        "seller_pressure_percent":
            round(
                seller_pressure,
                2
            ),

        "pressure":
            pressure,

        "largest_bid_wall":
            bid_wall,

        "largest_ask_wall":
            ask_wall,

        "timestamp":
            utc_now()

    }


# =========================================================
# RECENT EXECUTED DEALS
# =========================================================

def get_recent_deals(symbol):

    symbol = normalize_symbol(
        symbol
    )

    data = mexc_get(

        f"/api/v1/contract/deals/{symbol}",

        None

    )

    if not isinstance(
        data,
        list
    ):

        return {

            "status":
                "UNAVAILABLE",

            "message":
                "No recent deal data"

        }

    buy_volume = 0.0

    sell_volume = 0.0

    buy_count = 0

    sell_count = 0

    trades = []

    for row in data:

        if not isinstance(
            row,
            dict
        ):

            continue

        price = safe_float(
            row.get(
                "p",
                row.get(
                    "price",
                    0
                )
            )
        )

        volume = safe_float(
            row.get(
                "v",
                row.get(
                    "vol",
                    row.get(
                        "volume",
                        0
                    )
                )
            )
        )

        side = row.get(
            "T",
            row.get(
                "side"
            )
        )

        # MEXC contract docs:
        # T=1 buy / purchase
        # T=2 sell

        if str(side) == "1":

            buy_volume += volume

            buy_count += 1

            side_name = "BUY"

        elif str(side) == "2":

            sell_volume += volume

            sell_count += 1

            side_name = "SELL"

        else:

            side_name = "UNKNOWN"

        trades.append({

            "price":
                price,

            "volume":
                volume,

            "side":
                side_name

        })

    total = (

        buy_volume +

        sell_volume
    )

    if total > 0:

        delta = (

            buy_volume -

            sell_volume
        )

        delta_percent = (

            delta /

            total

        ) * 100

    else:

        delta = 0

        delta_percent = 0

    if delta_percent >= 15:

        aggression = "BUYER_AGGRESSION"

    elif delta_percent <= -15:

        aggression = "SELLER_AGGRESSION"

    else:

        aggression = "BALANCED"

    return {

        "status":
            "READY",

        "buy_volume":
            round(
                buy_volume,
                4
            ),

        "sell_volume":
            round(
                sell_volume,
                4
            ),

        "buy_count":
            buy_count,

        "sell_count":
            sell_count,

        "delta":
            round(
                delta,
                4
            ),

        "delta_percent":
            round(
                delta_percent,
                2
            ),

        "aggression":
            aggression,

        "recent_trades":
            trades[-20:]

    }


# =========================================================
# FUNDING
# =========================================================

def get_funding_rate(symbol):

    symbol = normalize_symbol(
        symbol
    )

    try:

        data = mexc_get(

            f"/api/v1/contract/funding_rate/{symbol}"

        )

        return {

            "status":
                "READY",

            "funding_rate":
                safe_float(

                    data.get(
                        "fundingRate"
                    )

                ),

            "funding_rate_percent":
                round(

                    safe_float(

                        data.get(
                            "fundingRate"
                        )

                    ) * 100,

                    5
                ),

            "next_settle_time":
                data.get(
                    "nextSettleTime"
                )

        }

    except Exception as e:

        return {

            "status":
                "UNAVAILABLE",

            "message":
                str(e)

        }


# =========================================================
# ORDER FLOW ANALYSIS
# =========================================================

def analyze_order_flow(

    orderbook,

    deals

):

    score = 0

    reasons = []

    # Order book

    ob_pressure = orderbook.get(
        "pressure",
        "BALANCED"
    )

    # Executed flow

    aggression = deals.get(
        "aggression",
        "BALANCED"
    )

    # Buyer side

    if (

        ob_pressure == "BUYER_DOMINANT"

        and

        aggression == "BUYER_AGGRESSION"

    ):

        score += 4

        reasons.append(

            "Order book and executed flow both support buyers"
        )

    elif (

        ob_pressure == "SELLER_DOMINANT"

        and

        aggression == "SELLER_AGGRESSION"

    ):

        score -= 4

        reasons.append(

            "Order book and executed flow both support sellers"
        )

    # Divergence / possible absorption

    if (

        ob_pressure == "BUYER_DOMINANT"

        and

        aggression == "SELLER_AGGRESSION"

    ):

        score -= 2

        reasons.append(

            "Buyer liquidity visible but executed selling suggests possible absorption"
        )

    if (

        ob_pressure == "SELLER_DOMINANT"

        and

        aggression == "BUYER_AGGRESSION"

    ):

        score += 2

        reasons.append(

            "Seller liquidity visible but executed buying suggests possible absorption"
        )

    return {

        "score":
            score,

        "reasons":
            reasons,

        "orderbook_pressure":
            ob_pressure,

        "executed_aggression":
            aggression

    }


# =========================================================
# FINAL SIGNAL ENGINE
# =========================================================

def generate_signal(

    df,

    symbol,

    mtf,

    orderbook,

    deals,

    funding

):

    last = df.iloc[
        -1
    ]

    bias = get_market_bias(
        df
    )

    structure = get_market_structure(
        df
    )

    bos = detect_bos(
        df
    )

    levels = get_key_levels(
        df
    )

    sweep = detect_liquidity_sweep(

        df,

        levels
    )

    breakout = detect_breakout(

        df,

        levels
    )

    retest = detect_retest(

        df,

        levels
    )

    candle = candle_confirmation(
        df
    )

    rsi = safe_float(
        last["RSI"],
        50
    )

    atr = safe_float(
        last["ATR"]
    )

    price = safe_float(
        last["close"]
    )

    score = 0

    reasons = []

    # -----------------------------------------------------
    # CONTEXT / BIAS
    # -----------------------------------------------------

    if bias["bias"] == "BULLISH":

        score += 2

        reasons.append(
            "Market context is bullish"
        )

    elif bias["bias"] == "BEARISH":

        score -= 2

        reasons.append(
            "Market context is bearish"
        )

    # -----------------------------------------------------
    # STRUCTURE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # BOS
    # -----------------------------------------------------

    if bos == "BULLISH_BOS":

        score += 3

        reasons.append(
            "Bullish break of structure"
        )

    elif bos == "BEARISH_BOS":

        score -= 3

        reasons.append(
            "Bearish break of structure"
        )

    # -----------------------------------------------------
    # LIQUIDITY
    # -----------------------------------------------------

    if sweep == "BULLISH_SWEEP":

        score += 3

        reasons.append(
            "Bullish liquidity sweep"
        )

    elif sweep == "BEARISH_SWEEP":

        score -= 3

        reasons.append(
            "Bearish liquidity sweep"
        )

    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    if breakout == "BULLISH_BREAKOUT":

        score += 2

        reasons.append(
            "Bullish key-level breakout"
        )

    elif breakout == "BEARISH_BREAKDOWN":

        score -= 2

        reasons.append(
            "Bearish key-level breakdown"
        )

    # -----------------------------------------------------
    # RETEST
    # -----------------------------------------------------

    if retest == "BULLISH_RETEST":

        score += 3

        reasons.append(
            "Bullish breakout retest"
        )

    elif retest == "BEARISH_RETEST":

        score -= 3

        reasons.append(
            "Bearish breakdown retest"
        )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    volume_ratio = safe_float(

        last[
            "VOLUME_RATIO"
        ],

        1
    )

    if volume_ratio >= 1.2:

        if score > 0:

            score += 2

            reasons.append(
                "Volume confirms bullish participation"
            )

        elif score < 0:

            score -= 2

            reasons.append(
                "Volume confirms bearish participation"
            )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 50 < rsi < 70:

        score += 1

        reasons.append(
            "RSI supports bullish momentum"
        )

    elif 30 < rsi < 50:

        score -= 1

        reasons.append(
            "RSI supports bearish momentum"
        )

    elif rsi >= 75:

        score -= 2

        reasons.append(
            "Overbought warning"
        )

    elif rsi <= 25:

        score += 2

        reasons.append(
            "Oversold recovery possibility"
        )

    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

    if candle in [

        "BULLISH_REJECTION",

        "STRONG_BULLISH"

    ]:

        score += 1

        reasons.append(
            "Bullish candle confirmation"
        )

    elif candle in [

        "BEARISH_REJECTION",

        "STRONG_BEARISH"

    ]:

        score -= 1

        reasons.append(
            "Bearish candle confirmation"
        )

    # -----------------------------------------------------
    # MTF
    # -----------------------------------------------------

    if mtf["confirmation"] == "BULLISH":

        score += 3

        reasons.append(
            "Higher timeframes support LONG"
        )

    elif mtf["confirmation"] == "BEARISH":

        score -= 3

        reasons.append(
            "Higher timeframes support SHORT"
        )

    # -----------------------------------------------------
    # ORDER FLOW
    # -----------------------------------------------------

    flow = analyze_order_flow(

        orderbook,

        deals

    )

    score += flow[
        "score"
    ]

    reasons.extend(

        flow[
            "reasons"
        ]

    )

    # -----------------------------------------------------
    # FUNDING CONTEXT
    # -----------------------------------------------------

    funding_rate = safe_float(

        funding.get(
            "funding_rate"
        ),

        0
    )

    # Extreme positive funding:
    # caution for crowded longs.

    if funding_rate > 0.0008:

        if score > 0:

            score -= 1

            reasons.append(

                "High positive funding adds LONG crowding risk"
            )

    # Extreme negative funding:
    # caution for crowded shorts.

    if funding_rate < -0.0008:

        if score < 0:

            score += 1

            reasons.append(

                "Strong negative funding adds SHORT crowding risk"
            )

    # =====================================================
    # FINAL DIRECTION
    # =====================================================

    direction = "NO_TRADE"

    if score >= MIN_SCORE:

        direction = "LONG"

    elif score <= -MIN_SCORE:

        direction = "SHORT"

    # Higher timeframe conflict rejection

    if (

        direction == "LONG"

        and

        mtf[
            "confirmation"
        ] == "BEARISH"

    ):

        direction = "NO_TRADE"

        reasons.append(

            "LONG rejected due to higher timeframe conflict"
        )

    if (

        direction == "SHORT"

        and

        mtf[
            "confirmation"
        ] == "BULLISH"

    ):

        direction = "NO_TRADE"

        reasons.append(

            "SHORT rejected due to higher timeframe conflict"
        )

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    entry = price

    stop_loss = None

    tp1 = None

    tp2 = None

    tp3 = None

    if direction == "LONG":

        atr_stop = (

            entry -

            atr *

            SL_ATR
        )

        stop_loss = min(

            atr_stop,

            levels[
                "support"
            ]

        )

        risk = (

            entry -

            stop_loss
        )

        if risk > 0:

            tp1 = (

                entry +

                risk *

                TP1_RR
            )

            tp2 = (

                entry +

                risk *

                TP2_RR
            )

            tp3 = (

                entry +

                risk *

                TP3_RR
            )

    elif direction == "SHORT":

        atr_stop = (

            entry +

            atr *

            SL_ATR
        )

        stop_loss = max(

            atr_stop,

            levels[
                "resistance"
            ]

        )

        risk = (

            stop_loss -

            entry
        )

        if risk > 0:

            tp1 = (

                entry -

                risk *

                TP1_RR
            )

            tp2 = (

                entry -

                risk *

                TP2_RR
            )

            tp3 = (

                entry -

                risk *

                TP3_RR
            )

    # =====================================================
    # SIGNAL
    # =====================================================

    if direction == "LONG":

        signal = (

            "STRONG BUY"

            if score >= 14

            else

            "BUY"
        )

    elif direction == "SHORT":

        signal = (

            "STRONG SELL"

            if score <= -14

            else

            "SELL"
        )

    else:

        signal = "NO TRADE"

    confidence = min(

        95,

        50 +

        abs(score) * 2.5

    )

    if direction == "NO_TRADE":

        confidence = min(

            confidence,

            55
        )

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "signal":
            signal,

        "direction":
            direction,

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
                8
            ),

        "entry":
            round(
                entry,
                8
            ),

        "stop_loss":
            (
                round(
                    stop_loss,
                    8
                )

                if stop_loss is not None

                else None
            ),

        "take_profit_1":
            (
                round(
                    tp1,
                    8
                )

                if tp1 is not None

                else None
            ),

        "take_profit_2":
            (
                round(
                    tp2,
                    8
                )

                if tp2 is not None

                else None
            ),

        "take_profit_3":
            (
                round(
                    tp3,
                    8
                )

                if tp3 is not None

                else None
            ),

        "market_bias":
            bias,

        "market_structure":
            structure,

        "break_of_structure":
            bos,

        "liquidity_sweep":
            sweep,

        "breakout":
            breakout,

        "retest":
            retest,

        "candle":
            candle,

        "rsi":
            round(
                rsi,
                2
            ),

        "atr":
            round(
                atr,
                8
            ),

        "volume_ratio":
            round(
                volume_ratio,
                3
            ),

        "support":
            round(
                levels[
                    "support"
                ],
                8
            ),

        "resistance":
            round(
                levels[
                    "resistance"
                ],
                8
            ),

        "order_flow":
            flow,

        "orderbook":
            orderbook,

        "executed_flow":
            deals,

        "funding":
            funding,

        "multi_timeframe":
            mtf,

        "reasons":
            reasons,

        "warning":
            (
                "Analysis only. "
                "No guaranteed profit. "
                "Paper trade before real money."
            )

    }


# =========================================================
# MONEY MANAGEMENT
# =========================================================

def calculate_position_size(

    entry,

    stop_loss,

    account_size,

    risk_percent

):

    if stop_loss is None:

        return {

            "risk_amount":
                0,

            "position_size":
                0,

            "position_value":
                0

        }

    risk_amount = (

        account_size *

        risk_percent /

        100

    )

    distance = abs(

        entry -

        stop_loss
    )

    if distance <= 0:

        return {

            "risk_amount":
                0,

            "position_size":
                0,

            "position_value":
                0

        }

    position_size = (

        risk_amount /

        distance
    )

    position_value = (

        position_size *

        entry
    )

    return {

        "risk_amount":
            round(
                risk_amount,
                2
            ),

        "position_size":
            round(
                position_size,
                8
            ),

        "position_value":
            round(
                position_value,
                2
            )

    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "status":
            "success",

        "bot":
            "GM Smart Scalper V4",

        "mode":
            "MEXC FUTURES ANALYSIS ONLY",

        "version":
            "4.0",

        "endpoints": [

            "/health",

            "/analysis/BTC_USDT",

            "/orderflow/BTC_USDT",

            "/backtest/BTC_USDT"

        ]

    })


# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():

    try:

        ping = mexc_get(

            "/api/v1/contract/ping"

        )

        return jsonify({

            "status":
                "healthy",

            "mexc":
                "connected",

            "server_time":
                ping,

            "bot":
                "GM Smart Scalper V4",

            "timestamp":
                utc_now()

        })

    except Exception as e:

        return jsonify({

            "status":
                "unhealthy",

            "message":
                str(e),

            "timestamp":
                utc_now()

        }), 500


# =========================================================
# ORDER FLOW ENDPOINT
# =========================================================

@app.route(
    "/orderflow/<symbol>"
)
def orderflow(symbol):

    try:

        symbol = normalize_symbol(
            symbol
        )

        orderbook = get_orderbook(

            symbol,

            ORDERBOOK_LEVELS

        )

        deals = get_recent_deals(
            symbol
        )

        funding = get_funding_rate(
            symbol
        )

        flow = analyze_order_flow(

            orderbook,

            deals

        )

        return jsonify({

            "status":
                "success",

            "symbol":
                symbol,

            "orderbook":
                orderbook,

            "executed_flow":
                deals,

            "funding":
                funding,

            "combined_order_flow":
                flow,

            "timestamp":
                utc_now()

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                normalize_symbol(
                    symbol
                ),

            "message":
                str(e)

        }), 400


# =========================================================
# LIVE ANALYSIS
# =========================================================

@app.route(
    "/analysis/<symbol>"
)
def analysis(symbol):

    try:

        symbol = normalize_symbol(
            symbol
        )

        account_size = float(

            request.args.get(

                "account_size",

                1000

            )

        )

        risk_percent = float(

            request.args.get(

                "risk_percent",

                1

            )

        )

        df = get_data(

            symbol,

            "5m",

            1000

        )

        df = add_indicators(
            df
        )

        mtf = get_mtf_confirmation(
            symbol
        )

        orderbook = get_orderbook(

            symbol,

            ORDERBOOK_LEVELS

        )

        deals = get_recent_deals(
            symbol
        )

        funding = get_funding_rate(
            symbol
        )

        result = generate_signal(

            df,

            symbol,

            mtf,

            orderbook,

            deals,

            funding

        )

        money = calculate_position_size(

            result[
                "entry"
            ],

            result[
                "stop_loss"
            ],

            account_size,

            risk_percent

        )

        result[
            "money_management"
        ] = {

            "account_size":
                account_size,

            "risk_percent":
                risk_percent,

            **money

        }

        result[
            "timestamp"
        ] = utc_now()

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                normalize_symbol(
                    symbol
                ),

            "message":
                str(e),

            "timestamp":
                utc_now()

        }), 400


# =========================================================
# SIMPLE BACKTEST
# =========================================================

def run_backtest(

    df,

    symbol,

    fee_percent=0.06,

    slippage_percent=0.02

):

    df = add_indicators(
        df
    )

    wins = 0

    losses = 0

    total_trades = 0

    net_return = 0

    trades = []

    start = 220

    for i in range(

        start,

        len(df) -

        MAX_HOLD_CANDLES -

        1

    ):

        historical = (

            df.iloc[
                :i + 1
            ]

            .copy()

        )

        fake_mtf = {

            "confirmation":

                get_market_bias(

                    historical

                )[
                    "bias"
                ],

            "timeframes":
                {},

            "bullish_count":
                0,

            "bearish_count":
                0

        }

        # Historical backtest does not use
        # live order book/deals.

        empty_ob = {

            "pressure":
                "BALANCED",

            "imbalance_percent":
                0

        }

        empty_deals = {

            "aggression":
                "BALANCED",

            "delta_percent":
                0

        }

        empty_funding = {

            "funding_rate":
                0

        }

        result = generate_signal(

            historical,

            symbol,

            fake_mtf,

            empty_ob,

            empty_deals,

            empty_funding

        )

        direction = result[
            "direction"
        ]

        if direction == "NO_TRADE":

            continue

        entry = safe_float(

            result[
                "entry"
            ]

        )

        sl = result[
            "stop_loss"
        ]

        tp = result[
            "take_profit_1"
        ]

        if (

            sl is None

            or

            tp is None

        ):

            continue

        total_trades += 1

        outcome = "OPEN"

        exit_price = None

        for j in range(

            i + 1,

            min(

                i +

                1 +

                MAX_HOLD_CANDLES,

                len(df)

            )

        ):

            high = safe_float(

                df.iloc[j][
                    "high"
                ]

            )

            low = safe_float(

                df.iloc[j][
                    "low"
                ]

            )

            if direction == "LONG":

                if low <= sl:

                    outcome = "LOSS"

                    exit_price = sl

                    break

                if high >= tp:

                    outcome = "WIN"

                    exit_price = tp

                    break

            else:

                if high >= sl:

                    outcome = "LOSS"

                    exit_price = sl

                    break

                if low <= tp:

                    outcome = "WIN"

                    exit_price = tp

                    break

        if outcome == "OPEN":

            exit_price = safe_float(

                df.iloc[
                    min(

                        i +

                        MAX_HOLD_CANDLES,

                        len(df) - 1

                    )

                ][
                    "close"
                ]

            )

        if direction == "LONG":

            gross = (

                (
                    exit_price -

                    entry
                )

                /

                entry

            ) * 100

        else:

            gross = (

                (
                    entry -

                    exit_price
                )

                /

                entry

            ) * 100

        total_cost = (

            fee_percent * 2

            +

            slippage_percent

        )

        net = (

            gross -

            total_cost

        )

        if net > 0:

            wins += 1

            outcome = "WIN"

        else:

            losses += 1

            outcome = "LOSS"

        net_return += net

        trades.append({

            "direction":
                direction,

            "entry":
                round(
                    entry,
                    8
                ),

            "exit":
                round(
                    exit_price,
                    8
                ),

            "result":
                outcome,

            "net_return_percent":
                round(
                    net,
                    4
                )

        })

    win_rate = (

        wins /

        total_trades *

        100

    ) if total_trades else 0

    return {

        "status":
            "success",

        "symbol":
            symbol,

        "strategy":
            "GM Smart Scalper V4",

        "candles_tested":
            len(df) - start,

        "total_trades":
            total_trades,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_percent":
            round(
                win_rate,
                2
            ),

        "net_return_percent":
            round(
                net_return,
                4
            ),

        "fee_percent_per_side":
            fee_percent,

        "slippage_percent":
            slippage_percent,

        "recent_trades":
            trades[-20:],

        "warning":
            (
                "Historical backtest only. "
                "Order Book and executed trade flow "
                "are not available historically in "
                "this simple backtest."
            )

    }


@app.route(
    "/backtest/<symbol>"
)
def backtest(symbol):

    try:

        symbol = normalize_symbol(
            symbol
        )

        limit = int(

            request.args.get(

                "limit",

                1000

            )

        )

        fee = float(

            request.args.get(

                "fee",

                0.06

            )

        )

        slippage = float(

            request.args.get(

                "slippage",

                0.02

            )

        )

        df = get_data(

            symbol,

            "5m",

            limit

        )

        result = run_backtest(

            df,

            symbol,

            fee,

            slippage

        )

        result[
            "timestamp"
        ] = utc_now()

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "symbol":
                normalize_symbol(
                    symbol
                ),

            "message":
                str(e)

        }), 400


# =========================================================
# RUN
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
