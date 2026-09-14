import json
import re
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

with open("watchlist.json", encoding="utf-8") as _f:
    _WL = json.load(_f)
HOLDINGS = _WL.get("holdings", [])
WATCHLIST_US = _WL.get("watchlist_us", [])
WATCHLIST_HK = _WL.get("watchlist_hk", [])
WATCHLIST = WATCHLIST_US + WATCHLIST_HK
TICKERS = HOLDINGS + WATCHLIST
BENCHMARK = {"US": _WL.get("benchmark_us", "^GSPC"), "HK": _WL.get("benchmark_hk", "^HSI")}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

try:
    with open("positions.json", encoding="utf-8") as _f:
        POSITIONS = json.load(_f)
except FileNotFoundError:
    POSITIONS = {}

try:
    with open("sectors.json", encoding="utf-8") as _f:
        SECTORS = json.load(_f)
except FileNotFoundError:
    SECTORS = {}


def market_of(ticker):
    return "HK" if ticker.endswith(".HK") else "US"


def fetch_chart(ticker, range_="2y", interval="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?range={range_}&interval={interval}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s"),
        "open": q["open"],
        "high": q["high"],
        "low": q["low"],
        "close": q["close"],
        "volume": q["volume"],
    })
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    meta = result["meta"]
    return df, meta


def compute_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    # --- volume: OBV + 20d average volume ---
    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (direction * df["volume"]).fillna(0).cumsum()
    df["obv_ma20"] = df["obv"].rolling(20).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # --- ATR(14) for risk levels ---
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    return df


def compute_relative_strength(df, bench_df, window=60):
    d = df[["date", "close"]].rename(columns={"close": "px"})
    b = bench_df[["date", "close"]].rename(columns={"close": "bx"})
    m = pd.merge(d, b, on="date", how="inner")
    if len(m) < window + 1:
        window = max(5, len(m) - 1)
    stock_ret = m["px"].iloc[-1] / m["px"].iloc[-1 - window] - 1
    bench_ret = m["bx"].iloc[-1] / m["bx"].iloc[-1 - window] - 1
    rel = stock_ret - bench_ret
    rs_line = (m["px"] / m["px"].iloc[0]) / (m["bx"] / m["bx"].iloc[0])
    return {
        "window_days": window,
        "stock_return_pct": round(stock_ret * 100, 1),
        "benchmark_return_pct": round(bench_ret * 100, 1),
        "relative_pct": round(rel * 100, 1),
        "rs_series": [round(float(x), 4) for x in rs_line.tail(260)],
        "rs_dates": [dt.strftime("%Y-%m-%d") for dt in m["date"].tail(260)],
    }


def compute_weekly_trend(df):
    d = df.set_index("date")["close"].resample("W").last().dropna()
    if len(d) < 42:
        return None
    sma10 = d.rolling(10).mean()
    sma40 = d.rolling(40).mean()
    if pd.isna(sma10.iloc[-1]) or pd.isna(sma40.iloc[-1]):
        return None
    return "bull" if sma10.iloc[-1] > sma40.iloc[-1] else "bear"


def detect_divergence(df, lookback=40):
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < lookback or d["rsi14"].isna().any():
        return None
    half = lookback // 2
    first, second = d.iloc[:half], d.iloc[half:]

    i1 = first["close"].idxmax()
    i2 = second["close"].idxmax()
    if d.loc[i2, "close"] > d.loc[i1, "close"] and d.loc[i2, "rsi14"] < d.loc[i1, "rsi14"] and d.loc[i2, "rsi14"] > 50:
        return "bearish"

    j1 = first["close"].idxmin()
    j2 = second["close"].idxmin()
    if d.loc[j2, "close"] < d.loc[j1, "close"] and d.loc[j2, "rsi14"] > d.loc[j1, "rsi14"] and d.loc[j2, "rsi14"] < 50:
        return "bullish"

    return None


CANDLESTICK_NAMES_ZH = {
    "bull_engulf": "看漲吞噬",
    "bear_engulf": "看跌吞噬",
    "hammer": "槌子線",
    "hanging_man": "上吊線",
    "shooting_star": "流星線",
}


def compute_candlestick_series(df):
    """Classic single/two-candle price-action patterns (bullish/bearish engulfing, hammer,
    hanging man, shooting star), each gated by a simple prior-trend filter (close 5 sessions
    ago vs. yesterday's close) since these are meant to be *reversal* signals — a hammer-shaped
    candle in the middle of a flat range isn't the same claim as one after a real decline.
    Returns (score, name) numpy arrays aligned to df's rows: score in {-1,0,+1}, name is the
    matched pattern key (see CANDLESTICK_NAMES_ZH) or None. This is a genuinely different kind
    of short-term factor than RSI/MACD/Bollinger — those confirm an already-moving indicator,
    this reads the raw shape of the last 1-2 candles directly."""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    body = np.abs(c - o)
    rng = h - l
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    bullish_candle = c > o
    bearish_candle = c < o

    close_shift1 = np.roll(c, 1); close_shift1[:1] = np.nan
    open_shift1 = np.roll(o, 1); open_shift1[:1] = np.nan
    close_shift6 = np.roll(c, 6); close_shift6[:6] = np.nan
    with np.errstate(invalid="ignore"):
        trend_down = close_shift1 < close_shift6
        trend_up = close_shift1 > close_shift6

        small_body = body <= rng * 0.35
        long_lower = lower_shadow >= body * 2
        long_upper = upper_shadow >= body * 2
        short_upper = upper_shadow <= body * 0.5
        short_lower = lower_shadow <= body * 0.5

        hammer = small_body & long_lower & short_upper & trend_down
        hanging_man = small_body & long_lower & short_upper & trend_up
        shooting_star = small_body & long_upper & short_lower & trend_up

        prev_bearish = close_shift1 < open_shift1
        prev_bullish = close_shift1 > open_shift1
        engulf_up = (o <= close_shift1) & (c >= open_shift1)
        engulf_down = (o >= close_shift1) & (c <= open_shift1)
        bull_engulf = prev_bearish & bullish_candle & engulf_up & trend_down
        bear_engulf = prev_bullish & bearish_candle & engulf_down & trend_up

    score = np.zeros(n)
    name = np.array([None] * n, dtype=object)
    for cond, val, key in [
        (hammer, 1, "hammer"),
        (bull_engulf, 1, "bull_engulf"),
        (hanging_man, -1, "hanging_man"),
        (shooting_star, -1, "shooting_star"),
        (bear_engulf, -1, "bear_engulf"),
    ]:
        score = np.where(cond, val, score)
        name = np.where(cond, key, name)
    return score, name


def parse_upside_pct(price_target_str):
    if not price_target_str:
        return None
    m = re.search(r'\(([+-]?\d+\.?\d*)\s*%', price_target_str)
    if m:
        return float(m.group(1))
    return None


def rating_tier(rating_str):
    if not rating_str:
        return None
    r = rating_str.lower()
    if "strong buy" in r:
        return 2
    if "strong sell" in r:
        return -2
    if "buy" in r:
        return 1
    if "sell" in r:
        return -1
    if "hold" in r or "neutral" in r:
        return 0
    return None


def build_fundamental_factor(fund_entry):
    if not fund_entry:
        return None
    tier = rating_tier(fund_entry.get("rating"))
    upside = parse_upside_pct(fund_entry.get("priceTarget"))
    if tier is None or upside is None:
        return None
    if tier >= 1 and upside >= 15:
        return (1, f"分析師評等為「{fund_entry.get('rating')}」，目標價 {fund_entry.get('priceTarget')} 隱含約 {upside:.0f}% 上檔空間")
    if tier <= -1 and upside <= 0:
        return (-1, f"分析師評等為「{fund_entry.get('rating')}」，現價已高於目標價 {fund_entry.get('priceTarget')}")
    return None


def build_news_factor(news_entries):
    if not news_entries:
        return None
    bulls = [n for n in news_entries if n.get("sentiment") == "bull"]
    bears = [n for n in news_entries if n.get("sentiment") == "bear"]
    if len(bulls) > len(bears) and bulls:
        top = bulls[0]
        reason = top.get("sentiment_reason") or top.get("summary") or ""
        return (1, f"近期新聞偏多：{top['title']}（{reason}）")
    if len(bears) > len(bulls) and bears:
        top = bears[0]
        reason = top.get("sentiment_reason") or top.get("summary") or ""
        return (-1, f"近期新聞偏空：{top['title']}（{reason}）")
    return None


HORIZON_LABELS = {"short": "短線", "medium": "中線", "long": "長線"}


def _horizon_label(score):
    if score >= 1:
        return "偏多"
    if score <= -1:
        return "偏空"
    return "中性"


def build_signal(df, rel_strength, fund_entry=None, news_entries=None):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    reasons_bull = []
    reasons_bear = []
    horizon_scores = {"short": 0, "medium": 0, "long": 0}

    def add(horizon, delta, bull_text, bear_text):
        horizon_scores[horizon] += delta
        target = reasons_bull if delta > 0 else reasons_bear
        target.append({"text": bull_text if delta > 0 else bear_text, "horizon": horizon})

    # ---------- 中線（趨勢：週線/月線級別，抱幾週到幾個月）----------
    weekly_trend = compute_weekly_trend(df)
    if weekly_trend == "bull":
        add("medium", 1, "週線均線也維持多頭排列，日線訊號較不易是單純雜訊", None)
    elif weekly_trend == "bear":
        add("medium", -1, None, "週線均線已轉空頭，日線訊號可能只是短線反彈")

    divergence = detect_divergence(df)
    if divergence == "bullish":
        add("medium", 1, "近期股價創低但 RSI 未同步創低（底背離），留意止跌訊號", None)
    elif divergence == "bearish":
        add("medium", -1, None, "近期股價創高但 RSI 未同步創高（頂背離），留意動能減弱")

    if last["sma50"] > last["sma200"]:
        add("medium", 1, "50日均線在200日均線之上（多頭排列）", None)
    elif last["sma50"] < last["sma200"]:
        add("medium", -1, None, "50日均線在200日均線之下（空頭排列）")

    if prev["sma20"] <= prev["sma50"] and last["sma20"] > last["sma50"]:
        add("medium", 1, "20日均線剛向上穿越50日均線（黃金交叉）", None)
    elif prev["sma20"] >= prev["sma50"] and last["sma20"] < last["sma50"]:
        add("medium", -1, None, "20日均線剛向下穿越50日均線（死亡交叉）")

    if rel_strength and rel_strength["relative_pct"] >= 5:
        add("medium", 1, f"近{rel_strength['window_days']}日走勢優於大盤 {rel_strength['relative_pct']:+.1f} 個百分點", None)
    elif rel_strength and rel_strength["relative_pct"] <= -5:
        add("medium", -1, None, f"近{rel_strength['window_days']}日走勢落後大盤 {rel_strength['relative_pct']:+.1f} 個百分點")

    # ---------- 短線（日內～數日就可能改變的動能訊號）----------
    if last["rsi14"] < 30:
        add("short", 1, f"RSI14 為 {last['rsi14']:.1f}，處於超賣區", None)
    elif last["rsi14"] > 70:
        add("short", -1, None, f"RSI14 為 {last['rsi14']:.1f}，處於超買區")

    if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        add("short", 1, "MACD 剛向上穿越訊號線", None)
    elif prev["macd"] >= prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        add("short", -1, None, "MACD 剛向下穿越訊號線")

    if last["close"] < last["bb_lower"]:
        add("short", 1, "股價跌破布林通道下軌，短線可能超跌", None)
    elif last["close"] > last["bb_upper"]:
        add("short", -1, None, "股價突破布林通道上軌，短線可能過熱")

    price_up = last["close"] > prev["close"]
    vol_above_avg = pd.notna(last["vol_ma20"]) and last["volume"] > last["vol_ma20"] * 1.2
    obv_rising = pd.notna(last["obv_ma20"]) and last["obv"] > last["obv_ma20"]
    if price_up and vol_above_avg and obv_rising:
        add("short", 1, "價格上漲且成交量放大、OBV 走升，量價同步確認", None)
    elif (not price_up) and vol_above_avg and (not obv_rising):
        add("short", -1, None, "價格下跌且成交量放大、OBV 走弱，賣壓有量能支撐")

    candle_score, candle_name = compute_candlestick_series(df)
    if candle_score[-1] != 0:
        pattern_zh = CANDLESTICK_NAMES_ZH[candle_name[-1]]
        if candle_score[-1] > 0:
            add("short", 1, f"出現「{pattern_zh}」K線形態，屬於價格行為的反轉訊號", None)
        else:
            add("short", -1, None, f"出現「{pattern_zh}」K線形態，屬於價格行為的反轉訊號")

    # ---------- 長線（基本面評等/目標價、新聞事件，影響通常是好幾季）----------
    fund_factor = build_fundamental_factor(fund_entry)
    if fund_factor:
        delta, reason = fund_factor
        add("long", delta, reason, reason)

    news_factor = build_news_factor(news_entries)
    if news_factor:
        delta, reason = news_factor
        add("long", delta, reason, reason)

    score = horizon_scores["short"] + horizon_scores["medium"] + horizon_scores["long"]

    if score >= 2:
        label = "偏多"
    elif score <= -2:
        label = "偏空"
    else:
        label = "中性 / 觀望"

    data_days = len(df)
    thin_data = data_days < MIN_HISTORY_DAYS

    return {
        "score": int(score),
        "label": label,
        "reasons_bull": reasons_bull,
        "reasons_bear": reasons_bear,
        "breakdown": {k: int(v) for k, v in horizon_scores.items()},
        "horizons": {
            k: {"score": int(v), "label": _horizon_label(v)}
            for k, v in horizon_scores.items()
        },
        "thin_data": thin_data,
        "data_days": data_days,
    }


def build_position(ticker, price, signal, risk):
    pos = POSITIONS.get(ticker)
    if not pos:
        return None
    shares = pos["shares"]
    cost_basis = pos["costBasis"]
    market_value = price * shares
    unrealized_abs = (price - cost_basis) * shares
    unrealized_pct = (price / cost_basis - 1) * 100
    in_profit = unrealized_pct >= 0
    bullish = "偏多" in signal["label"]
    bearish = "偏空" in signal["label"]

    bull_text = "、".join(r["text"] for r in signal["reasons_bull"])
    bear_text = "、".join(r["text"] for r in signal["reasons_bear"])
    has_bear_divergence = "頂背離" in bear_text
    has_bull_divergence = "底背離" in bull_text
    lagging_benchmark = "走勢落後大盤" in bear_text
    leading_benchmark = "走勢優於大盤" in bull_text
    big_gain = unrealized_pct >= 15
    big_loss = unrealized_pct <= -10

    clues = []
    if has_bear_divergence:
        clues.append("出現頂背離（動能轉弱的早期訊號）")
    if lagging_benchmark:
        clues.append("走勢落後大盤")
    if has_bull_divergence:
        clues.append("出現底背離")
    if leading_benchmark:
        clues.append("走勢優於大盤")

    clue_text = "，且" + "、".join(clues) if clues else ""

    horizons = signal.get("horizons") or {}
    h_short = horizons.get("short", {}).get("label")
    h_medium = horizons.get("medium", {}).get("label")
    h_long = horizons.get("long", {}).get("label")
    horizon_conflict = None
    if h_short and h_medium and h_long:
        labels = {h_short, h_medium, h_long}
        if "偏多" in labels and "偏空" in labels:
            horizon_conflict = f"短線{h_short}／中線{h_medium}／長線{h_long}，三個時間尺度方向不一致，請依你的交易期限自行判斷要參考哪一層。"

    if in_profit and big_gain and (has_bear_divergence or not bullish):
        note = f"獲利已達 {unrealized_pct:.1f}%{clue_text}，訊號不如先前強勁，可考慮分批獲利了結、而非等訊號真正轉空才動作。"
    elif in_profit and bullish:
        note = f"獲利中（{unrealized_pct:+.1f}%），技術訊號同步偏多{clue_text}，暫無明顯風險訊號，可續抱。"
    elif in_profit and bearish:
        note = f"目前獲利（{unrealized_pct:+.1f}%），但技術訊號已轉空{clue_text}，建議對照下方參考停損價，若跌破可考慮減碼或停利了結。"
    elif in_profit:
        note = f"目前小幅獲利（{unrealized_pct:+.1f}%），技術訊號中性{clue_text}，可持續觀察均線與成交量變化，暫不用急著動作。"
    elif big_loss and bearish:
        note = f"浮虧已達 {unrealized_pct:.1f}%，技術訊號同步偏空{clue_text}，現價已接近或低於參考停損價，建議認真考慮停損。"
    elif (not in_profit) and bearish:
        note = f"目前浮虧（{unrealized_pct:.1f}%），技術訊號同步偏空{clue_text}，建議對照下方參考停損價自行評估。"
    elif (not in_profit) and bullish:
        note = f"目前浮虧（{unrealized_pct:.1f}%），但技術訊號已轉強{clue_text}，可觀察是否出現止跌回升再決定是否加碼。"
    else:
        note = f"目前浮虧（{unrealized_pct:.1f}%），技術訊號中性{clue_text}，尚無明確止跌或轉弱訊號，建議續觀察。"

    if horizon_conflict:
        note += "　" + horizon_conflict

    dist_to_stop_pct = None
    dist_to_target_pct = None
    if risk:
        dist_to_stop_pct = (price / risk["stop_loss"] - 1) * 100
        dist_to_target_pct = (risk["take_profit"] / price - 1) * 100

    return {
        "shares": shares,
        "cost_basis": cost_basis,
        "market_value": round(market_value, 2),
        "unrealized_abs": round(unrealized_abs, 2),
        "unrealized_pct": round(unrealized_pct, 2),
        "dist_to_stop_pct": None if dist_to_stop_pct is None else round(dist_to_stop_pct, 1),
        "dist_to_target_pct": None if dist_to_target_pct is None else round(dist_to_target_pct, 1),
        "note": note,
    }


def build_portfolio_summary(results, holdings, usd_per_hkd):
    """Point 1 (風險管理) and point 4 (分散風險) both need a portfolio-level view, not
    just per-stock risk — this converts every holding to USD, sums market value and the
    capital that would be lost if every stop-loss fired simultaneously, and flags
    concentration (single position or single sector too large a share of the book)."""
    rows = []
    total_value_usd = 0.0
    total_risk_usd = 0.0
    for t in holdings:
        r = results.get(t)
        if not r or not r.get("position"):
            continue
        pos = r["position"]
        risk = r.get("risk")
        currency = r["meta"]["currency"]
        fx = 1.0 if currency == "USD" else (1.0 / usd_per_hkd if usd_per_hkd else 1.0)
        value_usd = pos["market_value"] * fx
        risk_amount_usd = 0.0
        if risk:
            price = r["meta"]["price"]
            per_share_risk = max(price - risk["stop_loss"], 0)
            risk_amount_usd = per_share_risk * pos["shares"] * fx
        total_value_usd += value_usd
        total_risk_usd += risk_amount_usd
        rows.append({
            "ticker": t,
            "sector": SECTORS.get(t, "未分類"),
            "value_usd": round(value_usd, 2),
            "risk_amount_usd": round(risk_amount_usd, 2),
        })

    for row in rows:
        row["weight_pct"] = round(row["value_usd"] / total_value_usd * 100, 1) if total_value_usd else 0.0
    rows.sort(key=lambda r: -r["value_usd"])

    # group by the broad category before the "－" separator (e.g. "科技－雲端／軟體" -> "科技")
    # so MSFT/NOK/1810.HK count as the same "科技" bucket even with different sub-sector labels
    broad_sector_weights = {}
    for row in rows:
        broad = row["sector"].split("－")[0]
        broad_sector_weights[broad] = broad_sector_weights.get(broad, 0) + row["weight_pct"]

    concentration_flags = []
    for row in rows:
        if row["weight_pct"] >= 40:
            concentration_flags.append(f"{row['ticker']} 占投資組合 {row['weight_pct']:.0f}%，單一部位集中度偏高")
    for sector, weight in broad_sector_weights.items():
        if weight >= 40 and sector != "未分類":
            concentration_flags.append(f"「{sector}」類股合計占投資組合 {weight:.0f}%，同類股同漲同跌的風險偏高")

    return {
        "total_value_usd": round(total_value_usd, 2),
        "total_risk_usd": round(total_risk_usd, 2),
        "total_risk_pct": round(total_risk_usd / total_value_usd * 100, 1) if total_value_usd else 0.0,
        "usd_per_hkd": usd_per_hkd,
        "positions": rows,
        "concentration_flags": concentration_flags,
    }


def build_risk_levels(df):
    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr14"]) if pd.notna(last["atr14"]) else None
    if atr is None:
        return None
    return {
        "atr14": round(atr, 3),
        "atr_pct": round(atr / price * 100, 2),
        "stop_loss": round(price - 1.5 * atr, 3),
        "take_profit": round(price + 2.0 * atr, 3),
    }


MIN_HISTORY_DAYS = 250  # ~1 trading year; below this, weekly trend / backtest are unreliable


def compute_rolling_technical_score(df, bench_df):
    """Recompute, for every historical day, the subset of build_signal's factors that
    are derivable purely from time-series price/volume data (i.e. everything except the
    fundamental-rating and news-sentiment factors, which only exist as a single snapshot
    for "today" and have no historical daily equivalent). Used to backtest the same logic
    that actually drives the live signal, instead of an unrelated toy strategy. Returns a
    DataFrame with one +1/0/-1 column per factor (so callers can both sum them into a
    composite score and, separately, measure each factor's own historical hit rate)."""
    n = len(df)
    factors = {}

    trend_valid = df["sma50"].notna() & df["sma200"].notna()
    factors["sma_trend"] = np.where(trend_valid, np.where(df["sma50"] > df["sma200"], 1, np.where(df["sma50"] < df["sma200"], -1, 0)), 0)

    prev_sma20 = df["sma20"].shift(1)
    prev_sma50 = df["sma50"].shift(1)
    cross_valid = (prev_sma20.notna() & prev_sma50.notna()).values
    golden = ((prev_sma20 <= prev_sma50) & (df["sma20"] > df["sma50"])).values
    death = ((prev_sma20 >= prev_sma50) & (df["sma20"] < df["sma50"])).values
    factors["sma_cross"] = np.where(cross_valid, np.where(golden, 1, np.where(death, -1, 0)), 0)

    rsi_valid = df["rsi14"].notna()
    factors["rsi"] = np.where(rsi_valid, np.where(df["rsi14"] < 30, 1, np.where(df["rsi14"] > 70, -1, 0)), 0)

    prev_macd = df["macd"].shift(1)
    prev_macd_signal = df["macd_signal"].shift(1)
    macd_valid = (prev_macd.notna() & prev_macd_signal.notna()).values
    macd_gold = ((prev_macd <= prev_macd_signal) & (df["macd"] > df["macd_signal"])).values
    macd_death = ((prev_macd >= prev_macd_signal) & (df["macd"] < df["macd_signal"])).values
    factors["macd"] = np.where(macd_valid, np.where(macd_gold, 1, np.where(macd_death, -1, 0)), 0)

    bb_valid = df["bb_lower"].notna() & df["bb_upper"].notna()
    factors["bollinger"] = np.where(bb_valid, np.where(df["close"] < df["bb_lower"], 1, np.where(df["close"] > df["bb_upper"], -1, 0)), 0)

    prev_close = df["close"].shift(1)
    price_up = (df["close"] > prev_close).values
    vol_valid = (df["vol_ma20"].notna() & df["obv_ma20"].notna()).values
    vol_above_avg = (df["volume"] > df["vol_ma20"] * 1.2).values
    obv_rising = (df["obv"] > df["obv_ma20"]).values
    vol_bull = price_up & vol_above_avg & obv_rising
    vol_bear = (~price_up) & vol_above_avg & (~obv_rising)
    factors["volume"] = np.where(vol_valid, np.where(vol_bull, 1, np.where(vol_bear, -1, 0)), 0)

    candle_score, _ = compute_candlestick_series(df)
    factors["candlestick"] = candle_score

    # weekly trend: precompute once, then align to each daily row via merge_asof
    weekly = df.set_index("date")["close"].resample("W").last().dropna()
    if len(weekly) >= 42:
        wsma10 = weekly.rolling(10).mean()
        wsma40 = weekly.rolling(40).mean()
        wtrend = pd.DataFrame({
            "date": weekly.index,
            "wbull": (wsma10 > wsma40).values,
            "wvalid": (wsma10.notna() & wsma40.notna()).values,
        })
        merged_w = pd.merge_asof(df[["date"]].sort_values("date"), wtrend.sort_values("date"), on="date")
        factors["weekly_trend"] = np.where(merged_w["wvalid"].astype("boolean").fillna(False), np.where(merged_w["wbull"], 1, -1), 0)
    else:
        factors["weekly_trend"] = np.zeros(n)

    # divergence: needs a rolling 40-day window, cheap enough to loop (O(40) per row)
    lookback = 40
    div_scores = np.zeros(n)
    for i in range(lookback, n):
        window = df.iloc[i - lookback + 1:i + 1]
        if window["rsi14"].isna().any():
            continue
        half = lookback // 2
        first, second = window.iloc[:half], window.iloc[half:]
        i1, i2 = first["close"].idxmax(), second["close"].idxmax()
        if window.loc[i2, "close"] > window.loc[i1, "close"] and window.loc[i2, "rsi14"] < window.loc[i1, "rsi14"] and window.loc[i2, "rsi14"] > 50:
            div_scores[i] = -1
            continue
        j1, j2 = first["close"].idxmin(), second["close"].idxmin()
        if window.loc[j2, "close"] < window.loc[j1, "close"] and window.loc[j2, "rsi14"] > window.loc[j1, "rsi14"] and window.loc[j2, "rsi14"] < 50:
            div_scores[i] = 1
    factors["divergence"] = div_scores

    # relative strength vs benchmark, vectorized via merge_asof + 60-day rolling return spread
    bench = bench_df[["date", "close"]].rename(columns={"close": "bx"}).sort_values("date")
    stock = df[["date", "close"]].rename(columns={"close": "px"}).sort_values("date")
    merged = pd.merge_asof(stock, bench, on="date")
    window_days = 60
    stock_ret = merged["px"] / merged["px"].shift(window_days) - 1
    bench_ret = merged["bx"] / merged["bx"].shift(window_days) - 1
    rel_pct = (stock_ret - bench_ret) * 100
    rel_valid = rel_pct.notna().values
    factors["relative_strength"] = np.where(rel_valid, np.where(rel_pct >= 5, 1, np.where(rel_pct <= -5, -1, 0)), 0)

    return pd.DataFrame(factors, index=df.index)


FACTOR_HORIZON = {
    "rsi": "short", "macd": "short", "bollinger": "short", "volume": "short", "candlestick": "short",
    "sma_trend": "medium", "sma_cross": "medium", "weekly_trend": "medium",
    "divergence": "medium", "relative_strength": "medium",
}
FACTOR_NAMES_ZH = {
    "rsi": "RSI超買超賣", "macd": "MACD交叉", "bollinger": "布林通道", "volume": "成交量confirm",
    "candlestick": "K線形態",
    "sma_trend": "均線多空排列", "sma_cross": "均線黃金/死亡交叉", "weekly_trend": "週線趨勢",
    "divergence": "RSI背離", "relative_strength": "相對大盤強弱",
    "fundamental": "分析師評等/目標價", "news": "新聞情緒", "other": "其他",
}

REASON_FACTOR_MARKERS = [
    ("分析師評等", "fundamental"),
    ("近期新聞", "news"),
    ("背離", "divergence"),
    ("K線形態", "candlestick"),
    ("RSI14", "rsi"),
    ("MACD", "macd"),
    ("成交量", "volume"),
    ("走勢優於大盤", "relative_strength"),
    ("走勢落後大盤", "relative_strength"),
    ("跌破布林", "bollinger"),
    ("突破布林", "bollinger"),
    ("黃金交叉", "sma_cross"),
    ("死亡交叉", "sma_cross"),
    ("週線均線", "weekly_trend"),
    ("均線在", "sma_trend"),
]


def identify_factor(text):
    for marker, factor in REASON_FACTOR_MARKERS:
        if marker in text:
            return factor
    return "other"


PREDICTION_HORIZON_DAYS = {"short": 1, "medium": 20, "long": 60}
PREDICTION_HORIZON_LABEL = {
    "short": "短線（下一個交易日）",
    "medium": "中線（20個交易日≈1個月）",
    "long": "長線（60個交易日≈3個月，為了能實際驗證命中率而非真的等12個月分析師目標價的期限）",
}


def load_predictions():
    try:
        with open("predictions.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_predictions(predictions):
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)


def resolve_ticker_predictions(ticker, df, predictions):
    """Check every still-open prediction for this ticker: has enough real trading days
    passed since it was made? If so, look up the actual price N trading days later and
    record whether the predicted direction was right — this is the live, zero-lookahead
    counterpart to the retroactive factor_accuracy report."""
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    closes = df["close"].tolist()
    date_index = {d: i for i, d in enumerate(dates)}
    for p in predictions:
        if p["ticker"] != ticker or p["status"] != "pending":
            continue
        made_idx = date_index.get(p["made_date"])
        if made_idx is None:
            continue
        n = PREDICTION_HORIZON_DAYS[p["horizon"]]
        target_idx = made_idx + n
        if target_idx >= len(dates):
            continue
        resolved_price = closes[target_idx]
        actual_return_pct = (resolved_price / p["made_price"] - 1) * 100
        actual_direction = "up" if resolved_price > p["made_price"] else ("down" if resolved_price < p["made_price"] else "flat")
        p["status"] = "resolved"
        p["resolved_date"] = dates[target_idx]
        p["resolved_price"] = round(resolved_price, 3)
        p["actual_return_pct"] = round(actual_return_pct, 2)
        p["actual_direction"] = actual_direction
        p["hit"] = bool(actual_direction == p["predicted_direction"])


def make_new_prediction(ticker, today_str, price, signal, predictions):
    """Open a new prediction whenever a horizon has a directional call (score != 0) and
    there isn't already one outstanding for that ticker+horizon — short-term is 1 trading
    day so a fresh one opens daily by construction; medium/long stay non-overlapping
    (wait for the current window to resolve) so hit-rate samples aren't pseudo-replicated
    from counting the same multi-week trend over and over."""
    horizons = signal.get("horizons") or {}
    pending_keys = {(p["ticker"], p["horizon"]) for p in predictions if p["status"] == "pending"}
    all_reasons = signal["reasons_bull"] + signal["reasons_bear"]
    for h, info in horizons.items():
        score = info["score"]
        if score == 0:
            continue
        if h != "short" and (ticker, h) in pending_keys:
            continue
        if any(p["ticker"] == ticker and p["horizon"] == h and p["made_date"] == today_str for p in predictions):
            continue
        reasons_for_horizon = [r["text"] for r in all_reasons if r["horizon"] == h]
        predictions.append({
            "ticker": ticker,
            "horizon": h,
            "made_date": today_str,
            "made_price": round(price, 3),
            "predicted_direction": "up" if score > 0 else "down",
            "horizon_score": score,
            "reasons": reasons_for_horizon,
            "status": "pending",
            "resolved_date": None,
            "resolved_price": None,
            "actual_return_pct": None,
            "actual_direction": None,
            "hit": None,
        })


def load_baseline_predictions():
    try:
        with open("baseline_predictions.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_baseline_predictions(predictions):
    with open("baseline_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)


def resolve_baseline_predictions(ticker, df, predictions):
    """Same zero-lookahead resolution mechanics as resolve_ticker_predictions, but for the
    naive 'just extrapolate yesterday's direction' control group (always 1 trading day)."""
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    closes = df["close"].tolist()
    date_index = {d: i for i, d in enumerate(dates)}
    for p in predictions:
        if p["ticker"] != ticker or p["status"] != "pending":
            continue
        made_idx = date_index.get(p["made_date"])
        if made_idx is None:
            continue
        target_idx = made_idx + 1
        if target_idx >= len(dates):
            continue
        resolved_price = closes[target_idx]
        actual_return_pct = (resolved_price / p["made_price"] - 1) * 100
        actual_direction = "up" if resolved_price > p["made_price"] else ("down" if resolved_price < p["made_price"] else "flat")
        p["status"] = "resolved"
        p["resolved_date"] = dates[target_idx]
        p["resolved_price"] = round(resolved_price, 3)
        p["actual_return_pct"] = round(actual_return_pct, 2)
        p["actual_direction"] = actual_direction
        p["hit"] = bool(actual_direction == p["predicted_direction"])


def make_baseline_prediction(ticker, made_date_str, df, predictions):
    """A deliberately naive benchmark: 'tomorrow continues whatever today's close did versus
    yesterday's close.' It carries no technical judgment at all — its entire purpose is to be
    a fair yardstick. Fired every day for every ticker (unlike the real signal, which only
    fires when a factor actually triggers) so we can tell whether the real signal's hit rate
    is genuinely better than blind persistence, or just riding the same market drift."""
    if len(df) < 2:
        return
    if any(p["ticker"] == ticker and p["made_date"] == made_date_str for p in predictions):
        return
    last_close = float(df.iloc[-1]["close"])
    prev_close = float(df.iloc[-2]["close"])
    direction = "up" if last_close >= prev_close else "down"
    predictions.append({
        "ticker": ticker,
        "made_date": made_date_str,
        "made_price": round(last_close, 3),
        "predicted_direction": direction,
        "status": "pending",
        "resolved_date": None,
        "resolved_price": None,
        "actual_return_pct": None,
        "actual_direction": None,
        "hit": None,
    })


def build_baseline_tracking(ticker, predictions):
    items = [p for p in predictions if p["ticker"] == ticker]
    resolved = sorted([p for p in items if p["status"] == "resolved"], key=lambda p: p["resolved_date"])
    pending = [p for p in items if p["status"] == "pending"]
    latest_resolved = resolved[-1] if resolved else None
    latest_pending = max(pending, key=lambda p: p["made_date"]) if pending else None
    hits = sum(1 for p in resolved if p["hit"])
    total = len(resolved)
    return {
        "today_predicted": latest_resolved["predicted_direction"] if latest_resolved else None,
        "today_actual": latest_resolved["actual_direction"] if latest_resolved else None,
        "today_hit": latest_resolved["hit"] if latest_resolved else None,
        "tomorrow_predicted": latest_pending["predicted_direction"] if latest_pending else None,
        "total_resolved": total,
        "hits": hits,
        "hit_rate_pct": round(hits / total * 100, 1) if total else None,
    }


def build_effective_short_view(short_tracking, baseline_tracking):
    """Display-only merge for the UI: prefer the real system call, fall back to the naive
    baseline (tagged so the frontend can render it muted/'reference-only') so every ticker
    shows *something* instead of a blank dash. Never used for accuracy stats — those stay
    strictly separate in short_tracking / baseline_accuracy so the comparison stays honest."""
    def pick(real_val, base_val):
        return real_val if real_val is not None else base_val
    today_source = "signal" if short_tracking["today_predicted"] is not None else ("baseline" if baseline_tracking["today_predicted"] is not None else None)
    tomorrow_source = "signal" if short_tracking["tomorrow_predicted"] is not None else ("baseline" if baseline_tracking["tomorrow_predicted"] is not None else None)
    return {
        "today_predicted": pick(short_tracking["today_predicted"], baseline_tracking["today_predicted"]),
        "today_actual": pick(short_tracking["today_actual"], baseline_tracking["today_actual"]),
        "today_hit": pick(short_tracking["today_hit"], baseline_tracking["today_hit"]),
        "today_source": today_source,
        "tomorrow_predicted": pick(short_tracking["tomorrow_predicted"], baseline_tracking["tomorrow_predicted"]),
        "tomorrow_source": tomorrow_source,
    }


def summarize_baseline_accuracy(predictions):
    resolved = [p for p in predictions if p["status"] == "resolved"]
    total = len(resolved)
    hits = sum(1 for p in resolved if p["hit"])
    by_market = {}
    for p in resolved:
        m = market_of(p["ticker"])
        agg = by_market.setdefault(m, {"n": 0, "hits": 0})
        agg["n"] += 1
        agg["hits"] += 1 if p["hit"] else 0
    market_summary = [
        {"market": m, "n": agg["n"], "hit_rate_pct": round(agg["hits"] / agg["n"] * 100, 1) if agg["n"] else None}
        for m, agg in sorted(by_market.items())
    ]
    return {
        "n": total,
        "hits": hits,
        "hit_rate_pct": round(hits / total * 100, 1) if total else None,
        "by_market": market_summary,
    }


def build_short_tracking(ticker, predictions):
    """Per-ticker view of the short-horizon (next-trading-day) prediction stream: today's
    prediction vs. today's realized outcome (the most recently resolved short prediction),
    tomorrow's freshly-made prediction (the current pending one), and a running hit/miss
    tally + recent history so a viewer can see not just whether it's working but why."""
    shorts = [p for p in predictions if p["ticker"] == ticker and p["horizon"] == "short"]
    resolved = sorted([p for p in shorts if p["status"] == "resolved"], key=lambda p: p["resolved_date"])
    pending = [p for p in shorts if p["status"] == "pending"]
    latest_resolved = resolved[-1] if resolved else None
    latest_pending = max(pending, key=lambda p: p["made_date"]) if pending else None

    hits = sum(1 for p in resolved if p["hit"])
    total = len(resolved)

    recent = [{
        "made_date": p["made_date"],
        "resolved_date": p["resolved_date"],
        "predicted_direction": p["predicted_direction"],
        "actual_direction": p["actual_direction"],
        "actual_return_pct": p["actual_return_pct"],
        "hit": p["hit"],
        "reasons": p["reasons"],
    } for p in resolved[-10:][::-1]]

    return {
        "today_predicted": latest_resolved["predicted_direction"] if latest_resolved else None,
        "today_actual": latest_resolved["actual_direction"] if latest_resolved else None,
        "today_hit": latest_resolved["hit"] if latest_resolved else None,
        "today_date": latest_resolved["resolved_date"] if latest_resolved else None,
        "tomorrow_predicted": latest_pending["predicted_direction"] if latest_pending else None,
        "tomorrow_made_date": latest_pending["made_date"] if latest_pending else None,
        "total_resolved": total,
        "hits": hits,
        "hit_rate_pct": round(hits / total * 100, 1) if total else None,
        "recent": recent,
    }


def summarize_predictions(predictions):
    by_horizon = {}
    reason_stats = {}
    for p in predictions:
        if p["status"] != "resolved":
            continue
        h = p["horizon"]
        agg = by_horizon.setdefault(h, {"n": 0, "hits": 0})
        agg["n"] += 1
        agg["hits"] += 1 if p["hit"] else 0
        for reason_text in p["reasons"]:
            factor = identify_factor(reason_text)
            key = (h, factor)
            r = reason_stats.setdefault(key, {"n": 0, "hits": 0})
            r["n"] += 1
            r["hits"] += 1 if p["hit"] else 0

    horizon_summary = []
    for h in ["short", "medium", "long"]:
        agg = by_horizon.get(h, {"n": 0, "hits": 0})
        hit_rate = round(agg["hits"] / agg["n"] * 100, 1) if agg["n"] else None
        horizon_summary.append({
            "horizon": h,
            "label": PREDICTION_HORIZON_LABEL[h],
            "n": agg["n"],
            "hit_rate_pct": hit_rate,
        })

    factor_summary = []
    for (h, factor), r in reason_stats.items():
        if r["n"] < 5:
            continue
        factor_summary.append({
            "horizon": h,
            "factor": factor,
            "name": FACTOR_NAMES_ZH.get(factor, factor),
            "n": r["n"],
            "hit_rate_pct": round(r["hits"] / r["n"] * 100, 1),
        })
    factor_summary.sort(key=lambda x: x["hit_rate_pct"])

    recent_resolved = sorted(
        [p for p in predictions if p["status"] == "resolved"],
        key=lambda p: p["resolved_date"], reverse=True,
    )

    return {
        "by_horizon": horizon_summary,
        "by_factor": factor_summary,
        "recent": recent_resolved[:20],
        "total_pending": sum(1 for p in predictions if p["status"] == "pending"),
    }


FACTOR_ACCURACY_HORIZON_DAYS = {"short": 10, "medium": 20}


def accumulate_factor_accuracy(df, factor_df, accumulator):
    """For every historical day a factor fired (+1/-1), look up the actual forward return
    N trading days later and record whether the factor's direction was right, plus the
    unconditional (baseline) forward return over the same horizon for comparison. This is
    a retroactive check of "when this factor said 偏多/偏空 in the past, was it actually
    right" — the root-cause data behind the 系統校準 report, using the 2 years of price
    history we already have rather than waiting months for live tracking to accumulate."""
    close = df["close"].values
    n = len(close)
    for factor, horizon in FACTOR_HORIZON.items():
        h = FACTOR_ACCURACY_HORIZON_DAYS[horizon]
        if n <= h:
            continue
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n - h] = close[h:] / close[:n - h] - 1
        values = factor_df[factor].values
        acc = accumulator.setdefault(factor, {"wins": 0, "events": 0, "sum_ret_when_fired": 0.0, "sum_ret_baseline": 0.0, "baseline_n": 0})
        valid_baseline = ~np.isnan(fwd_ret)
        acc["sum_ret_baseline"] += np.nansum(fwd_ret[valid_baseline])
        acc["baseline_n"] += int(valid_baseline.sum())
        fired = (values != 0) & valid_baseline
        if not fired.any():
            continue
        direction = values[fired]
        rets = fwd_ret[fired]
        wins = ((direction > 0) & (rets > 0)) | ((direction < 0) & (rets < 0))
        acc["wins"] += int(wins.sum())
        acc["events"] += int(fired.sum())
        acc["sum_ret_when_fired"] += float(np.sum(np.where(direction > 0, rets, -rets)))


def summarize_factor_accuracy(accumulator):
    report = []
    for factor, acc in accumulator.items():
        if acc["events"] < 20:
            continue
        hit_rate = acc["wins"] / acc["events"] * 100
        avg_ret_when_fired = acc["sum_ret_when_fired"] / acc["events"] * 100
        avg_ret_baseline = (acc["sum_ret_baseline"] / acc["baseline_n"] * 100) if acc["baseline_n"] else 0.0
        edge_pct = avg_ret_when_fired - avg_ret_baseline
        report.append({
            "factor": factor,
            "name": FACTOR_NAMES_ZH.get(factor, factor),
            "horizon": FACTOR_HORIZON.get(factor),
            "horizon_days": FACTOR_ACCURACY_HORIZON_DAYS.get(FACTOR_HORIZON.get(factor)),
            "events": acc["events"],
            "hit_rate_pct": round(hit_rate, 1),
            "avg_return_when_fired_pct": round(avg_ret_when_fired, 2),
            "avg_return_baseline_pct": round(avg_ret_baseline, 2),
            "edge_pct": round(edge_pct, 2),
            "weak": bool(hit_rate < 50 or edge_pct <= 0),
        })
    report.sort(key=lambda r: -r["edge_pct"])
    return report


ROUND_TRIP_COST_PCT = {
    "US": 0.0015,  # ~0.15%: mostly slippage, most US brokers are commission-free now
    "HK": 0.0035,  # ~0.35%: 0.1% stamp duty each side (0.2% total) + brokerage/trading fees/slippage
}


def backtest_composite_signal(df, factor_df, market="US", entry_threshold=2, exit_threshold=-1, atr_stop_mult=1.5):
    """Backtests the SAME rolling technical score used in the live signal (minus the
    fundamental/news factors, which have no historical time series), instead of an
    unrelated SMA-crossover toy strategy. Enters when the score clears entry_threshold,
    exits on either the score dropping to exit_threshold or an ATR-based stop-loss.
    Each round-trip trade is charged an estimated cost (stamp duty + fees + slippage) so
    the reported return reflects what you would have actually kept, not a costless ideal."""
    if len(df) < 30:
        return None
    cost_pct = ROUND_TRIP_COST_PCT.get(market, ROUND_TRIP_COST_PCT["US"])
    tech_score = factor_df.sum(axis=1)
    d = df.copy().reset_index(drop=True)
    d["tech_score"] = tech_score.values
    d = d[d["sma200"].notna()].reset_index(drop=True)
    if len(d) < 30:
        return None

    position = 0
    entry_price = 0.0
    stop_price = 0.0
    trades = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0

    for _, row in d.iterrows():
        if position == 0 and row["tech_score"] >= entry_threshold:
            position = 1
            entry_price = row["close"]
            atr = row["atr14"]
            stop_price = entry_price - atr_stop_mult * atr if pd.notna(atr) else entry_price * 0.9
        elif position == 1:
            hit_stop = row["close"] < stop_price
            exit_signal = row["tech_score"] <= exit_threshold
            if hit_stop or exit_signal:
                ret = row["close"] / entry_price - 1 - cost_pct
                equity *= (1 + ret)
                trades.append({"ret": ret, "stop": hit_stop})
                position = 0
        unreal = equity * (row["close"] / entry_price) if position == 1 else equity
        peak = max(peak, unreal)
        dd = (unreal - peak) / peak
        max_dd = min(max_dd, dd)

    if position == 1:
        ret = d.iloc[-1]["close"] / entry_price - 1 - cost_pct
        trades.append({"ret": ret, "stop": False})

    buy_hold_return = d.iloc[-1]["close"] / d.iloc[0]["close"] - 1
    strategy_return = equity - 1
    wins = [t for t in trades if t["ret"] > 0]
    win_rate = (len(wins) / len(trades)) if trades else 0.0
    stop_exits = sum(1 for t in trades if t["stop"])
    total_cost_drag_pct = cost_pct * len(trades) * 100

    return {
        "period_start": d.iloc[0]["date"].strftime("%Y-%m-%d"),
        "period_end": d.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "strategy_return_pct": round(strategy_return * 100, 1),
        "buy_hold_return_pct": round(buy_hold_return * 100, 1),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate * 100, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "stop_loss_exits": stop_exits,
        "cost_pct_per_trade": round(cost_pct * 100, 2),
        "total_cost_drag_pct": round(total_cost_drag_pct, 1),
        "rule": f"技術面滾動分數 ≥{entry_threshold} 進場；≤{exit_threshold} 或跌破 {atr_stop_mult}×ATR 停損則出場（已扣除每次交易約 {cost_pct*100:.2f}% 的成本估算；不含基本面/新聞因子，因無歷史逐日資料）",
    }


def to_records(df, tail=260):
    d = df.tail(tail)
    out = []
    for _, r in d.iterrows():
        out.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "close": round(float(r["close"]), 3),
            "volume": int(r["volume"]) if pd.notna(r["volume"]) else None,
            "sma20": None if pd.isna(r["sma20"]) else round(float(r["sma20"]), 3),
            "sma50": None if pd.isna(r["sma50"]) else round(float(r["sma50"]), 3),
            "sma200": None if pd.isna(r["sma200"]) else round(float(r["sma200"]), 3),
            "rsi14": None if pd.isna(r["rsi14"]) else round(float(r["rsi14"]), 2),
            "macd": None if pd.isna(r["macd"]) else round(float(r["macd"]), 3),
            "macd_signal": None if pd.isna(r["macd_signal"]) else round(float(r["macd_signal"]), 3),
            "macd_hist": None if pd.isna(r["macd_hist"]) else round(float(r["macd_hist"]), 3),
            "bb_upper": None if pd.isna(r["bb_upper"]) else round(float(r["bb_upper"]), 3),
            "bb_lower": None if pd.isna(r["bb_lower"]) else round(float(r["bb_lower"]), 3),
            "obv": None if pd.isna(r["obv"]) else round(float(r["obv"]), 1),
            "vol_ma20": None if pd.isna(r["vol_ma20"]) else round(float(r["vol_ma20"]), 1),
        })
    return out


def load_prev_state():
    try:
        with open("prev_state.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def build_alerts(results, prev_state):
    alerts = []
    for t, r in results.items():
        prev = prev_state.get(t)
        if not prev:
            continue
        signal = r["signal"]
        risk = r["risk"]
        price = r["meta"]["price"]
        is_holding = r["group"] == "holding"

        if prev.get("label") != signal["label"]:
            alerts.append(f"{t}：訊號從「{prev.get('label')}」轉為「{signal['label']}」（分數 {prev.get('score')} → {signal['score']}）")

        if is_holding and risk:
            prev_price = prev.get("price")
            prev_risk = prev.get("risk") or {}
            if prev_price is not None and prev_risk.get("stop_loss") is not None:
                was_below_stop = prev_price < prev_risk["stop_loss"]
                now_below_stop = price < risk["stop_loss"]
                if now_below_stop and not was_below_stop:
                    alerts.append(f"{t}（持股）：現價 {price} 已跌破參考停損價 {risk['stop_loss']}")

            if prev_price is not None and prev_risk.get("take_profit") is not None:
                was_above_target = prev_price > prev_risk["take_profit"]
                now_above_target = price > risk["take_profit"]
                if now_above_target and not was_above_target:
                    alerts.append(f"{t}（持股）：現價 {price} 已站上參考停利價 {risk['take_profit']}")

    return alerts


def save_state(results):
    state = {}
    for t, r in results.items():
        state[t] = {
            "label": r["signal"]["label"],
            "score": r["signal"]["score"],
            "price": r["meta"]["price"],
            "risk": r["risk"],
        }
    with open("prev_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def _reason_rank(reason):
    # Lower rank = more distinctive / informative, shown first in recommendation cards.
    # The generic trend factors (weekly trend, SMA50/200 alignment) fire for almost every
    # bullish stock, so surfacing them first made every recommendation card look identical.
    text = reason["text"] if isinstance(reason, dict) else reason
    markers = [
        ("分析師評等", 0),
        ("近期新聞", 0),
        ("背離", 1),
        ("RSI14", 1),
        ("MACD", 1),
        ("成交量", 1),
        ("走勢優於大盤", 2),
        ("跌破布林", 1),
        ("突破布林", 1),
        ("黃金交叉", 2),
        ("死亡交叉", 2),
        ("週線均線", 3),
        ("均線在", 3),
    ]
    for marker, rank in markers:
        if marker in text:
            return rank
    return 4


def build_recommendations(results, watchlist_us, watchlist_hk, top_n=3, min_score=2):
    def top_for(tickers):
        candidates = []
        for t in tickers:
            r = results.get(t)
            if not r:
                continue
            score = r["signal"]["score"]
            if score < min_score:
                continue
            rel_pct = (r.get("relative_strength") or {}).get("relative_pct", 0) or 0
            candidates.append((score, rel_pct, t))
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        picks = []
        for score, rel_pct, t in candidates[:top_n]:
            r = results[t]
            ranked_reasons = sorted(r["signal"]["reasons_bull"], key=_reason_rank)
            horizons = r["signal"].get("horizons") or {}
            strong_horizons = [k for k, v in horizons.items() if v["score"] >= 1]
            picks.append({
                "ticker": t,
                "price": r["meta"]["price"],
                "currency": r["meta"]["currency"],
                "score": score,
                "label": r["signal"]["label"],
                "top_reasons": ranked_reasons[:2],
                "strong_horizons": strong_horizons,
                "risk": r.get("risk"),
            })
        return picks

    return {
        "us": top_for(watchlist_us),
        "hk": top_for(watchlist_hk),
    }


def main():
    prev_state = load_prev_state()
    try:
        with open("fund.json", encoding="utf-8") as f:
            fund_all = json.load(f)
    except FileNotFoundError:
        fund_all = {}
    try:
        with open("news.json", encoding="utf-8") as f:
            news_all = json.load(f)
    except FileNotFoundError:
        news_all = {}

    needed_markets = {market_of(t) for t in TICKERS}
    bench_cache = {}
    for key in needed_markets:
        bdf, _ = fetch_chart(BENCHMARK[key])
        bench_cache[key] = bdf

    factor_accuracy_accumulator = {}
    predictions = load_predictions()
    baseline_predictions = load_baseline_predictions()

    results = {}
    for t in TICKERS:
        try:
            df, meta = fetch_chart(t)
        except Exception as e:
            print(t, "FAILED to fetch:", e)
            continue
        df = compute_indicators(df)
        rel = compute_relative_strength(df, bench_cache[market_of(t)])
        signal = build_signal(df, rel, fund_all.get(t), news_all.get(t))
        factor_df = compute_rolling_technical_score(df, bench_cache[market_of(t)])
        backtest = backtest_composite_signal(df, factor_df, market=market_of(t))
        accumulate_factor_accuracy(df, factor_df, factor_accuracy_accumulator)
        # Use this ticker's own last trading-day date (not wall-clock "today") as the
        # prediction's made_date, so a run on a weekend/holiday reuses the prior trading
        # day's date instead of stamping a date that never appears in the price series
        # (which would leave that prediction permanently unresolvable).
        last_trading_date_str = df["date"].iloc[-1].strftime("%Y-%m-%d")
        resolve_ticker_predictions(t, df, predictions)
        make_new_prediction(t, last_trading_date_str, float(df.iloc[-1]["close"]), signal, predictions)
        short_tracking = build_short_tracking(t, predictions)
        resolve_baseline_predictions(t, df, baseline_predictions)
        make_baseline_prediction(t, last_trading_date_str, df, baseline_predictions)
        baseline_tracking = build_baseline_tracking(t, baseline_predictions)
        short_view = build_effective_short_view(short_tracking, baseline_tracking)
        risk = build_risk_levels(df)
        last = df.iloc[-1]
        results[t] = {
            "group": "holding" if t in HOLDINGS else "watchlist",
            "market": market_of(t),
            "meta": {
                "name": meta.get("longName") or meta.get("shortName") or t,
                "currency": meta.get("currency"),
                "exchange": meta.get("fullExchangeName"),
                "price": round(float(last["close"]), 3),
                "fifty_two_high": meta.get("fiftyTwoWeekHigh"),
                "fifty_two_low": meta.get("fiftyTwoWeekLow"),
                "benchmark_name": "S&P 500" if market_of(t) == "US" else "恒生指數",
            },
            "signal": signal,
            "backtest": backtest,
            "risk": risk,
            "relative_strength": rel,
            "short_tracking": short_tracking,
            "baseline_tracking": baseline_tracking,
            "short_view": short_view,
            "series": to_records(df),
        }
        print(t, "done, rows:", len(df), "signal:", signal["label"], signal["score"])

    with open("analysis_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    recommendations = build_recommendations(results, WATCHLIST_US, WATCHLIST_HK)
    with open("recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False)

    alerts = build_alerts(results, prev_state)
    with open("alerts.json", "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False)
    save_state(results)

    factor_accuracy = summarize_factor_accuracy(factor_accuracy_accumulator)
    with open("factor_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(factor_accuracy, f, ensure_ascii=False, indent=2)
    weak = [r["name"] for r in factor_accuracy if r["weak"]]
    print("factor accuracy report: ", len(factor_accuracy), "factors,", len(weak), "flagged weak:", weak)

    save_predictions(predictions)
    prediction_summary = summarize_predictions(predictions)
    with open("prediction_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(prediction_summary, f, ensure_ascii=False, indent=2)
    print("predictions: total=", len(predictions), "pending=", prediction_summary["total_pending"],
          "by_horizon=", [(h["horizon"], h["n"], h["hit_rate_pct"]) for h in prediction_summary["by_horizon"]])

    save_baseline_predictions(baseline_predictions)
    baseline_accuracy = summarize_baseline_accuracy(baseline_predictions)
    with open("baseline_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(baseline_accuracy, f, ensure_ascii=False, indent=2)
    print("baseline (naive persistence) accuracy: n=", baseline_accuracy["n"], "hit_rate_pct=", baseline_accuracy["hit_rate_pct"])

    print("wrote analysis_output.json, tickers:", len(results))
    print("alerts:", len(alerts))
    for a in alerts:
        print(" -", a)


if __name__ == "__main__":
    main()
