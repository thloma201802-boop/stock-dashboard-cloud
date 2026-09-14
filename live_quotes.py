import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

from fetch_analyze import fetch_chart, market_of, TICKERS

US_TZ = ZoneInfo("America/New_York")
HK_TZ = ZoneInfo("Asia/Hong_Kong")


def is_us_market_open(now_utc):
    now = now_utc.astimezone(US_TZ)
    if now.weekday() >= 5:
        return False
    return time(9, 30) <= now.time() <= time(16, 0)


def is_hk_market_open(now_utc):
    now = now_utc.astimezone(HK_TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (time(9, 30) <= t <= time(12, 0)) or (time(13, 0) <= t <= time(16, 0))


def main():
    now_utc = datetime.now(ZoneInfo("UTC"))
    active_markets = set()
    if is_us_market_open(now_utc):
        active_markets.add("US")
    if is_hk_market_open(now_utc):
        active_markets.add("HK")

    if not active_markets:
        print("no tracked market currently open, skipping live quote fetch")
        with open("live_quotes.json", "w", encoding="utf-8") as f:
            json.dump({"generated_at": now_utc.isoformat(), "markets_open": [], "quotes": {}}, f)
        return

    tickers = [t for t in TICKERS if market_of(t) in active_markets]
    quotes = {}
    for t in tickers:
        try:
            _, meta = fetch_chart(t, range_="5d", interval="1d")
            price = meta.get("regularMarketPrice")
            prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
            change_pct = None
            if price is not None and prev_close:
                change_pct = round((price / prev_close - 1) * 100, 2)
            quotes[t] = {"price": round(price, 3) if price is not None else None, "change_pct": change_pct}
        except Exception as e:
            print(t, "live quote fetch failed:", e)

    out = {
        "generated_at": now_utc.isoformat(),
        "markets_open": sorted(active_markets),
        "quotes": quotes,
    }
    with open("live_quotes.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote live_quotes.json:", len(quotes), "tickers, markets open:", sorted(active_markets))


if __name__ == "__main__":
    main()
