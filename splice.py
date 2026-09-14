import json
import io
import os
from datetime import datetime, timezone, timedelta

with io.open("analysis_output.json", "r", encoding="utf-8") as f:
    data_json_text = f.read()
    DATA = json.loads(data_json_text)

with io.open("watchlist.json", "r", encoding="utf-8") as f:
    WL = json.load(f)
HOLDINGS = [t for t in WL.get("holdings", []) if t in DATA]
WATCHLIST_US = [t for t in WL.get("watchlist_us", []) if t in DATA]
WATCHLIST_HK = [t for t in WL.get("watchlist_hk", []) if t in DATA]
WATCHLIST = WATCHLIST_US + WATCHLIST_HK
ORDER = HOLDINGS + WATCHLIST

try:
    with io.open("fund.json", "r", encoding="utf-8") as f:
        FUND = json.load(f)
except FileNotFoundError:
    FUND = {}

try:
    with io.open("news.json", "r", encoding="utf-8") as f:
        NEWS = json.load(f)
except FileNotFoundError:
    NEWS = {}

try:
    with io.open("recommendations.json", "r", encoding="utf-8") as f:
        RECOMMENDATIONS = json.load(f)
except FileNotFoundError:
    RECOMMENDATIONS = {"us": [], "hk": []}

try:
    with io.open("factor_accuracy.json", "r", encoding="utf-8") as f:
        FACTOR_ACCURACY = json.load(f)
except FileNotFoundError:
    FACTOR_ACCURACY = []

try:
    with io.open("prediction_accuracy.json", "r", encoding="utf-8") as f:
        PREDICTION_ACCURACY = json.load(f)
except FileNotFoundError:
    PREDICTION_ACCURACY = None

try:
    with io.open("baseline_accuracy.json", "r", encoding="utf-8") as f:
        BASELINE_ACCURACY = json.load(f)
except FileNotFoundError:
    BASELINE_ACCURACY = None

last_date = max(DATA[t]["series"][-1]["date"] for t in ORDER if DATA[t]["series"])
tw_now = datetime.now(timezone(timedelta(hours=8)))
GENERATED_AT = f"{tw_now.strftime('%Y-%m-%d %H:%M')}（台北時間，資料截至 {last_date} 收盤，由GitHub Actions自動更新，不含持股部位/基本面/新聞）"

with io.open("template.html", "r", encoding="utf-8") as f:
    tpl = f.read()

out = tpl.replace("__DATA_JSON__", data_json_text)
out = out.replace("__FUND_JSON__", json.dumps(FUND, ensure_ascii=False))
out = out.replace("__NEWS_JSON__", json.dumps(NEWS, ensure_ascii=False))
out = out.replace("__ORDER_JSON__", json.dumps(ORDER, ensure_ascii=False))
out = out.replace("__HOLDINGS_JSON__", json.dumps(HOLDINGS, ensure_ascii=False))
out = out.replace("__WATCHLIST_JSON__", json.dumps(WATCHLIST, ensure_ascii=False))
out = out.replace("__WATCHLIST_US_JSON__", json.dumps(WATCHLIST_US, ensure_ascii=False))
out = out.replace("__WATCHLIST_HK_JSON__", json.dumps(WATCHLIST_HK, ensure_ascii=False))
out = out.replace("__RECOMMENDATIONS_JSON__", json.dumps(RECOMMENDATIONS, ensure_ascii=False))
out = out.replace("__FACTOR_ACCURACY_JSON__", json.dumps(FACTOR_ACCURACY, ensure_ascii=False))
out = out.replace("__PREDICTION_ACCURACY_JSON__", json.dumps(PREDICTION_ACCURACY, ensure_ascii=False))
out = out.replace("__BASELINE_ACCURACY_JSON__", json.dumps(BASELINE_ACCURACY, ensure_ascii=False))
out = out.replace("__GENERATED_AT__", json.dumps(GENERATED_AT, ensure_ascii=False))

os.makedirs("dist", exist_ok=True)
with io.open("dist/index.html", "w", encoding="utf-8") as f:
    f.write(out)

print("wrote dist/index.html, bytes:", len(out.encode("utf-8")))
