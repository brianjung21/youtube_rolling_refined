"""
Rolling1: Moving 7-day DAILY panel with as-of updates (1-day lag).

Each run sets T = (today UTC - 1 day) by default and :
    1) DISCOVERY (T only): find videos published on T (optionally +- cushion days),
    brand-match locally, and append NEW rows to the Video Registry (CSV).
    2) STATS REFRESH for [T-6 .. T]: fetch current stats for all registry videos whose
    published_at_utc is inside the window; write rows (video_id, as_of_date_utc=T).
    3) REBUILD 7-DAY DAILY PANEL for [T-6 .. T] using stats (as_of=T),
    recompute top_channels (per day, per brand), append rows keyed by (date, brand, as_of_date_utc=T).

Notes:
    - Everything is UTC.
    - Window is inclusive: [T-6 .. T].
    - We DO NOT re-discover for older dates; we only refresh stats for already-known video_ids in the window.
    - Start now -> we'll accumulate proper as-of history going forward.
"""

import re
import time
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Iterable, Tuple
import pandas as pd
import requests


# --- API key rotation support ---
API_KEYS = [

]
_current_key_idx = 0

def get_api_key() -> str:
    return API_KEYS[_current_key_idx]

def rotate_api_key() -> bool:
    """Advance to the next API key. Returns True if switched, False if none left."""
    global _current_key_idx
    if _current_key_idx < len(API_KEYS) - 1:
        _current_key_idx += 1
        print(f"[api-key] Rotating to key #{_current_key_idx+1}/{len(API_KEYS)}")
        return True
    return False

KEYWORDS_PATH = Path('../files/kpop_keywords_sample.csv')

DATA_DIR = Path('../data')
REGISTRY_CSV = DATA_DIR / "yt_video_registry.csv"
STATS_CSV = DATA_DIR / "yt_video_stats_daily.csv"
PANEL_CSV = DATA_DIR / "yt_brand_daily_panel.csv"
ROLL7_CSV = DATA_DIR / "yt_brand_roll7_daily.csv"

REGION_CODE = None
RELEVANCE_LANGUAGE = None
DISCOVERY_CUSHION_DAYS = 0

INCLUDE_TAGS = True
USE_KO_BOUNDARY = False

MAX_RETRIES = 5
BACKOFF_BASE = 0.5
BACKOFF_MAX = 8.0
PAGE_SLEEP_SEC = 1.0
BRAND_SLEEP_SEC = 0.5
SEARCH_PAGE_CAP = 10
CONTINUE_ON_QUOTA_ERRORS = True

BASE = "https://www.googleapis.com/youtube/v3"


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).lower()


def brand_to_pattern(kw: str) -> re.Pattern:
    tokens = [re.escape(t) for t in kw.split() if t]
    if not tokens:
        return re.compile(r"$^")
    sep = r"[-_\.\s]*"
    core = sep.join(tokens)
    if USE_KO_BOUNDARY:
        left, right = r"(?<![0-9A-Za-z가-힣])", r"(?![0-9A-Za-z가-힣])"
    else:
        left, right = r"(?<!\w)", r"(?!\w)"
    return re.compile(rf"{left}{core}{right}", re.IGNORECASE)


def load_keywords(csv_path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Keyword" not in df.columns:
        raise ValueError("Expected a header column named 'Keyword'")
    if "Aliases" not in df.columns:
        df["Aliases"] = ""
    out = {}
    for _, r in df.iterrows():
        main = str(r["Keyword"]).strip().lower()
        if not main:
            continue
        alts = [a.strip().lower() for a in str(r["Aliases"]).split("|") if a.strip()]
        out[main] = [main] + alts
    return out


def iso8601(dt: datetime) -> str:
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _sleep_backoff(attempt: int):
    delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
    time.sleep(delay)


def get_json(url: str, params: dict) -> dict:
    quota_reasons = {"quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded", "userRateLimitExceeded"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            # Retryable HTTPs first
            if resp.status_code in (429, 500, 502, 503, 504):
                # On 429 (often rate limited), try rotating key immediately
                if resp.status_code == 429 and rotate_api_key():
                    params["key"] = get_api_key()
                    continue
                if attempt < MAX_RETRIES - 1:
                    _sleep_backoff(attempt)
                    continue
            if resp.status_code >= 400:
                reason = None
                message = None
                try:
                    err = resp.json().get("error", {})
                    errs = err.get("errors", [])
                    if errs:
                        reason = errs[0].get("reason")
                        message = errs[0].get("message")
                except Exception:
                    message = resp.text[:300]
                # If quota/rate, rotate key and retry
                if (resp.status_code in (403, 429)) and (reason in quota_reasons or (message and any(q in message for q in quota_reasons))):
                    print(f"[get_json] Quota/rate error: reason={reason} message={message}")
                    if rotate_api_key():
                        params["key"] = get_api_key()
                        continue
                raise RuntimeError(f"YouTube API {resp.status_code}: reason={reason} message={message}")
            return resp.json()
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                _sleep_backoff(attempt)
                continue
            raise


def yt_search_pages(q: str, published_after: str, published_before: str) -> Iterable[dict]:
    params = {
        "key": get_api_key(),
        "part": "snippet",
        "q": q,
        "type": "video",
        "order": "date",
        "maxResults": 50,
        "publishedAfter": published_after,
        "publishedBefore": published_before,
    }
    if REGION_CODE:
        params['regionCode'] = REGION_CODE
    if RELEVANCE_LANGUAGE:
        params['relevanceLanguage'] = RELEVANCE_LANGUAGE

    token = None
    page_count = 0
    while True:
        # Ensure we always send the current (possibly rotated) key for each page
        params["key"] = get_api_key()
        if token:
            params['pageToken'] = token
        try:
            data = get_json(f"{BASE}/search", params)
        except RuntimeError as e:
            msg = str(e)
            if CONTINUE_ON_QUOTA_ERRORS and any(k in msg for k in ['quotaExceeded', 'rateLimitExceeded', 'dailyLimitExceeded', 'userRateLimitExceeded']):
                print(f"[search] Quota/rate for query '{q}': {msg}. Truncating search.")
                break
            raise

        for it in data.get("items", []):
            yield it
        token = data.get("nextPageToken")
        page_count += 1
        if not token:
            break
        if page_count >= SEARCH_PAGE_CAP:
            print(f"[search] Reached SEARCH_PAGE_CAP={SEARCH_PAGE_CAP} for '{q}'.")
            break
        time.sleep(PAGE_SLEEP_SEC)


def yt_videos_details(video_ids: List[str]) -> dict:
    out = {}
    if not video_ids:
        return out
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        params = {
            "key": get_api_key(),
            "part": "snippet,statistics",
            "id": ",".join(chunk),
            "maxResults": 50
        }
        data = get_json(f"{BASE}/videos", params)
        for it in data.get("items", []):
            vid = it.get('id')
            sn = it.get("snippet", {}) or {}
            st = it.get("statistics", {}) or {}
            out[vid] = {
                "tags": sn.get("tags", []) or [],
                "likeCount": int(st.get("likeCount", 0) or 0),
                "commentCount": int(st.get("commentCount", 0) or 0),
                "viewCount": int(st.get('viewCount', 0) or 0),
                "channelId": sn.get("channelId", ""),
                "channelTitle": sn.get("channelTitle", ""),
                "publishedAt": sn.get("publishedAt", "")
            }
        time.sleep(PAGE_SLEEP_SEC)
    return out


def yt_channels_stats(channel_ids: List[str]) -> dict:
    out = {}
    if not channel_ids:
        return out
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i+50]
        params = {
            "key": get_api_key(),
            "part": "statistics",
            "id": ",".join(chunk),
            "maxResults": 50,
        }
        data = get_json(f"{BASE}/channels", params)
        for it in data.get("items", []):
            cid = it.get("id")
            st = it.get("statistics", {}) or {}
            out[cid] = {
                "subscribers": int(st.get("subscriberCount", 0) or 0),
                "videoCount": int(st.get("videoCount", 0) or 0)
            }
        time.sleep(PAGE_SLEEP_SEC)
    return out


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df


def _read_csv(path: Path, cols:List[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8")
        return _ensure_cols(df, cols)
    return pd.DataFrame(columns=cols)


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def day_bounds_utc(d: date) -> Tuple[str, str]:
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return iso8601(start), iso8601(end)


def discover_for_day(T: date, kw_map: Dict[str, List[str]]) -> pd.DataFrame:
    patterns = {k: [brand_to_pattern(v) for v in vs] for k, vs in kw_map.items()}

    dates_to_search = [T]
    for i in range(1, DISCOVERY_CUSHION_DAYS + 1):
        dates_to_search.extend([T - timedelta(days=i), T + timedelta(days=i)])
    dates_to_search = sorted(set(dates_to_search))

    new_rows = []
    for day in dates_to_search:
        after, before = day_bounds_utc(day)
        for main_kw, pats in patterns.items():
            hits = []
            vids = []
            for item in yt_search_pages(main_kw, after, before):
                vid = item['id']['videoId']
                sn = item.get('snippet', {}) or {}
                hits.append({
                    "video_id": vid,
                    "title": sn.get("title", ""),
                    "description": sn.get("description", ""),
                    "channelTitle": sn.get("channelTitle", "")
                })
                vids.append(vid)
            id2det = yt_videos_details(vids) if vids else {}
            for h in hits:
                det = id2det.get(h["video_id"], {})
                tags = det.get("tags", []) if INCLUDE_TAGS else []
                text_norm = normalize_text(f"{h['title']}\n{h['description']}\n{' '.join(tags)}")
                if any(p.search(text_norm) for p in pats):
                    pub_iso = det.get('publishedAt') or after
                    pub_dt = datetime.fromisoformat(pub_iso.replace("Z", "+00:00")).date()
                    if pub_dt != day:
                        pub_dt = day
                    new_rows.append({
                        "video_id": h["video_id"],
                        "brand": main_kw,
                        "published_at_utc": pub_dt.isoformat(),
                        "channel_id": det.get("channelId", ""),
                        "channel_title": det.get("channelTitle", h.get("channelTitle", "")),
                        "first_seen_utc": datetime.now(timezone.utc).isoformat(),
                        "matched_fields": "title|description|tags" if INCLUDE_TAGS else "title|description"
                    })
        time.sleep(BRAND_SLEEP_SEC)
    return pd.DataFrame(new_rows).drop_duplicates(subset=["brand", "video_id"])


def refresh_stats_for_window(T: date, registry: pd.DataFrame) -> pd.DataFrame:
    win_start = T - timedelta(days=6)
    reg = registry.copy()
    reg['published_at_utc'] = pd.to_datetime(reg['published_at_utc']).dt.date
    mask = (reg['published_at_utc'] >= win_start) & (reg['published_at_utc'] <= T)
    vids = reg.loc[mask, "video_id"].dropna().unique().tolist()

    rows = []
    for i in range(0, len(vids), 50):
        chunk = vids[i:i+50]
        det = yt_videos_details(chunk)
        for vid in chunk:
            d = det.get(vid, {}) or {}
            rows.append({
                "video_id": vid,
                "as_of_date_utc": T.isoformat(),
                "viewCount": int(d.get("viewCount", 0) or 0),
                "likeCount": int(d.get("likeCount", 0) or 0),
                "commentCount": int(d.get("commentCount", 0) or 0)
            })
        time.sleep(PAGE_SLEEP_SEC)
    return pd.DataFrame(rows).drop_duplicates(subset=["video_id", "as_of_date_utc"])


def compute_top_channels_per_day(day_df: pd.DataFrame) -> str:
    if day_df.empty:
        return ""
    tmp = day_df.copy()
    for col in ['subscribers', 'channel_video_count', 'likeCount', 'commentCount']:
        tmp[col] = tmp[col].fillna(0)
        tmp[f"r_{col}"] = tmp[col].rank(method='min', ascending=False)
    tmp['rank_sum'] = tmp[['r_subscribers', 'r_channel_video_count', 'r_likeCount', 'r_commentCount']].sum(axis=1)
    tmp = tmp.sort_values(['rank_sum', 'subscribers', 'likeCount'], ascending=[True, False, False])
    return ";".join(tmp['channel_title'].head(3).tolist())


def rebuild_daily_panel(T: date, registry: pd.DataFrame, stats_today: pd.DataFrame) -> pd.DataFrame:
    win_days = [(T - timedelta(days=i)) for i in range(6, -1, -1)]
    reg = registry.copy()
    reg['published_at_utc'] = pd.to_datetime(reg['published_at_utc']).dt.date
    stats_today = stats_today[["video_id", "viewCount", "likeCount", "commentCount"]].copy()
    joined = reg.merge(stats_today, on="video_id", how="left").fillna(
        {"viewCount": 0, "likeCount": 0, "commentCount": 0})

    mask = reg["published_at_utc"].isin(win_days)
    ch_ids = joined.loc[mask, "channel_id"].dropna().unique().tolist()
    ch_map = yt_channels_stats(ch_ids) if ch_ids else {}

    rows = []
    for d in win_days:
        day_rows = joined[joined["published_at_utc"] == d]
        for b in sorted(day_rows["brand"].unique()):
            sub = day_rows[day_rows["brand"] == b].copy()
            if sub.empty:
                continue
            # per-channel aggregates (for ranking)
            ch_frame = (sub.groupby(["channel_id", "channel_title"], as_index=False)
                        .agg({"likeCount": "sum", "commentCount": "sum"}))
            ch_frame["subscribers"] = ch_frame["channel_id"].map(lambda c: ch_map.get(c, {}).get("subscribers", 0))
            ch_frame["channel_video_count"] = ch_frame["channel_id"].map(
                lambda c: ch_map.get(c, {}).get("videoCount", 0))
            top_channels = compute_top_channels_per_day(ch_frame)

            rows.append({
                "date": d.isoformat(),
                "brand": b,
                "as_of_date_utc": T.isoformat(),
                "video_mentions": int(sub["video_id"].nunique()),
                "views": int(sub["viewCount"].sum()),
                "likes": int(sub["likeCount"].sum()),
                "comments": int(sub["commentCount"].sum()),
                "top_channels": top_channels,
            })
    panel = pd.DataFrame(rows)

    all_days = [d.isoformat() for d in win_days]
    all_brands = sorted(registry["brand"].unique().tolist())
    if all_brands:
        idx = pd.MultiIndex.from_product([all_days, all_brands], names=["date", "brand"])
        panel = (panel.set_index(["date", "brand"])
                 .reindex(idx)
                 .reset_index())
        panel["as_of_date_utc"] = panel["as_of_date_utc"].fillna(T.isoformat())
        for c in ["video_mentions", "views", "likes", "comments", "top_channels"]:
            if c != "top_channels":
                panel[c] = panel[c].fillna(0).astype(int)
            else:
                panel[c] = panel[c].fillna("")
    return panel


# ── Rolling‑2 builder (no extra API calls) ─────────────────────────────────────
def build_roll7_from_panel(panel_T: pd.DataFrame, T: date) -> pd.DataFrame:
    """Given the 7×brand daily panel for as_of=T, compute one row per brand with 7‑day sums.
    Returns columns: report_date_utc, brand, roll7_video_mentions, roll7_views, roll7_likes, roll7_comments, roll7_top_channels.
    Note: roll7_top_channels left empty here to avoid extra API calls; can be added by reusing channel stats if desired.
    """
    if panel_T is None or panel_T.empty:
        return pd.DataFrame(columns=[
            "report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"
        ])
    agg = (panel_T.groupby("brand", as_index=False)
                  .agg({
                      "video_mentions": "sum",
                      "views": "sum",
                      "likes": "sum",
                      "comments": "sum",
                  }))
    agg = agg.rename(columns={
        "video_mentions": "roll7_video_mentions",
        "views": "roll7_views",
        "likes": "roll7_likes",
        "comments": "roll7_comments",
    })
    agg["report_date_utc"] = T.isoformat()
    agg["roll7_top_channels"] = ""  # optional: compute by aggregating per‑channel totals across 7 days
    # order columns
    cols = ["report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"]
    return agg[cols]


def run_once(as_of_date_utc: date):
    kw_map = load_keywords(KEYWORDS_PATH)
    if not kw_map:
        print("No keywords found; exiting.")
        return

    registry_cols = ["video_id", "brand", "published_at_utc", "channel_id", "channel_title", "first_seen_utc",
                     "matched_fields"]
    stats_cols = ["video_id", "as_of_date_utc", "viewCount", "likeCount", "commentCount"]
    panel_cols = ["date", "brand", "as_of_date_utc", "video_mentions", "views", "likes", "comments", "top_channels"]

    registry = _read_csv(REGISTRY_CSV, registry_cols)
    stats = _read_csv(STATS_CSV, stats_cols)
    panel = _read_csv(PANEL_CSV, panel_cols)

    T = as_of_date_utc

    new_reg = discover_for_day(T, kw_map)
    if not new_reg.empty:
        old_keys = set(zip(registry["brand"].astype(str), registry["video_id"].astype(str)))
        is_new = ~new_reg.apply(lambda r: (str(r["brand"]), str(r["video_id"])) in old_keys, axis=1)
        new_reg = new_reg[is_new]
        if not new_reg.empty:
            registry = pd.concat([registry, new_reg], ignore_index=True)
            print(f"[{T}] discovery: added {len(new_reg)} brand×video rows.")
        else:
            print(f"[{T}] discovery: no new rows (already known).")
    else:
        print(f"[{T}] discovery: no matches for today’s window.")

    stats_today = refresh_stats_for_window(T, registry)
    if not stats_today.empty:
        stats = stats[stats["as_of_date_utc"] != T.isoformat()]  # idempotent replace for this as_of
        stats = pd.concat([stats, stats_today], ignore_index=True)
    print(f"[{T}] stats: refreshed {len(stats_today)} video rows.")

    panel_T = rebuild_daily_panel(T, registry, stats_today)
    if not panel_T.empty:
        panel = panel[panel["as_of_date_utc"] != T.isoformat()]  # idempotent replace
        panel = pd.concat([panel, panel_T], ignore_index=True)
        print(f"[{T}] panel: wrote {len(panel_T)} day×brand rows.")
        # Rolling‑2: build and persist 7‑day rollup per brand for report_date_utc=T
        roll7 = build_roll7_from_panel(panel_T, T)
        # read existing roll7 (if any), idempotent replace for this report_date
        roll7_cols = ["report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"]
        roll7_all = _read_csv(ROLL7_CSV, roll7_cols)
        roll7_all = roll7_all[roll7_all["report_date_utc"] != T.isoformat()]
        roll7_all = pd.concat([roll7_all, roll7], ignore_index=True)
        _write_csv(roll7_all, ROLL7_CSV)
        print(f"[{T}] roll7: wrote {len(roll7)} brand rows.")
    else:
        print(f"[{T}] panel: no rows built for this window.")

    _write_csv(registry, REGISTRY_CSV)
    _write_csv(stats, STATS_CSV)
    _write_csv(panel, PANEL_CSV)

    print(f"✓ [{T}] Registry → {REGISTRY_CSV.resolve()} (rows={len(registry)})")
    print(f"✓ [{T}] Stats    → {STATS_CSV.resolve()} (rows={len(stats)})")
    print(f"✓ [{T}] Panel    → {PANEL_CSV.resolve()} (rows={len(panel)})")


def main():
    today_utc = datetime.now(timezone.utc).date()
    T = today_utc - timedelta(days=1)
    print(f"=== Rolling-1 run for as_of_date_utc = {T} (window [{T - timedelta(days=6)} .. {T}]) ===")
    run_once(T)


if __name__ == "__main__":
    main()
