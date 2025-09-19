#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a synthetic as_of snapshot for 2025-09-14 by perturbing 2025-09-13.

- Input files (expected in ../data):
    yt_brand_daily_panel.csv  (columns: date, brand, as_of_date_utc, video_mentions, views, likes, comments, top_channels)
    yt_brand_roll7_daily.csv  (columns: report_date_utc, brand, roll7_video_mentions, roll7_views, roll7_likes, roll7_comments, roll7_top_channels)

- Output:
    Appends a new 7×brand×day block to panel with as_of_date_utc=2025-09-14
    Appends a new roll-7 row per brand for report_date_utc=2025-09-14

Notes:
- We DO NOT backfill per-video stats; this is clearly a synthetic snapshot.
- Keep a disclosure in the UI that 2025-09-14 is imputed.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("../data")
PANEL_CSV = DATA_DIR / "yt_brand_daily_panel.csv"
ROLL7_CSV = DATA_DIR / "yt_brand_roll7_daily.csv"

TARGET_ASOF = pd.to_datetime("2025-09-14").date()
BASE_ASOF = pd.to_datetime("2025-09-13").date()

# Window for T=2025-09-14 is [2025-09-08 .. 2025-09-14]
WIN_START = pd.to_datetime("2025-09-08").date()
WIN_END   = pd.to_datetime("2025-09-14").date()
PREV_LAST = pd.to_datetime("2025-09-13").date()

# --- Helpers to find best available baseline rows ---
def best_row_for(panel_df: pd.DataFrame, brand: str, day: pd.Timestamp, base_asof: pd.Timestamp):
    """Return the most recent row for (brand, day) with as_of <= base_asof. None if not found."""
    sub = panel_df[(panel_df["brand"] == brand) & (panel_df["date"] == day) & (panel_df["as_of_date_utc"] <= base_asof)]
    if sub.empty:
        return None
    # pick the latest as_of available up to base_asof
    sub = sub.sort_values("as_of_date_utc").iloc[-1]
    return sub

def safe_metrics_from_row(row, fallback: dict):
    if row is None:
        return fallback.copy()
    return {
        "video_mentions": int(pd.to_numeric(row.get("video_mentions", fallback["video_mentions"]), errors="coerce") or fallback["video_mentions"]),
        "views": int(pd.to_numeric(row.get("views", fallback["views"]), errors="coerce") or fallback["views"]),
        "likes": int(pd.to_numeric(row.get("likes", fallback["likes"]), errors="coerce") or fallback["likes"]),
        "comments": int(pd.to_numeric(row.get("comments", fallback["comments"]), errors="coerce") or fallback["comments"]),
        "top_channels": str(row.get("top_channels", fallback.get("top_channels", ""))) if row is not None else fallback.get("top_channels", ""),
    }

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def clip_int(x):
    # safe integer casting for engagement metrics
    return int(max(0, round(x)))

def draw_multiplier(mu: float, sigma: float, low: float, high: float, size: int) -> np.ndarray:
    vals = np.random.normal(loc=mu, scale=sigma, size=size)
    return np.clip(vals, low, high)

def main(args):
    np.random.seed(args.seed)

    panel = load_csv(PANEL_CSV)
    # Basic schema check
    need_cols = {"date","brand","as_of_date_utc","video_mentions","views","likes","comments","top_channels"}
    missing = need_cols - set(panel.columns)
    if missing:
        raise ValueError(f"{PANEL_CSV} missing columns: {sorted(missing)}")

    # Normalize types
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.date
    panel["as_of_date_utc"] = pd.to_datetime(panel["as_of_date_utc"], errors="coerce").dt.date
    for m in ["video_mentions","views","likes","comments"]:
        panel[m] = pd.to_numeric(panel[m], errors="coerce").fillna(0).astype(int)
    panel["brand"] = panel["brand"].astype(str)
    panel["top_channels"] = panel["top_channels"].fillna("").astype(str)

    # Brand universe (all brands in the file, or restricted by --brands)
    brands_all = sorted(panel["brand"].unique())
    if args.brands:
        keep = set([b.strip().lower() for b in args.brands.split(",") if b.strip()])
        brands_all = [b for b in brands_all if b.lower() in keep]
        if not brands_all:
            raise ValueError("After filtering by brands, no brands remain.")

    # Precompute simple fallbacks
    global_medians = {
        "video_mentions": int(panel["video_mentions"].median() if not panel["video_mentions"].empty else 0),
        "views": int(panel["views"].median() if not panel["views"].empty else 0),
        "likes": int(panel["likes"].median() if not panel["likes"].empty else 0),
        "comments": int(panel["comments"].median() if not panel["comments"].empty else 0),
        "top_channels": "",
    }
    brand_medians = (
        panel.groupby("brand")[["video_mentions","views","likes","comments"]]
              .median().rename(columns=lambda x: f"med_{x}")
              .reset_index()
    )

    # --- Build a full 7-day window for TARGET_ASOF covering ALL brands ---
    days = pd.date_range(WIN_START, PREV_LAST, freq="D").date
    rows_prior6 = []
    rows_last   = []

    for b in brands_all:
        # brand-specific median fallback
        bm_row = brand_medians.loc[brand_medians["brand"] == b]
        b_fallback = {
            "video_mentions": int(bm_row["med_video_mentions"].iloc[0]) if not bm_row.empty else global_medians["video_mentions"],
            "views": int(bm_row["med_views"].iloc[0]) if not bm_row.empty else global_medians["views"],
            "likes": int(bm_row["med_likes"].iloc[0]) if not bm_row.empty else global_medians["likes"],
            "comments": int(bm_row["med_comments"].iloc[0]) if not bm_row.empty else global_medians["comments"],
            "top_channels": "",
        }
        # 6 prior days: prefer rows from BASE_ASOF; if missing, pick most recent as_of <= BASE_ASOF; else fallback to medians with zeros allowed
        for d in days:
            br = best_row_for(panel, b, d, BASE_ASOF)
            met = safe_metrics_from_row(br, b_fallback)
            rows_prior6.append({
                "date": d,
                "brand": b,
                "as_of_date_utc": TARGET_ASOF,
                "video_mentions": int(met["video_mentions"]),
                "views": int(met["views"]),
                "likes": int(met["likes"]),
                "comments": int(met["comments"]),
                "top_channels": met.get("top_channels", ""),
            })

        # Baseline for 09-13 to synthesize 09-14
        base13 = best_row_for(panel, b, PREV_LAST, BASE_ASOF)
        if base13 is None:
            # search up to 7 days back for a baseline
            found = None
            for lag in range(1, 8):
                cand_day = PREV_LAST - pd.Timedelta(days=lag)
                found = best_row_for(panel, b, cand_day.date(), BASE_ASOF)
                if found is not None:
                    base13 = found
                    break
        # If still none, use brand median fallback
        base_metrics = safe_metrics_from_row(base13, b_fallback)

        rows_last.append({
            "brand": b,
            "base_metrics": base_metrics
        })

    # Create synthetic 09-14 rows by perturbing per-brand baselines
    n = len(rows_last)
    m_views    = draw_multiplier(args.mu, args.sigma, args.low, args.high, n)
    m_likes    = draw_multiplier(args.mu, args.sigma, args.low, args.high, n)
    m_comments = draw_multiplier(args.mu, args.sigma, args.low, args.high, n)
    m_mentions = np.random.choice([0.9, 1.0, 1.1], size=n, p=[0.25, 0.5, 0.25])

    synth_rows = []
    for i, row in enumerate(rows_last):
        b = row["brand"]
        base = row["base_metrics"]
        views    = clip_int(base["views"]    * m_views[i])
        likes    = clip_int(base["likes"]    * m_likes[i])
        comments = clip_int(base["comments"] * m_comments[i])
        mentions = max(0, int(round(base["video_mentions"] * m_mentions[i])))
        synth_rows.append({
            "date": WIN_END,
            "brand": b,
            "as_of_date_utc": TARGET_ASOF,
            "video_mentions": mentions,
            "views": views,
            "likes": likes,
            "comments": comments,
            "top_channels": "",
        })

    new_panel = pd.DataFrame(rows_prior6 + synth_rows)
    # Ensure one row per (brand,date,as_of)
    new_panel = (new_panel
                 .sort_values(["brand","date"])
                 .drop_duplicates(subset=["brand","date","as_of_date_utc"], keep="last"))

    panel_out = panel[panel["as_of_date_utc"] != TARGET_ASOF].copy()
    panel_out = pd.concat([panel_out, new_panel], ignore_index=True)

    # Write panel
    save_csv(panel_out, PANEL_CSV)
    print(f"✓ Wrote synthetic panel for as_of={TARGET_ASOF} (rows added: {len(new_panel)})")

    # ---- Recompute roll-7 for TARGET_ASOF and write ----
    # roll-7 is brand-level sum across the 7-day window we just created
    roll7_block = (new_panel.groupby("brand", as_index=False)
                   .agg({
                       "video_mentions": "sum",
                       "views": "sum",
                       "likes": "sum",
                       "comments": "sum"
                   })
                   .rename(columns={
                       "video_mentions": "roll7_video_mentions",
                       "views": "roll7_views",
                       "likes": "roll7_likes",
                       "comments": "roll7_comments"
                   }))
    roll7_block["report_date_utc"] = TARGET_ASOF
    roll7_block["roll7_top_channels"] = ""  # leave empty or compute if you want

    # Load existing roll7 and replace target date rows idempotently
    if ROLL7_CSV.exists():
        roll7_all = pd.read_csv(ROLL7_CSV)
        roll7_all = roll7_all[roll7_all["report_date_utc"] != str(TARGET_ASOF)]
        roll7_all = pd.concat([roll7_all, roll7_block], ignore_index=True)
    else:
        cols = ["report_date_utc","brand","roll7_video_mentions","roll7_views","roll7_likes","roll7_comments","roll7_top_channels"]
        roll7_all = roll7_block[cols]

    save_csv(roll7_all, ROLL7_CSV)
    print(f"✓ Wrote synthetic roll7 for report_date_utc={TARGET_ASOF} (brands: {roll7_block['brand'].nunique()})")

    if args.echo_preview:
        print("\n--- Preview (first 10 synthetic rows) ---")
        print(new_panel.sort_values(['brand','date']).head(10).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create a synthetic as_of snapshot for 2025-09-14")
    ap.add_argument("--brands", type=str, default="", help="Comma-separated subset of brands to include (optional).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--mu", type=float, default=1.00, help="Mean multiplier for views/likes/comments.")
    ap.add_argument("--sigma", type=float, default=0.12, help="Stddev for multiplier.")
    ap.add_argument("--low", type=float, default=0.75, help="Lower clip bound for multiplier.")
    ap.add_argument("--high", type=float, default=1.35, help="Upper clip bound for multiplier.")
    ap.add_argument("--echo-preview", action="store_true", help="Print a small preview of synthetic rows.")
    args = ap.parse_args()
    main(args)