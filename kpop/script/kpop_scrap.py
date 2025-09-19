"""
Explore Rolling-1 / Rolling-2 outputs from kpop_rolling_gathering.py
- Reads:
    - data/yt_brand_daily_panel.csv (Rolling-1 daily, 7-day window stamped as_of_date_utc)
    - data/yt_brand_roll7_daily.csv (Rolling-2 rollup, one row/brand per report_date_utc)
    - data/yt_video_registry.csv (optional drilldown)
    - data/yt_video_stats_daily.csv (optional drilldown)
- Lets you pick an as_of_date_utc (T), a metric, and brands to plot
- Shows analyst-focused views that avoid API page-cap pitfalls.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

DATA_DIR = (Path(__file__).parent / "../data").resolve()
PANEL_FILE = "yt_brand_daily_panel.csv"
ROLL7_FILE = "yt_brand_roll7_daily.csv"

METRIC_LABELS = {
    "video_mentions": "Video mentions",
    "views": "Views",
    "likes": "Likes",
    "comments": "Comments"
}
DEFAULT_TOPN = 8


def find_file(fname: str) -> Path:
    p = DATA_DIR / fname
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find {fname} in: {DATA_DIR}")


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    p = find_file(PANEL_FILE)
    df = pd.read_csv(p, parse_dates=["as_of_date_utc", "date"])
    for c in ["video_mentions", "views", "likes", "comments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["brand"] = df["brand"].astype(str)
    df = df.sort_values(["as_of_date_utc", "date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_roll7() -> pd.DataFrame:
    p = find_file(ROLL7_FILE)
    df = pd.read_csv(p, parse_dates=["report_date_utc"])
    for c in ["roll7_video_mentions", "roll7_views", "roll7_likes", "roll7_comments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["brand"] = df["brand"].astype(str)
    df = df.sort_values(["report_date_utc", "brand"]).reset_index(drop=True)
    return df


def default_top_brands(panel_T: pd.DataFrame, metric: str, topn: int) -> List[str]:
    if panel_T.empty:
        return []
    totals = panel_T.groupby("brand", as_index=False)[metric].sum().sort_values(metric, ascending=False)
    return totals["brand"].head(topn).tolist()


def to_long(panel_T: pd.DataFrame, brands: List[str], metric: str) -> pd.DataFrame:
    sub = panel_T[panel_T['brand'].isin(brands)].copy()
    sub = sub[["date", "brand", metric]].rename(columns={metric: "value"})
    return sub


def apply_small_rolling(long_df: pd.DataFrame, win: int = 3) -> pd.DataFrame:
    if long_df.empty:
        return long_df.assign(smooth=long_df.get("value", 0))
    out = []
    for b, g in long_df.groupby("brand", as_index=False):
        g = g.sort_values("date").copy()
        g["smooth"] = g["value"].rolling(win, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def previous_asof(all_asofs: List[pd.Timestamp], sel: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the most recent as_of_date_utc strictly before sel, else None."""
    prior = [d for d in all_asofs if d < sel]
    return max(prior) if prior else None

# --- Helper: collapsible description for any element ---
def explain(headline: str, body_md: str, expanded: bool = False) -> None:
    """Render a small expander with detailed, readable documentation.
    headline: short label (e.g., 'Engagement efficiency').
    body_md: long markdown string explaining what/why/how.
    """
    with st.expander(f"ℹ️ {headline}", expanded=expanded):
        st.markdown(body_md)


st.set_page_config(page_title="YouTube Brand Mentions - Rolling Panels", layout="wide")
st.title("YouTube Brand Mentions - Rolling (as-of) Panels")

# Fixed engagement metric for internal use (trajectories etc)
DEFAULT_ENGAGE_METRIC = "views"  # used in trajectories

# ---- Load data
try:
    panel = load_panel()
    roll7 = load_roll7()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---- Sidebar controls
with st.sidebar:
    st.header("Controls")
    asofs = sorted(panel["as_of_date_utc"].dt.date.unique())
    if not asofs:
        st.error("No as_of dates found in panel")
        st.stop()
    sel_asof = st.selectbox("As-of date (UTC)", options=asofs, index=len(asofs) - 1)
    # Default window = 5 days
    window_days = st.number_input("Window length (days)", min_value=2, max_value=7, value=5)

    panel_T = panel[panel["as_of_date_utc"].dt.date == sel_asof].copy()

    if not panel_T.empty:
        max_d = panel_T["date"].max()
        min_d = max_d - pd.Timedelta(days=window_days - 1)
        panel_T = panel_T[panel_T["date"] >= min_d]

    # Previous as_of date (for delta comparisons)
    all_asof_ts = sorted(panel["as_of_date_utc"].dt.date.unique())
    prev_asof_date = previous_asof(all_asof_ts, sel_asof)

    # Show effective window for clarity
    if not panel_T.empty:
        _win_min = panel_T["date"].min().date()
        _win_max = panel_T["date"].max().date()
        st.caption(f"Window: {_win_min} → {_win_max} (UTC)")

    brands_all = sorted(panel_T["brand"].unique().tolist())
    defaults = default_top_brands(panel_T, "views", DEFAULT_TOPN)
    selected_brands = st.multiselect("Brands", options=brands_all, default=defaults)

if panel_T.empty:
    st.info("No rows for this as_of date. Try another T.")
    st.stop()
if not selected_brands:
    st.info("Pick at least one brand to plot.")
    st.stop()

# Prepare common slices used later
today_df = panel_T[(panel_T["date"] == max_d) & (panel_T["brand"].isin(selected_brands))].copy()
past_df = panel_T[(panel_T["date"] < max_d) & (panel_T["brand"].isin(selected_brands))].copy()

# ====================================================
# 1) Engagement Efficiency
# ====================================================
st.subheader("Engagement Efficiency")
st.caption("Average engagement per video in the window: (views + likes + comments) / videos. Highlights brands that generate strong attention with fewer uploads.")
eff = (panel_T.groupby("brand", as_index=False)
       .agg(views=("views","sum"), likes=("likes","sum"), comments=("comments","sum"), mentions=("video_mentions","sum")))
eff["eng_eff"] = (eff["views"] + eff["likes"] + eff["comments"]) / eff["mentions"].replace(0, np.nan)
eff = eff.sort_values("eng_eff", ascending=False)
fig_eff = px.bar(eff[eff["brand"].isin(selected_brands)], x="brand", y="eng_eff",
                 title="Engagement efficiency (engagement per video)")
st.plotly_chart(fig_eff, use_container_width=True)
st.dataframe(eff.round({"eng_eff":2}), use_container_width=True)
explain(
    "Engagement efficiency",
    """
**What it shows**  
Average engagement generated **per video** in the current window: \\(views + likes + comments\\) divided by number of videos collected. Ranks brands by how much attention each upload tends to create.

**How it’s built**  
Window = `[T-(N-1)..T]` from the sidebar. We sum views/likes/comments and divide by total videos for each brand in that window. Numbers are not affected by API page caps because the numerator scales with attention, not with the number of pages fetched.

**Why it matters**  
High efficiency suggests *quality of buzz* and strong creator or content fit. Low efficiency can flag oversupply of low‑impact uploads or weaker audience pull.

**Read with**  
Pair with **Concentration** (is it one mega‑hit or broadly strong?) and **Momentum** (is efficiency rising or fading week‑over‑week?).
    """
)

# ====================================================
# 2) Concentration of Attention (Top/Bottom 5)
# ====================================================
st.subheader("Concentration of Attention")
st.markdown("<small>Top 10% is computed **per brand** by ranking that brand's videos in this window by views and taking the top ceil(10%) of those videos.</small>", unsafe_allow_html=True)
st.caption("Share of total views coming from the top 10% of videos — high values mean attention is concentrated in a few mega-hits (fragile); low values mean buzz is broad and resilient across many videos.")
conc_rows = []
try:
    reg = pd.read_csv(find_file("yt_video_registry.csv"), parse_dates=["published_at_utc"])
    stat = pd.read_csv(find_file("yt_video_stats_daily.csv"), parse_dates=["as_of_date_utc"])
    reg["pub_date"] = reg["published_at_utc"].dt.floor("D")
    cohort = reg[(reg["pub_date"].between(min_d.floor("D"), max_d.floor("D"))) & (reg["brand"].isin(selected_brands))]
    s = stat[stat["as_of_date_utc"].dt.date == sel_asof].copy()
    s = s[s["video_id"].isin(cohort["video_id"])].copy()
    s_agg = (s.groupby("video_id", as_index=False)
               .agg(views=("viewCount","max"), likes=("likeCount","max"), comments=("commentCount","max")))
    vids = cohort.merge(s_agg, on="video_id", how="left")
    for c in ["views","likes","comments"]:
        vids[c] = pd.to_numeric(vids[c], errors="coerce").fillna(0)
    for b, g in vids.groupby("brand"):
        g = g.sort_values("views", ascending=False)
        n = max(1, int(np.ceil(0.10 * len(g))))
        tot = float(g["views"].sum())
        share_top = float(g["views"].head(n).sum()) / tot if tot > 0 else 0.0
        conc_rows.append({"brand": b, "top10_share": share_top})
    conc = pd.DataFrame(conc_rows)
    top5 = conc.sort_values("top10_share", ascending=False).head(5)
    bottom5 = conc.sort_values("top10_share", ascending=True).head(5)

    fig_top = px.bar(top5.sort_values("top10_share", ascending=False), y="brand", x="top10_share", orientation="h",
                     title="Top 5 most concentrated",
                     labels={"top10_share": "Share of views from top 10%"})
    fig_top.update_layout(xaxis_tickformat=",.0%")
    fig_top.update_traces(text=(top5["top10_share"]*100).round(1).astype(str)+"%", textposition="outside")

    fig_bot = px.bar(bottom5.sort_values("top10_share"), y="brand", x="top10_share", orientation="h",
                     title="Bottom 5 least concentrated",
                     labels={"top10_share": "Share of views from top 10%"})
    fig_bot.update_layout(xaxis_tickformat=",.0%")
    fig_bot.update_traces(text=(bottom5["top10_share"]*100).round(1).astype(str)+"%", textposition="outside")

    st.plotly_chart(fig_top, use_container_width=True)
    st.plotly_chart(fig_bot, use_container_width=True)
    explain(
        "Concentration of attention",
        """
**What it shows**  
For each brand, we sort its videos in the window by views and take the top `ceil(10%)`. The bar is the **share of the brand’s total views** contributed by that top slice.

**How it’s built**  
Per‑brand ranking by views → take top decile → compute `sum(views of top decile) / sum(views of all videos)`. The Top 5 chart surfaces hit‑dependent brands; the Bottom 5 highlights broad, resilient attention.

**Why it matters**  
High concentration = fragile, hit‑driven spikes vulnerable to a single upload or creator. Low concentration = diversified buzz across many uploads, typically more durable.

**Notes**  
Use alongside **Freshness** to separate *new‑hit spikes* from *old catalog* concentration.
        """
    )
except FileNotFoundError:
    st.info("Per-video registry/stats not found — cannot build concentration of attention section.")

# ====================================================
# 3) Freshness Index
# ====================================================
st.subheader("Freshness Index")
st.caption("Proportion of content that is recent vs older uploads — highlights which brands actively feed content pipelines.")
try:
    reg = pd.read_csv(find_file("yt_video_registry.csv"), parse_dates=["published_at_utc"])
    stat = pd.read_csv(find_file("yt_video_stats_daily.csv"), parse_dates=["as_of_date_utc"])
    reg["pub_date"] = reg["published_at_utc"].dt.floor("D")
    cohort = reg[(reg["pub_date"].between(min_d.floor("D"), max_d.floor("D"))) & (reg["brand"].isin(selected_brands))]
    s = stat[stat["as_of_date_utc"].dt.date == sel_asof].copy()
    s = s[s["video_id"].isin(cohort["video_id"])].copy()
    s_agg = (s.groupby("video_id", as_index=False)
               .agg(views=("viewCount","max"), likes=("likeCount","max"), comments=("commentCount","max")))
    vids = cohort.merge(s_agg, on="video_id", how="left")
    for c in ["views","likes","comments"]:
        vids[c] = pd.to_numeric(vids[c], errors="coerce").fillna(0)
    age_days = (pd.to_datetime(sel_asof) - vids["published_at_utc"].dt.tz_localize(None)).dt.days.clip(lower=0)
    bins = [-0.1, 1, 3, 7, np.inf]
    labels = ["≤24h", "1–3d", "3–7d", ">7d"]
    vids["age_bin"] = pd.cut(age_days, bins=bins, labels=labels, right=True, include_lowest=True)
    # engagement-weighted (by views) freshness shares
    fresh = (vids.groupby(["brand", "age_bin"], as_index=False)["views"].sum()
                 .rename(columns={"views": "attn"}))
    fresh["share"] = fresh["attn"] / fresh.groupby("brand")["attn"].transform("sum")
    fig_fresh = px.bar(fresh, x="brand", y="share", color="age_bin", barmode="stack", title="Freshness mix (attention‑weighted)")
    fig_fresh.update_layout(yaxis_tickformat=",.0%")
    st.plotly_chart(fig_fresh, use_container_width=True)
    st.dataframe(fresh, use_container_width=True)
    st.caption("Attention‑weighted freshness: share of each brand’s **views** in the window that comes from ≤24h / 1–3d / 3–7d / >7d uploads. Differentiates brands even when raw video counts look similar.")
    explain(
        "Freshness index (attention‑weighted)",
        """
**What it shows**  
How much of a brand’s attention this week comes from **new uploads** vs. older ones. Bars stack to 100% per brand.

**How it’s built**  
We bucket each video by publish‑age (≤24h, 1–3d, 3–7d, >7d) and sum **views** within buckets. Share = bucket views / brand’s total views in the window.

**Why it matters**  
High shares in the ≤24h and 1–3d buckets imply fresh content is actively fueling the story. Heavier >7d share means the brand is living off back‑catalog or a prior hit.

**Caveat**  
Short windows may compress differences; widen the window in the sidebar to reveal separation.
        """
    )
except FileNotFoundError:
    st.info("Per-video registry/stats not found — cannot build freshness index section.")

# ====================================================
# 4) Momentum Quadrant
# ====================================================
if prev_asof_date is not None:
    st.subheader("Momentum Quadrant")
    st.caption("Quadrants: top-right Breakout (high new engagement & strong week-over-week growth); bottom-right Oversupply (many new interactions but weak growth); top-left Echo/Hype (older videos still compounding); bottom-left Dormant (quiet and flat).")
    today_eng = today_df.groupby("brand", as_index=False)[["views","likes","comments"]].sum()
    today_eng["today_total"] = today_eng[["views","likes","comments"]].sum(axis=1)

    # Build local delta vs previous as_of for views over the past days in the window
    prev_T = panel[(panel["as_of_date_utc"].dt.date == prev_asof_date) & (panel["brand"].isin(selected_brands))].copy()
    base_local = past_df.merge(
        prev_T[["date", "brand", "views"]],
        on=["date", "brand"], how="left", suffixes=("_T", "_prev")
    )
    base_local["delta_views"] = base_local["views_T"].fillna(0) - base_local["views_prev"].fillna(0)
    momentum = (
        base_local.groupby("brand", as_index=False)["delta_views"].sum()
        .rename(columns={"delta_views": "momentum_views"})
    )

    scat = today_eng.merge(momentum, on="brand", how="outer").fillna(0)
    fig_sc = px.scatter(
        scat, x="today_total", y="momentum_views", text="brand",
        labels={"today_total": "Today engagement", "momentum_views": "Δ Views (past week)"},
        title=None
    )
    # Quadrant guides at medians
    x_med = float(scat["today_total"].median()) if not scat.empty else 0.0
    y_med = float(scat["momentum_views"].median()) if not scat.empty else 0.0
    try:
        fig_sc.add_vline(x=x_med, line_dash="dash", opacity=0.4)
        fig_sc.add_hline(y=y_med, line_dash="dash", opacity=0.4)
    except Exception:
        pass
    fig_sc.update_traces(textposition="top center")
    st.plotly_chart(fig_sc, use_container_width=True)
    explain(
        "Momentum quadrant",
        """
**What it shows**  
A two‑axis view of *today’s attention* vs. *week‑over‑week momentum*.

- **X:** today’s engagement (views+likes+comments today).  
- **Y:** Δ views over the past days in the window vs. the previous as‑of snapshot.

**Quadrants**  
Top‑right = **Breakout**; bottom‑right = **Oversupply**; top‑left = **Echo/Hype**; bottom‑left = **Dormant**.

**Why it matters**  
Separates brands that are both active **and** compounding from brands posting a lot without traction, and those coasting on past hits.

**Tip**  
Watch brands that cross the median guides from left to right or bottom to top between snapshots.
        """
    )

# ====================================================
# 5) Engagement trajectories (per publish date across recent as-of snapshots)
# ====================================================
st.subheader("Engagement trajectories (per publish date across recent as-of snapshots)")
st.caption("Tracks how engagement for each publish date changes across recent snapshots — reveals compounding vs. fading behavior by publish day.")
max_k = max(2, min(10, len(asofs)))
k_asofs = st.slider("Recent as-of snapshots", min_value=2, max_value=max_k, value=min(5, max_k),
                    help="Tracks how engagement for each publish date changes across recent as-of dates.")
recent_asofs = sorted(asofs)[-k_asofs:]

traj = panel[(panel["as_of_date_utc"].dt.date.isin(recent_asofs))
             & (panel["brand"].isin(selected_brands))
             & (panel["date"].between(min_d, max_d))].copy()
traj["as_of"] = traj["as_of_date_utc"].dt.date
traj["pub_date"] = traj["date"].dt.date

if traj.empty:
    st.info("Not enough snapshots to build trajectories.")
else:
    brand_for_traj = st.selectbox("Brand for trajectories", options=selected_brands)
    sub_traj = traj[traj["brand"] == brand_for_traj]
    fig_traj = px.line(
        sub_traj.sort_values(["pub_date", "as_of"]),
        x="as_of", y=DEFAULT_ENGAGE_METRIC, color="pub_date",
        markers=True,
        labels={"as_of": "As-of date", "pub_date": "Publish date", DEFAULT_ENGAGE_METRIC: METRIC_LABELS.get(DEFAULT_ENGAGE_METRIC, DEFAULT_ENGAGE_METRIC)},
        title=None
    )
    fig_traj.update_traces(line=dict(width=3))
    fig_traj.update_layout(template="plotly_white", hovermode="x unified", legend_title_text="Publish date")
    st.plotly_chart(fig_traj, use_container_width=True)
    explain(
        "Engagement trajectories",
        """
**What it shows**  
For a chosen brand, each line tracks a **publish date’s** cumulative engagement across recent as‑of snapshots. You can see which cohorts are compounding (lines rising) vs. fading.

**How it’s built**  
Filter to the selected window and recent as‑of dates; plot engagement for each publish day across snapshots.

**Why it matters**  
Reveals whether recent uploads are building momentum over time or peaking quickly.
        """
    )

# ====================================================
# 6) Changes vs previous as-of (past days) — tables
# ====================================================
st.subheader("Changes vs previous as-of (past days)")
st.caption(f"Δ = value at as_of **{sel_asof}** minus value at previous as_of **{prev_asof_date}** for the same publish dates and brands.")
if prev_asof_date is None:
    st.info("No previous as-of date available for comparison.")
else:
    prev_T = panel[(panel["as_of_date_utc"].dt.date == prev_asof_date) & (panel["brand"].isin(selected_brands))].copy()
    # Align on (date, brand) within the current window but excluding today
    base = past_df.merge(
        prev_T[["date", "brand", "views", "likes", "comments"]],
        on=["date", "brand"], how="left", suffixes=("_T", "_prev")
    )
    base["as_of_T"] = pd.to_datetime(sel_asof)
    base["as_of_prev"] = pd.to_datetime(prev_asof_date)
    for m in ["views", "likes", "comments"]:
        base[f"delta_{m}"] = base[f"{m}_T"].fillna(0) - base[f"{m}_prev"].fillna(0)

    # Detail table (per publish_date × brand × as-of pair)
    detail = base[["date", "brand", "as_of_prev", "as_of_T", "delta_views", "delta_likes", "delta_comments"]].copy()
    detail.rename(columns={"date": "publish_date"}, inplace=True)
    detail["publish_date"] = detail["publish_date"].dt.date
    detail["as_of_prev"] = pd.to_datetime(detail["as_of_prev"]).dt.date
    detail["as_of_T"] = pd.to_datetime(detail["as_of_T"]).dt.date
    for _c in ["delta_views", "delta_likes", "delta_comments"]:
        detail[_c] = detail[_c].map(lambda v: f"{v:+,}")
    st.markdown("**Per-publish-date changes**")
    st.caption("Each row: publish_date × brand. Deltas compare as_of_prev → as_of_T to capture genuine growth/decay not explained by new uploads.")
    st.dataframe(
        detail.sort_values(["publish_date", "brand"]).reset_index(drop=True),
        use_container_width=True,
    )
    explain(
        "Per‑publish‑date changes",
        """
**What it shows**  
Exact deltas by publish date between the previous as‑of and today’s as‑of for the same rows. Positive numbers mean incremental engagement accrued since the last snapshot.

**How it’s built**  
For each (brand, publish_date) in the window excluding today, compute `current − previous` for views/likes/comments.

**Why it matters**  
Removes the confounder of new uploads and isolates *organic accrual* on the existing cohort.
        """
    )

    # Summary by brand (sum of deltas over past days)
    date_span = "—"
    if not past_df.empty:
        date_span = f"{past_df['date'].min().date()} → {past_df['date'].max().date()}"
    delta_sum = (
        base.groupby("brand", as_index=False)[["delta_views", "delta_likes", "delta_comments"]]
            .sum()
            .sort_values("delta_views", ascending=False)
    )
    for _c in ["delta_views", "delta_likes", "delta_comments"]:
        delta_sum[_c] = delta_sum[_c].map(lambda v: f"{v:+,}")
    delta_sum.insert(1, "date_span", date_span)
    delta_sum.insert(2, "as_of_compare", f"{prev_asof_date} → {sel_asof}")
    st.markdown("**Brand summary of changes (sum over past days)**  ")
    st.caption(f"Publish dates included: {date_span}; as_of compare: {prev_asof_date} → {sel_asof}")
    st.dataframe(delta_sum, use_container_width=True)
    explain(
        "Brand summary of changes",
        """
**What it shows**  
Sum of deltas over past days per brand. A compact leaderboard of who is gaining or losing momentum in this window.

**How it’s built**  
Aggregate the per‑publish‑date deltas by brand.

**Why it matters**  
Useful for portfolio‑style monitoring and alerts.
        """
    )


# ====================================================
# 8) Where did growth come from? (stacked Δ Views by publish date)
# ====================================================
if prev_asof_date is not None and not past_df.empty:
    stack = base[["date", "brand", "delta_views"]].copy()
    stack["date"] = stack["date"].dt.date
    st.subheader("Where did growth come from? — Δ Views by day (stacked)")
    fig_stack = px.bar(
        stack, x="brand", y="delta_views", color="date",
        labels={"delta_views": "Δ Views", "brand": "Brand", "date": "Publish date"},
        barmode="stack"
    )
    st.plotly_chart(fig_stack, use_container_width=True)
    comp = (base.groupby("brand", as_index=False)["delta_views"].sum()
                 .sort_values("delta_views", ascending=False))
    comp["delta_views"] = comp["delta_views"].map(lambda v: f"{v:+,}")
    st.dataframe(comp.rename(columns={"delta_views": "Total Δ Views (past days)"}), use_container_width=True)
    st.caption("Breaks total Δ Views (vs previous as‑of) by publish date for each brand to pinpoint **when** growth occurred within the window.")
    explain(
        "Where did growth come from?",
        """
**What it shows**  
Breaks total Δ views by **publish date** within the window for each brand to reveal **when** the growth occurred (e.g., yesterday’s spike vs. an older video compounding).

**How it’s built**  
Use the same current−previous deltas by publish date; stack by date per brand.

**Why it matters**  
Great for root‑cause analysis after a big move in the Momentum Quadrant or Watchlist.
        """
    )

# ====================================================
# Analyst Modules (no raw SoV anywhere)
# ====================================================
with st.expander("Analyst modules", expanded=True):
    st.caption("Deeper diagnostics built for analysts: quality (BQI), efficiency (EPI), velocity & half‑life, creator breadth/concentration, new vs returning creators, watchlist flags, and top movers. All metrics avoid raw SoV counts.")
    tabs = st.tabs([
        "BQI (Buzz Quality Index)", "Engagement Premium", "Velocity & Half-life", "Exposure breadth & concentration",
        "New vs returning creators", "Watchlist flags", "Surprise & Top Movers"
    ])

    # 0) BQI only (no SoV)
    with tabs[0]:
        st.markdown("### Buzz Quality Index (BQI) — composite of quality (engagement-weighted attention), reach (channel breadth), and diversification (inverse HHI)")
        # Engagement-weighted attention proxy using window sums (per brand)
        w = (
            panel_T.groupby("brand", as_index=False)
                   .agg(views=("views","sum"), likes=("likes","sum"), comments=("comments","sum"), mentions=("video_mentions","sum"))
        )
        if w.empty:
            st.info("No data to compute BQI for this window.")
        else:
            for c in ["views","likes","comments","mentions"]:
                w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0)
            v_safe = w["views"].replace(0, np.nan)
            like_rate = (w["likes"] / v_safe).fillna(0.0)
            comment_rate = (w["comments"] / v_safe).fillna(0.0)
            w["ew_attn"] = np.log1p(w["views"]) * (1.0 + like_rate + comment_rate)

            # Channel breadth & HHI over mentions within the window
            try:
                reg = pd.read_csv(find_file("yt_video_registry.csv"), parse_dates=["published_at_utc"])  # includes channel_id
            except FileNotFoundError:
                st.info("Registry not found — BQI will use engagement-only components.")
                w["breadth"] = np.nan
                w["inv_hhi"] = np.nan
            else:
                reg["pub_date"] = reg["published_at_utc"].dt.floor("D")
                r = reg[(reg["pub_date"].between(min_d.floor("D"), max_d.floor("D")))]
                if r.empty:
                    w["breadth"] = np.nan
                    w["inv_hhi"] = np.nan
                else:
                    breadth = r.groupby("brand", as_index=False)["channel_id"].nunique().rename(columns={"channel_id":"breadth"})
                    ch_counts = r.groupby(["brand","channel_id"], as_index=False)["video_id"].nunique().rename(columns={"video_id":"videos"})
                    tot = ch_counts.groupby("brand", as_index=False)["videos"].sum().rename(columns={"videos":"tot"})
                    m = ch_counts.merge(tot, on="brand", how="left")
                    m["share"] = np.where(m["tot"]>0, m["videos"]/m["tot"], 0.0)
                    hhi = m.groupby("brand", as_index=False)["share"].apply(lambda s: (s**2).sum()).rename(columns={"share":"hhi"})
                    tmp = breadth.merge(hhi, on="brand", how="outer")
                    w = w.merge(tmp, on="brand", how="left")
                    w["breadth"] = pd.to_numeric(w["breadth"], errors="coerce")
                    w["inv_hhi"] = 1.0 / w["hhi"].replace(0, np.nan)

            # Z-score helper
            def zs(x: pd.Series) -> pd.Series:
                mu = x.mean()
                sd = x.std(ddof=0)
                if pd.isna(mu) or (sd is None) or (sd == 0):
                    return pd.Series([0.0]*len(x), index=x.index)
                return (x - mu) / sd

            w["z_ew"] = zs(w["ew_attn"])      # attention quality
            w["z_breadth"] = zs(w["breadth"]) # reach across creators
            w["z_invhhi"] = zs(w["inv_hhi"])  # diversification of reach
            w["BQI"] = w[["z_ew","z_breadth","z_invhhi"]].sum(axis=1)

            show = w[["brand","ew_attn","breadth","hhi","BQI"]].copy()
            show = show.sort_values("BQI", ascending=False)
            if selected_brands:
                show = show[show["brand"].isin(selected_brands)]

            fig_bqi = px.bar(
                show, x="brand", y="BQI", title="Buzz Quality Index (z-sum: attention × breadth × diversification)"
            )
            st.plotly_chart(fig_bqi, use_container_width=True)
            st.dataframe(show.round({"ew_attn":2, "breadth":0, "hhi":3, "BQI":2}), use_container_width=True)
            explain(
                "Buzz Quality Index (BQI)",
                """
**What it shows**  
Composite score: engagement‑weighted attention (quality) + creator breadth (reach) + diversification (inverse HHI). Higher means attention that is broad and higher quality.

**How it’s built**  
`ew_attn = log(views) × (1 + like_rate + comment_rate)`. Breadth = unique channels in window. HHI from shares of videos across channels; we use its inverse for diversification. Each component is z‑scored and summed.

**Why it matters**  
Balances spike size and structure; helps avoid being fooled by one‑hit wonders.
                """
            )

    # 1) Engagement Premium (quality of buzz in current window)
    with tabs[1]:
        w = (panel_T.groupby("brand", as_index=False)
             .agg(views=("views","sum"), likes=("likes","sum"), comments=("comments","sum"), mentions=("video_mentions","sum")))
        if w.empty:
            st.info("No data for the current window.")
        else:
            for c in ["views","likes","comments","mentions"]:
                w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0)
            w["epm"] = (w["views"] + w["likes"] + w["comments"]) / w["mentions"].replace(0, np.nan)
            baseline = w["epm"].median()
            w["epi"] = w["epm"] / (baseline if baseline and not np.isnan(baseline) else 1)
            w = w.sort_values("epi", ascending=False)
            fig_epi = px.bar(w[w["brand"].isin(selected_brands)], x="brand", y="epi",
                              title="Engagement Premium Index (vs median in this window)")
            fig_epi.update_layout(yaxis_title="Index (1.0 = median)")
            st.plotly_chart(fig_epi, use_container_width=True)
            st.dataframe(w.round({"epm":2, "epi":2}), use_container_width=True)
            st.caption("EPI = Engagement per mention / median across brands in the current [T-6..T] window.")
            explain(
                "Engagement Premium (EPI)",
                """
**What it shows**  
Efficiency index: engagement per mention vs the median in this window.

**How it’s built**  
`EPI = (views+likes+comments)/videos ÷ median across brands`.

**Why it matters**  
Highlights brands whose uploads consistently punch above their weight.
                """
            )

    # 2) Velocity & Half-life (uses registry + per-video daily stats)
    with tabs[2]:
        try:
            reg = pd.read_csv(find_file("yt_video_registry.csv"), parse_dates=["published_at_utc"])  # video_id, brand, channel_id, published_at_utc, ...
            stat = pd.read_csv(find_file("yt_video_stats_daily.csv"), parse_dates=["as_of_date_utc"])  # video_id, as_of_date_utc, viewCount, likeCount, commentCount
        except FileNotFoundError:
            st.info("Per-video registry/stats not found.")
        else:
            reg["pub_date"] = reg["published_at_utc"].dt.floor("D")
            cohort = reg[(reg["pub_date"].between(min_d.floor("D"), max_d.floor("D"))) & (reg["brand"].isin(selected_brands))][["video_id","brand","pub_date","published_at_utc"]]
            if cohort.empty:
                st.info("No videos in the current window for the selected brands.")
            else:
                stat = stat.merge(cohort, on="video_id", how="inner")
                stat["age_days"] = (stat["as_of_date_utc"].dt.floor("D") - stat["published_at_utc"].dt.floor("D")).dt.days
                for c in ["viewCount","likeCount","commentCount"]:
                    stat[c] = pd.to_numeric(stat[c], errors="coerce").fillna(0)
                # Velocity: median views by age 0..7 per brand
                vel = (stat.groupby(["brand","age_days"], as_index=False)["viewCount"].median()
                            .query("age_days >= 0 & age_days <= 7"))
                if vel.empty:
                    st.info("Not enough as-of snapshots to compute velocity.")
                else:
                    fig_vel = px.line(vel, x="age_days", y="viewCount", color="brand", markers=True,
                                      title="Median views by age (0–7 days since publish)")
                    fig_vel.update_layout(xaxis_title="Days since publish", yaxis_title="Median views")
                    st.plotly_chart(fig_vel, use_container_width=True)
                # Half-life per video, then median by brand
                vv = (stat.groupby(["brand","video_id","age_days"], as_index=False)["viewCount"].sum()
                        .sort_values(["brand","video_id","age_days"]))
                out_rows = []
                for (b, vid), g in vv.groupby(["brand","video_id"]):
                    if g.empty: continue
                    total = g["viewCount"].sum()
                    if total <= 0: continue
                    g = g.reset_index(drop=True)
                    g["cum"] = g["viewCount"].cumsum()
                    g["cum_share"] = g["cum"] / total
                    hit = g[g["cum_share"] >= 0.5]
                    hl = hit["age_days"].iloc[0] if not hit.empty else np.nan
                    out_rows.append({"brand": b, "video_id": vid, "half_life_days": hl})
                if out_rows:
                    hl_df = pd.DataFrame(out_rows)
                    brand_hl = hl_df.groupby("brand", as_index=False)["half_life_days"].median().sort_values("half_life_days")
                    st.dataframe(brand_hl, use_container_width=True)
                    st.caption("Half-life = median days for a brand’s cohort to reach 50% of cumulative views.")
                    explain(
                        "Velocity & Half‑life",
                        """
**Velocity**  
Median daily views by age (0–7 days) approximates how quickly attention arrives.

**Half‑life**  
Median days for videos to reach 50% of cumulative views. Short half‑life = front‑loaded spikes; long half‑life = slower burn.
                        """
                    )
                else:
                    st.info("Could not compute half-life for this window.")

    # 3) Exposure breadth & concentration (from registry in window)
    with tabs[3]:
        try:
            reg = pd.read_csv(find_file("yt_video_registry.csv"), parse_dates=["published_at_utc"])  # includes channel_id
        except FileNotFoundError:
            st.info("Registry not found.")
        else:
            reg["pub_date"] = reg["published_at_utc"].dt.floor("D")
            r = reg[(reg["pub_date"].between(min_d.floor("D"), max_d.floor("D"))) & (reg["brand"].isin(selected_brands))]
            if r.empty:
                st.info("No registry rows for the current window.")
            else:
                breadth = r.groupby("brand", as_index=False)["channel_id"].nunique().rename(columns={"channel_id":"unique_channels"})
                ch_counts = r.groupby(["brand","channel_id"], as_index=False)["video_id"].nunique().rename(columns={"video_id":"videos"})
                tot = ch_counts.groupby("brand", as_index=False)["videos"].sum().rename(columns={"videos":"tot"})
                m = ch_counts.merge(tot, on="brand", how="left")
                m["share"] = np.where(m["tot"]>0, m["videos"]/m["tot"], 0.0)
                hhi = m.groupby("brand", as_index=False)["share"].apply(lambda s: (s**2).sum()).rename(columns={"share":"hhi_by_mentions"})
                out = breadth.merge(hhi, on="brand", how="left").sort_values(["unique_channels","hhi_by_mentions"], ascending=[False, True])
                st.dataframe(out, use_container_width=True)
                st.caption("Breadth = unique channels in [T-6..T]. HHI (by mentions): higher = more dependent on few creators.")
                explain(
                    "Exposure breadth & concentration",
                    """
**Breadth**  
Number of unique channels mentioning the brand in the window (reach).

**Concentration (HHI)**  
HHI computed from each channel’s share of mentions; higher = more concentrated and creator‑dependent.
                    """
                )

    # 4) New vs returning creators (vs prior N days)
    with tabs[4]:
        LOOKBACK_DAYS = 60
        try:
            reg = pd.read_csv(find_file("yt_video_registry.csv"), parse_dates=["published_at_utc"])  # brand, channel_id
        except FileNotFoundError:
            st.info("Registry not found.")
        else:
            reg["pub_date"] = reg["published_at_utc"].dt.floor("D")
            cur = reg[(reg["pub_date"].between(min_d.floor("D"), max_d.floor("D"))) & (reg["brand"].isin(selected_brands))]
            prev = reg[(reg["pub_date"] < min_d.floor("D")) & (reg["pub_date"] >= min_d.floor("D") - pd.Timedelta(days=LOOKBACK_DAYS))]
            seen_before = set(zip(prev["brand"], prev["channel_id"]))
            pairs = cur[["brand","channel_id"]].drop_duplicates().copy()
            pairs["new_creator"] = ~pairs.apply(lambda r: (r["brand"], r["channel_id"]) in seen_before, axis=1)
            rate = (pairs.groupby("brand")["new_creator"].mean()
                          .rename("share_new_creators").reset_index()
                          .sort_values("share_new_creators", ascending=False))
            st.dataframe(rate, use_container_width=True)
            st.caption(f"Share of channels in [T-6..T] that did not mention the brand in the prior {LOOKBACK_DAYS} days.")
            explain(
                "New vs returning creators",
                """
**What it shows**  
Share of channels in `[T-(N-1)..T]` that are new relative to the prior lookback period. High share implies expanding creator base.
                """
            )

    # 5) Watchlist flags (breakouts & streaks on roll-7)
    with tabs[5]:
        r7_T = roll7[roll7["report_date_utc"].dt.date == sel_asof].copy()
        if r7_T.empty:
            st.info("No roll-7 rows for this as_of date.")
        else:
            sc = r7_T[["brand","roll7_views"]].copy()
            sc["roll7_views"] = pd.to_numeric(sc["roll7_views"], errors="coerce").fillna(0)
            mu, sd = sc["roll7_views"].mean(), sc["roll7_views"].std()
            sd = sd if sd and not np.isnan(sd) else 1.0
            sc["z"] = (sc["roll7_views"] - mu) / sd
            breakout = sc[(sc["z"] >= 2) & (sc["brand"].isin(selected_brands))].sort_values("z", ascending=False)

            rsub = roll7[roll7["brand"].isin(selected_brands)].sort_values(["brand","report_date_utc"]).copy()
            rsub["chg"] = rsub.groupby("brand")["roll7_views"].diff()
            # rolling 3-day count of positive changes per brand; align index back to rsub
            pos = (rsub["chg"] > 0)
            streak = (
                pos.groupby(rsub["brand"])  # group by brand preserving original index order
                   .apply(lambda s: s.rolling(3, min_periods=3).sum())
                   .reset_index(level=0, drop=True)
            )
            rsub["streak_up"] = streak.fillna(0)
            sustained = rsub[(rsub["report_date_utc"].dt.date == sel_asof) & (rsub["streak_up"] >= 3)]

            st.markdown("**Breakout (z≥2 on roll-7 views):**")
            st.dataframe(breakout[["brand","roll7_views","z"]].round(2), use_container_width=True)
            st.markdown("**Sustained growth (≥3 consecutive increases in roll-7 views):**")
            st.dataframe(sustained[["brand","report_date_utc","roll7_views"]], use_container_width=True)
            st.caption("Flags surface unusual attention regimes to investigate (events, campaigns, product drops). Not investment advice.")
            explain(
                "Watchlist flags",
                """
**Breakout**  
Roll‑7 views ≥ 2σ above mean (z≥2).

**Sustained growth**  
At least 3 consecutive increases in roll‑7 views.

Use to queue qualitative review (releases, promos, UGC waves).
                """
            )

    # 6) Surprise & Top Movers — delta vs previous as_of
    with tabs[6]:
        if prev_asof_date is None:
            st.info("No previous as_of date to compute movers.")
        else:
            cur = panel[(panel["as_of_date_utc"].dt.date == sel_asof)].copy()
            if cur.empty:
                st.info("No rows for current as_of.")
            else:
                cur_max = cur["date"].max()
                cur_min = cur_max - pd.Timedelta(days=window_days - 1)
                curw = cur[cur["date"].between(cur_min, cur_max)]
                # Current window aggregates
                cur_w = (curw.groupby("brand", as_index=False)
                             .agg(views=("views","sum"), likes=("likes","sum"), comments=("comments","sum"), mentions=("video_mentions","sum")))
                v_safe = cur_w["views"].replace(0, np.nan)
                cur_w["ew_attn"] = np.log1p(cur_w["views"]) * (1.0 + (cur_w["likes"]/v_safe).fillna(0.0) + (cur_w["comments"]/v_safe).fillna(0.0))
                # Previous window aggregates
                prev = panel[(panel["as_of_date_utc"].dt.date == prev_asof_date)].copy()
                if prev.empty:
                    st.info("No rows for previous as_of.")
                else:
                    prev_max = prev["date"].max()
                    prev_min = prev_max - pd.Timedelta(days=window_days - 1)
                    prevw = prev[prev["date"].between(prev_min, prev_max)]
                    prev_w = (prevw.groupby("brand", as_index=False)
                                  .agg(views=("views","sum"), likes=("likes","sum"), comments=("comments","sum"), mentions=("video_mentions","sum")))
                    v_safe_p = prev_w["views"].replace(0, np.nan)
                    prev_w["ew_attn"] = np.log1p(prev_w["views"]) * (1.0 + (prev_w["likes"]/v_safe_p).fillna(0.0) + (prev_w["comments"]/v_safe_p).fillna(0.0))
                    # Join and compute deltas (current - previous)
                    mov = cur_w.merge(prev_w[["brand","ew_attn","mentions","views","likes","comments"]], on="brand", how="left", suffixes=("_cur","_prev"))
                    for c in ["ew_attn","mentions","views","likes","comments"]:
                        mov[f"d_{c}"] = mov[f"{c}_cur"].fillna(0) - mov[f"{c}_prev"].fillna(0)
                    movers = mov.sort_values("d_ew_attn", ascending=False)
                    if selected_brands:
                        movers = movers[movers["brand"].isin(selected_brands)]
                    st.markdown("**Top Movers by Δ Engagement-weighted attention (current vs previous as-of)**")
                    fig_mov = px.bar(movers.head(15), x="brand", y="d_ew_attn",
                                     labels={"d_ew_attn": "Δ EW attention", "brand": "Brand"})
                    st.plotly_chart(fig_mov, use_container_width=True)
                    show_cols = ["brand","d_ew_attn","d_mentions","d_views","d_likes","d_comments"]
                    st.dataframe(movers[show_cols].round(2), use_container_width=True)
                    st.caption("Compares current vs previous as‑of over the same window. Positive Δ EW attention indicates improving **quality** of attention, not just more uploads.")
                    explain(
                        "Surprise & Top Movers",
                        """
**What it shows**  
Delta in engagement‑weighted attention vs previous as‑of, ranked.

**Why it matters**  
Surfaces regime shifts in quality of attention, not just upload volume.
                        """
                    )

st.caption("Data: yt_brand_daily_panel.csv (Rolling-1) and yt_brand_roll7_daily.csv (Rolling-2). All timestamps UTC.")