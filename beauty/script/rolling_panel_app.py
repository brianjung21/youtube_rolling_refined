"""
Explore Rolling-1 / Rolling-2 outputs from kpop_rolling_gathering.py
- Reads:
    - data/yt_brand_daily_panel.csv (Rolling-1 daily, 7-day window stamped as_of_date_utc)
    - data/yt_brand_roll7_daily.csv (Rolling-2 rollup, one row/brand per report_date_utc)
    - data/yt_video_registry.csv (optional drilldown - not used in first pass)
    - data/yt_video_stats_daily.csv (optional drilldown - not used in first pass)
- Let's you pick an as_of_date_utc (T), a metric, and brands to plot
- Shows: 7-day daily line chart for [T-6..T], and roll-7 bar table for the same T.
- Optional expander: compare roll-7 across available T's for the selected brands
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
        if c  in df.columns:
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


st.set_page_config(page_title="YouTube Brand Mentions - Rolling Panels", layout="wide")
st.title("YouTube Brand Mentions - Rolling (as-of) Panels")


try:
    panel = load_panel()
    roll7 = load_roll7()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


with st.sidebar:
    st.header("Controls")
    asofs = sorted(panel["as_of_date_utc"].dt.date.unique())
    if not asofs:
        st.error("No as_of dates found in panel")
        st.stop()
    sel_asof = st.selectbox("As-of date (UTC)", options=asofs, index=len(asofs) - 1)

    metric_key = st.radio(
        "Metric",
        list(METRIC_LABELS.keys()),
        index=0,
        format_func=lambda k: METRIC_LABELS[k]
    )
    engage_metric = st.radio(
        "Engagement metric",
        ["views", "likes", "comments"],
        index=0,
        format_func=lambda k: METRIC_LABELS[k]
    )
    window_days = st.number_input("Window length (days)", min_value=2, max_value=7, value=3)

    panel_T = panel[panel["as_of_date_utc"].dt.date == sel_asof].copy()

    if not panel_T.empty:
        max_d = panel_T["date"].max()
        min_d = max_d - pd.Timedelta(days=window_days - 1)
        panel_T = panel_T[panel_T["date"] >= min_d]

    # Previous as_of date (for delta comparisons)
    asof_ts = pd.to_datetime(sel_asof)
    all_asof_ts = sorted(panel["as_of_date_utc"].dt.date.unique())
    prev_asof_date = previous_asof(all_asof_ts, sel_asof)

    # Show effective window for clarity
    if not panel_T.empty:
        _win_min = panel_T["date"].min().date()
        _win_max = panel_T["date"].max().date()
        st.caption(f"Window: {_win_min} → {_win_max} (UTC)")

    brands_all = sorted(panel_T["brand"].unique().tolist())
    defaults = default_top_brands(panel_T, metric_key, DEFAULT_TOPN)
    selected_brands = st.multiselect("Brands", options=brands_all, default=defaults)

    do_smooth = st.checkbox("Show 3-day smoothing", value=True)
    trim_zeros = st.checkbox("Trim leading all-zero days (current T window)", value=True)

if panel_T.empty:
    st.info("No rows for this as_of date. Try another T.")
    st.stop()
if not selected_brands:
    st.info("Pick at least one brand to plot.")
    st.stop()

# ---------- A) Today (T) spotlight: discovery & engagement ----------
max_d = panel_T["date"].max()
min_d = panel_T["date"].min()

st.subheader(f"Today (T={sel_asof}) — new mentions & engagement")

today_df = panel_T[(panel_T["date"] == max_d) & (panel_T["brand"].isin(selected_brands))].copy()
if today_df.empty:
    st.info("No rows for the latest date in this window.")
else:
    # (A1) New video mentions today — bar chart
    fig_today_mentions = px.bar(
        today_df.sort_values("video_mentions", ascending=False),
        x="brand", y="video_mentions",
        title=None, labels={"video_mentions": "New videos (mentions)", "brand": "Brand"}
    )
    st.plotly_chart(fig_today_mentions, use_container_width=True)

    # (A2) Engagement snapshot today — table with formatted numbers
    cols_show = ["brand", "video_mentions", "views", "likes", "comments"]
    _tbl_today = today_df[cols_show].sort_values("views", ascending=False).copy()
    for _c in ["video_mentions", "views", "likes", "comments"]:
        _tbl_today[_c] = _tbl_today[_c].map(lambda v: f"{int(v):,}")
    st.dataframe(_tbl_today, use_container_width=True)

    # Build past-days slice (dates strictly before the latest date in window)
    past_df = panel_T[(panel_T["date"] < max_d) & (panel_T["brand"].isin(selected_brands))].copy()

# ---------- B) Engagement trajectories across as-of snapshots ----------
st.subheader("Engagement trajectories (per publish date across recent as-of snapshots)")
# Choose how many recent as-of snapshots to track
max_k = max(2, min(10, len(asofs)))
k_asofs = st.slider("Recent as-of snapshots", min_value=2, max_value=max_k, value=min(5, max_k), help="Tracks how engagement for each publish date changes across recent as-of dates.")
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
        x="as_of", y=engage_metric, color="pub_date",
        markers=True,
        labels={"as_of": "As-of date", "pub_date": "Publish date", engage_metric: METRIC_LABELS[engage_metric]},
        title=None
    )
    fig_traj.update_traces(line=dict(width=3))
    fig_traj.update_layout(template="plotly_white", hovermode="x unified", legend_title_text="Publish date")
    st.plotly_chart(fig_traj, use_container_width=True)

# ---------- C) Changes vs previous as-of (past days) ----------
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
    # Detail table (per publish_date × brand × as-of pair) with signed/commified deltas and explicit as-ofs
    detail = base[["date", "brand", "as_of_prev", "as_of_T", "delta_views", "delta_likes", "delta_comments"]].copy()
    detail.rename(columns={"date": "publish_date"}, inplace=True)
    detail["publish_date"] = detail["publish_date"].dt.date
    detail["as_of_prev"] = pd.to_datetime(detail["as_of_prev"]).dt.date
    detail["as_of_T"] = pd.to_datetime(detail["as_of_T"]).dt.date
    for _c in ["delta_views", "delta_likes", "delta_comments"]:
        detail[_c] = detail[_c].map(lambda v: f"{v:+,}")
    st.markdown("**Per-publish-date changes**")
    st.caption("Each row: publish_date × brand. Deltas compare as_of_prev → as_of_T.")
    st.dataframe(
        detail.sort_values(["publish_date", "brand"]).reset_index(drop=True),
        use_container_width=True,
    )

    # Summary by brand (sum of deltas over past days) with signed/commified values + date span and as-of compare
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

# ---------- D) Where did engagement grow? Heatmap of Δ Views (past days) ----------
show_hm = st.checkbox("Show Δ Views heatmap (past days)", value=False)
if show_hm and prev_asof_date is not None and not past_df.empty:
    hm_base = base.copy()
    hm_base["pub_date"] = hm_base["date"].dt.date
    hm_pivot = (hm_base[["brand", "pub_date", "delta_views"]]
                .pivot(index="brand", columns="pub_date", values="delta_views")
                .fillna(0))
    if not hm_pivot.empty:
        # sort rows and columns by total Δ views to surface hotspots
        row_tot = hm_pivot.sum(axis=1).sort_values(ascending=False)
        col_tot = hm_pivot.sum(axis=0).sort_values(ascending=False)
        hm_pivot = hm_pivot.loc[row_tot.index, col_tot.index]
        st.subheader("Where did engagement grow? — Δ Views heatmap (past days vs prev as-of)")
        fig_hm = px.imshow(
            hm_pivot.values,
            x=[c for c in hm_pivot.columns], y=hm_pivot.index,
            labels=dict(x="Publish date", y="Brand", color="Δ Views"),
            aspect="auto", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

# ---------- E) Today’s attention vs momentum (Δ views on past days) ----------
if prev_asof_date is not None:
    today_mentions = (today_df[["brand", "video_mentions"]]
                      .rename(columns={"video_mentions": "today_mentions"}))
    momentum = (base.groupby("brand", as_index=False)["delta_views"]
                    .sum().rename(columns={"delta_views": "momentum_views"}))
    scat = (today_mentions.merge(momentum, on="brand", how="outer").fillna(0))
    st.subheader("Today’s attention vs momentum")
    fig_sc = px.scatter(
        scat, x="today_mentions", y="momentum_views", text="brand",
        labels={"today_mentions": "New videos today", "momentum_views": "Δ Views (past days)"},
        title=None
    )
    fig_sc.update_traces(textposition="top center")
    st.plotly_chart(fig_sc, use_container_width=True)

# ---------- F) Composition of growth: stacked Δ Views by publish date ----------
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
    # Companion summary: total Δ views per brand over past days
    comp = (base.groupby("brand", as_index=False)["delta_views"].sum()
                 .sort_values("delta_views", ascending=False))
    comp["delta_views"] = comp["delta_views"].map(lambda v: f"{v:+,}")
    st.dataframe(comp.rename(columns={"delta_views": "Total Δ Views (past days)"}), use_container_width=True)

# ---------- Roll-N: on-the-fly rollup for the current window ----------
st.subheader(f"{window_days}-day rollup (one row/brand) — as_of = {sel_asof}")
rollN_tbl = (
    panel_T[panel_T["brand"].isin(selected_brands)]
        .groupby("brand", as_index=False)[metric_key]
        .sum()
        .sort_values(metric_key, ascending=False)
)
fig_barN = px.bar(rollN_tbl, x="brand", y=metric_key, title=None)
st.plotly_chart(fig_barN, use_container_width=True)
st.dataframe(rollN_tbl.rename(columns={metric_key: METRIC_LABELS[metric_key]}), use_container_width=True)

# ---------- Rolling-2: 7-day rollup for the same T ----------
if window_days == 7:
    st.subheader(f"7-day rollup (one row/brand) — report_date_utc = {sel_asof}")
    roll7_col = {
        "video_mentions": "roll7_video_mentions",
        "views": "roll7_views",
        "likes": "roll7_likes",
        "comments": "roll7_comments",
    }[metric_key]

    roll7_T = roll7[roll7["report_date_utc"].dt.date == sel_asof].copy()
    if roll7_T.empty:
        st.info("No roll-7 rows for this as_of date.")
    else:
        if selected_brands:
            roll7_T = roll7_T[roll7_T["brand"].isin(selected_brands)]
        top_tbl = (
            roll7_T[["brand", roll7_col]]
            .rename(columns={roll7_col: METRIC_LABELS[metric_key]})
            .sort_values(METRIC_LABELS[metric_key], ascending=False)
        )
        fig_bar = px.bar(top_tbl, x="brand", y=METRIC_LABELS[metric_key], title=None)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(top_tbl, use_container_width=True)

# ---------- Optional: compare roll-7 across available T’s ----------
if window_days == 7:
    with st.expander("Compare roll-7 across available as_of dates"):
        rsub = roll7[roll7["brand"].isin(selected_brands)].copy()
        rsub = rsub.sort_values(["brand", "report_date_utc"])
        rsub = rsub.rename(columns={
            "report_date_utc": "as_of_date_utc",
            "roll7_video_mentions": "video_mentions",
            "roll7_views": "views",
            "roll7_likes": "likes",
            "roll7_comments": "comments",
        })
        if not rsub.empty:
            fig_r7 = px.line(
                rsub,
                x="as_of_date_utc",
                y=metric_key,
                color="brand",
                labels={"as_of_date_utc": "As-of date (UTC)", metric_key: METRIC_LABELS[metric_key]},
                title=f"Roll-7 {METRIC_LABELS[metric_key]} across as-of dates",
            )
            st.plotly_chart(fig_r7, use_container_width=True)
            st.dataframe(
                rsub.pivot_table(index="as_of_date_utc", columns="brand", values=metric_key, aggfunc="first"),
                use_container_width=True,
            )

st.caption("Data: yt_brand_daily_panel.csv (Rolling-1) and yt_brand_roll7_daily.csv (Rolling-2). All timestamps UTC.")
