from pathlib import Path
import sys
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TICKERS_TRY = ("228810.KS", "228810.KQ")
OUTPUT_PATH = "../data/tiger_media_contents_prices.csv"
DROP_ZERO_VOLUME = False

# --- Plotting settings ---
PLOT_OUTPUT_PATH = "../data/tiger_media_contents_prices.png"  # where to save the plot image
SHOW_PLOT = False  # set True to display the window; False for headless save-only


def _to_tidy(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize a Yahoo Finance DataFrame to a tidy schema:
    columns = [date, close, volume, ticker]
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume", "ticker"])

    # Reset index to expose Date
    if not isinstance(df.index, pd.DatetimeIndex):
        # yfinance should return a DatetimeIndex, but handle fallback
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Date_"}).set_index("Date_")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.reset_index()

    # Column names can vary in casing; map robustly
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", None)
    close_col = cols.get("close", None)
    vol_col = cols.get("volume", None)

    missing = [name for name, col in [("Date", date_col), ("Close", close_col), ("Volume", vol_col)] if col is None]
    if missing:
        # Try to guess by suffix
        for c in df.columns:
            cl = c.lower()
            if date_col is None and cl.startswith("date"):
                date_col = c
            if close_col is None and cl.endswith("close") and "adj" not in cl:
                close_col = c
            if vol_col is None and cl.endswith("volume"):
                vol_col = c

    if any(x is None for x in (date_col, close_col, vol_col)):
        raise ValueError(f"Could not identify required columns in Yahoo frame. Found: {list(df.columns)}")

    out = df[[date_col, close_col, vol_col]].copy()
    out.columns = ["date", "close", "volume"]

    # Coerce types
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    if DROP_ZERO_VOLUME:
        out = out[(out["volume"].notna()) & (out["volume"] != 0)]

    out["ticker"] = ticker
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_full_history(tickers_try=TICKERS_TRY) -> pd.DataFrame:
    """
    Try tickers in order using Ticker.history(period='max'), which avoids MultiIndex.
    """
    last_err = None
    for t in tickers_try:
        try:
            df = yf.Ticker(t).history(period="max", auto_adjust=False)
            tidy = _to_tidy(df, t)
            if not tidy.empty:
                return tidy
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Failed to fetch any data from Yahoo Finance for the provided tickers.")


def main():
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_full_history(TICKERS_TRY)
    if df.empty:
        sys.stderr.write("No data returned; nothing to write.\n")
        sys.exit(2)

    # De-duplicate in case both tickers return overlapping rows
    df = df.drop_duplicates(subset=["date"]).sort_values("date")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(df):,} rows to {out_path.resolve()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Columns: {list(df.columns)}")

    # --- Plot daily Close and Volume time series ---
    df_plot = df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])  # ensure datetime for plotting

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Price line
    ax_price.plot(df_plot["date"], df_plot["close"], linewidth=1.2)
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{df_plot['ticker'].iloc[0]} – Daily Close & Volume")
    ax_price.grid(True, alpha=0.3)

    # Volume bars
    ax_vol.bar(df_plot["date"], df_plot["volume"], width=1.0)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, axis="y", alpha=0.3)

    # Nicely formatted dates
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_vol.xaxis.set_major_locator(locator)
    ax_vol.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    out_img = Path(PLOT_OUTPUT_PATH)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=150)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved plot image to {out_img.resolve()}")


if __name__ == "__main__":
    main()
