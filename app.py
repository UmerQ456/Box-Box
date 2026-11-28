"""
Overview landing page for Box-Box: 2026 Australian GP (Race - VER).

Reads the local VER race dataset once per session via @st.cache_data, exposes
shared data/analysis helpers for pages/, and displays top-level overview metrics
and project/methodology descriptions for the oral exam.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from box_box import analysis, cleaning, loader

DATA_PATH = (
    Path("data")
    / "2026"
    / "Australian Grand Prix"
    / "Race"
    / "VER"
    / "laptimes.json"
)


@st.cache_data(show_spinner="Loading local VER race laps...")
def load_ver_raw_laps(path_str: str) -> pd.DataFrame:
    """Load local lap times JSON without network requests or disk writes."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found at: {path.resolve()}")

    for fn_name in ("load_laptimes", "load_laps", "load_json_laps", "load_race_data"):
        fn = getattr(loader, fn_name, None)
        if callable(fn):
            try:
                res = fn(path)
                if isinstance(res, pd.DataFrame):
                    return res
            except TypeError:
                pass

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ("laps", "laptimes", "data", "results"):
            if isinstance(payload.get(key), list):
                return pd.DataFrame(payload[key])
    return pd.DataFrame(payload)


@st.cache_data(show_spinner="Cleaning green-flag laps...")
def get_cleaned_laps(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run clean_laps and return (clean_df, summary_info)."""
    result = cleaning.clean_laps(raw_df)
    if isinstance(result, tuple) and len(result) == 2:
        clean_df, info = result
        return clean_df, info if isinstance(info, dict) else {"info": info}
    return result, {}


@st.cache_data(show_spinner="Computing stint degradation & pace stats...")
def get_analysis_tables(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute stint degradation slopes and stint/pace summaries via box_box.analysis."""
    deg_df = pd.DataFrame()
    for fn_name in ("compute_stint_degradation", "stint_degradation", "calculate_degradation"):
        fn = getattr(analysis, fn_name, None)
        if callable(fn):
            deg_df = fn(clean_df)
            break

    stints_df = pd.DataFrame()
    for fn_name in ("summarize_stints", "compute_stint_summary", "pace_summary", "summarize_pace"):
        fn = getattr(analysis, fn_name, None)
        if callable(fn):
            out = fn(clean_df)
            if isinstance(out, pd.DataFrame):
                stints_df = out
            break

    if stints_df.empty and not clean_df.empty and {"stint", "time"}.issubset(clean_df.columns):
        group_cols = ["stint"] + (["compound"] if "compound" in clean_df.columns else [])
        stints_df = (
            clean_df.groupby(group_cols, as_index=False)["time"]
            .agg(median_time="median", mean_time="mean", n_laps="count")
        )

    return deg_df, stints_df


def render_fig(fig: plt.Figure) -> None:
    """Render a matplotlib Figure in Streamlit and close it to free memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def main() -> None:
    st.set_page_config(
        page_title="Box-Box | 2026 Australian GP (VER)",
        page_icon="🏎️",
        layout="wide",
    )

    st.title("Box-Box: Max Verstappen Race Telemetry & Strategy — 2026 Australian Grand Prix")
    st.markdown(
    "<p style='font-size: 1.2rem; line-height: 1.45; margin-top: -0.4rem;'>"
    "Driver: <strong>VER</strong> (Max Verstappen) · Session: <strong>Race</strong> · "
    "Deterministic local offline dataset (<code>laptimes.json</code> cached in memory via <code>@st.cache_data</code>)"
    "</p>",
    unsafe_allow_html=True,
    )

    st.info(
        "**What you're looking at:** An offline, deterministic Formula 1 race analytics dashboard for "
        "**Max Verstappen (`VER`)** at the **2026 Australian Grand Prix (Race)**, built as a "
        "multi-page Streamlit interface over the modular `src/box_box/` package."
    )

    try:
        raw_df = load_ver_raw_laps(str(DATA_PATH))
        clean_df, clean_meta = get_cleaned_laps(raw_df)
        deg_df, stints_df = get_analysis_tables(clean_df)
    except Exception as exc:
        st.error(f"Failed to load or process local race data: {exc}")
        st.stop()

    st.sidebar.header("Dataset Info")
    st.sidebar.text(f"Source: {DATA_PATH}")
    st.sidebar.markdown(
        "**Offline & Deterministic Mode**\n\n"
        "- Reads local JSON once per session with `@st.cache_data`\n"
        "- No network calls or extra disk writes during runs\n"
        "- Navigate to **pace**, **stints**, or **strategy** in the sidebar"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Laps", len(raw_df))
    col2.metric("Clean Green Laps", len(clean_df))
    median_pace = (
        f"{clean_df['time'].median():.3f} s"
        if not clean_df.empty and "time" in clean_df.columns
        else "N/A"
    )
    col3.metric("Median Green Pace", median_pace)
    avg_slope = (
        f"{deg_df['slope_sec_per_lap'].mean():+.4f} s/lap"
        if not deg_df.empty and "slope_sec_per_lap" in deg_df.columns
        else "N/A"
    )
    col4.metric("Mean Deg. Slope", avg_slope)

    st.divider()

    st.subheader("What I Did, How It Works & What It Accomplished")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("#### What I Built")
        st.write(
            "Separated pure data, statistical, and plotting logic into `src/box_box/` "
            "(`loader.py`, `cleaning.py`, `analysis.py`, `plots.py`) and paired it with a "
            "multi-page Streamlit UI (`app.py` + `pages/pace.py`, `pages/stints.py`, `pages/strategy.py`)."
        )
    with m2:
        st.markdown("#### How I Analyzed It")
        st.write(
            "Loaded the local 2026 Australian GP `VER` race `laptimes.json` via `pathlib` and "
            "`@st.cache_data`, filtered green-flag laps in `cleaning.py`, fitted stint degradation "
            "slopes (`scipy.stats.linregress`) plus a closed-form conjugate Normal-Inverse-Gamma "
            "(NIG) credible band in `plots.py`, and aggregated stint pace summaries."
        )
    with m3:
        st.markdown("#### What It Accomplished")
        st.write(
            "Turned raw race lap JSON into reproducible, clean visuals and tables that "
            "reveal compound lap-time progression, stint-by-stint tyre wear vs. fuel/track evolution, "
            "and pit-strategy pace trade-offs for a clear walkthrough."
        )

    st.divider()

    st.subheader("Dedicated Subpages")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. Pace (`pages/pace.py`)")
        st.write(
            "Tracks lap-by-lap green-flag lap times across the race colored by tyre compound, "
            "helping spot compound windows, stint transitions, and fastest-lap phases."
        )
    with c2:
        st.markdown("### 2. Stints (`pages/stints.py`)")
        st.write(
            "Models lap time against tyre age (`life`) per stint using linear regression slopes "
            "(s/lap) with an optional conjugate Normal-Inverse-Gamma credible band."
        )
    with c3:
        st.markdown("### 3. Strategy (`pages/strategy.py`)")
        st.write(
            "Compares stints and tyre compounds by summary pace (median/mean lap time and lap count) "
            "to evaluate how Verstappen's stint structure performed across the race."
        )

    with st.expander("Cleaning Metadata & Summary Tables"):
        if not deg_df.empty:
            st.markdown("**Stint Degradation Slopes**")
            st.markdown(
                "Per-stint ordinary least-squares fit of lap time (`time`) against tyre age (`life`).\n"
                "- `slope_sec_per_lap`: Rate of lap-time change per lap of tyre age (`s/lap`); "
                "negative values indicate net pace gains (fuel burn / track evolution), while positive values indicate net tyre wear.\n"
                "- `intercept_sec`: Extrapolated baseline lap time at tyre age `0` (`s`).\n"
                "- `r_squared`: Goodness of fit (`0` to `1`) showing how closely a straight line tracks the stint's lap times.\n"
                "- `p_value`: Hypothesis-test p-value indicating whether the slope differs meaningfully from zero rather than random scatter.\n"
                "- `n_laps`: Number of cleaned green-flag laps used in that stint's regression."
            )
            st.dataframe(deg_df, width="stretch")

        if not stints_df.empty:
            st.markdown("**Stint Pace Summary**")
            st.markdown(
                "Aggregated stint performance grouped by stint and tyre compound.\n"
                "- `median_time`: Typical lap time (`s`) in the stint, resistant to minor lap-to-lap spikes.\n"
                "- `mean_time`: Average lap time (`s`) across the stint's clean laps.\n"
                "- `n_laps`: Stint length in cleaned green-flag laps—helpful for distinguishing short high-pace stints from long endurance stints."
            )
            st.dataframe(stints_df, width="stretch")

        if clean_meta:
            st.markdown("**Cleaning Metadata**")
            st.markdown(
                "Audit trail from `cleaning.clean_laps()` showing how the raw `58`-lap race dataset for `VER` "
                "was filtered down to `48` clean green-flag laps:\n"
                "- `initial` (`58`): Total raw laps in `laptimes.json`.\n"
                "- `after_green_flag` (`51`): Laps remaining after removing non-green-flag laps.\n"
                "- `after_pit_filter` (`49`): Laps remaining after removing pit-in/pit-out laps.\n"
                "- `after_lap1_filter` (`48`): Laps remaining after excluding Lap 1.\n"
                "- `after_valid_time` (`48`): Laps remaining after validating lap time values.\n"
                "- `final_clean_laps` (`48`): Final analysis-ready green-flag laps used across all plots and tables."
            )
            st.json(clean_meta)


if __name__ == "__main__":
    main()