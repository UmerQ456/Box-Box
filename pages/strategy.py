import matplotlib.pyplot as plt
import streamlit as st

from app import DATA_PATH, get_analysis_tables, get_cleaned_laps, load_ver_raw_laps
from box_box import plots

st.set_page_config(page_title="Strategy | Box-Box", page_icon="🏁", layout="wide")
st.title("Pit Strategy & Stint Pace Comparison")

st.markdown(
    "Compares stint-level pace (`median_time` / `mean_time`) across stints and tyre compounds "
    "(`box_box.plots.plot_strategy_comparison`) to highlight how each stint contributed to "
    "Verstappen's overall race strategy."
)

raw_df = load_ver_raw_laps(str(DATA_PATH))
clean_df, _ = get_cleaned_laps(raw_df)
_, stints_df = get_analysis_tables(clean_df)

compounds = (
    sorted(stints_df["compound"].dropna().astype(str).unique().tolist())
    if not stints_df.empty and "compound" in stints_df.columns
    else []
)
selected = st.sidebar.multiselect("Tyre Compounds", options=compounds, default=compounds)

df = stints_df.copy()
if selected and not df.empty and "compound" in df.columns:
    df = df[df["compound"].astype(str).isin(selected)]

chart_col, _ = st.columns([0.8, 0.2])
with chart_col:
    fig = plots.plot_strategy_comparison(df, title="VER — Stint Pace & Strategy Comparison")
    fig.set_size_inches(8.5, 3.8)
    fig.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)

if not df.empty:
    st.dataframe(df, width="stretch")