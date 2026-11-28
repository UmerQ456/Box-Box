import matplotlib.pyplot as plt
import streamlit as st

from app import DATA_PATH, get_cleaned_laps, load_ver_raw_laps
from box_box import plots

st.set_page_config(page_title="Pace | Box-Box", page_icon="⏱️", layout="wide")
st.title("Green-Flag Lap Pace Progression")

st.markdown(
    "Tracks cleaned green-flag lap times across race laps colored by tyre compound "
    "(`box_box.plots.plot_pace_progression`). Use this view to trace compound windows, "
    "stint transitions, and pace relative to the race median."
)

raw_df = load_ver_raw_laps(str(DATA_PATH))
clean_df, _ = get_cleaned_laps(raw_df)

compounds = (
    sorted(clean_df["compound"].dropna().astype(str).unique().tolist())
    if "compound" in clean_df.columns and not clean_df.empty
    else []
)
selected = st.sidebar.multiselect("Tyre Compounds", options=compounds, default=compounds)

df = clean_df.copy()
if selected and "compound" in df.columns:
    df = df[df["compound"].astype(str).isin(selected)]

chart_col, _ = st.columns([0.8, 0.2])
with chart_col:
    fig = plots.plot_pace_progression(df, title="VER — 2026 Australian GP Lap Time Progression")
    fig.set_size_inches(8.5, 3.8)
    fig.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)