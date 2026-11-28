import matplotlib.pyplot as plt
import streamlit as st

from app import DATA_PATH, get_analysis_tables, get_cleaned_laps, load_ver_raw_laps
from box_box import plots

st.set_page_config(page_title="Stints | Box-Box", page_icon="📉", layout="wide")
st.title("Stint Tyre Degradation")

st.markdown(
    "Plots lap time against tyre age (`life`) per stint (`box_box.plots.plot_stint_degradation`). "
    "Shows linear degradation slopes (`s/lap`—negative values indicate net pace gains from fuel burn "
    "and track evolution; positive values indicate net wear), with a toggle for a conjugate "
    "Normal-Inverse-Gamma (NIG) credible band."
)

raw_df = load_ver_raw_laps(str(DATA_PATH))
clean_df, _ = get_cleaned_laps(raw_df)
deg_df, _ = get_analysis_tables(clean_df)

stints = (
    sorted(clean_df["stint"].dropna().unique().tolist())
    if "stint" in clean_df.columns and not clean_df.empty
    else []
)
selected_stints = st.sidebar.multiselect("Stints", options=stints, default=stints)
use_bayesian = st.sidebar.toggle("Bayesian Credible Band (NIG)", value=False)
credible_level = st.sidebar.slider(
    "Credible Interval Level",
    min_value=0.80,
    max_value=0.99,
    value=0.90,
    step=0.05,
    disabled=not use_bayesian,
)

df = clean_df.copy()
sub_deg = deg_df.copy()
if selected_stints and "stint" in df.columns:
    df = df[df["stint"].isin(selected_stints)]
if selected_stints and not sub_deg.empty and "stint" in sub_deg.columns:
    sub_deg = sub_deg[sub_deg["stint"].isin(selected_stints)]

chart_col, _ = st.columns([0.8, 0.2])
with chart_col:
    fig = plots.plot_stint_degradation(
        df,
        degradation_df=sub_deg,
        use_bayesian_band=use_bayesian,
        credible_level=credible_level,
        title="VER — Stint Tyre Degradation",
    )
    fig.set_size_inches(8.5, 3.9)
    fig.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)

if not sub_deg.empty:
    st.dataframe(sub_deg, width="stretch")