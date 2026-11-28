```markdown
# 🏎️ Box, Box — 2026 Australian Grand Prix F1 Analytics

An offline, deterministic Formula 1 race analytics pipeline and multi-page Streamlit dashboard focused on **Max Verstappen (`VER`)** at the **2026 Australian Grand Prix (Race)**. Built with `uv`, Python, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, and `streamlit`.

![Streamlit app](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

## What the App Does

Rather than stopping at final classification standings, **Box, Box** inspects lap-by-lap race dynamics from a local dataset snapshot:

| View / Page | What You Can Do |
|-------------|-----------------|
| **Overview** (`app.py`) | Inspect race KPIs (raw laps, clean green laps, median green pace, mean degradation slope), project/methodology notes, and expandable cleaning audit & summary tables. |
| **Pace** (`pages/pace.py`) | Visualize cleaned green-flag lap time progression across race laps colored by tyre compound, with interactive compound filtering. |
| **Stints** (`pages/stints.py`) | Analyze lap time versus tyre age (`life`) per stint with linear regression slopes, stint filtering, and a toggle for a conjugate Normal-Inverse-Gamma (NIG) Bayesian credible band. |
| **Strategy** (`pages/strategy.py`) | Compare stint-level pace (`median_time`, `mean_time`) and lap counts across stints and tyre compounds, with interactive compound filtering. |

### How the Metrics Work

- **Lap cleaning (`src/box_box/cleaning.py`):** Filters raw race laps down to green-flag analysis laps (`58` raw laps → `48` clean laps for `VER`) while recording each filtering step (`initial`, `after_green_flag`, `after_pit_filter`, `after_lap1_filter`, `after_valid_time`, `final_clean_laps`).
- **Linear stint degradation (`src/box_box/analysis.py`):** Fits ordinary least-squares regression (`scipy.stats.linregress`) of lap time (`time`) against tyre age (`life`) per stint segment:
  - `slope_sec_per_lap`: Seconds per lap change with tyre age (negative values indicate net pace gains from fuel burn/track evolution; positive values indicate net wear).
  - `r_squared` & `p_value`: Goodness of fit (`0` to `1`) and slope significance test.
- **Bayesian credible band (`src/box_box/plots.py`):** Optional closed-form conjugate Normal-Inverse-Gamma (NIG) posterior predictive mean and credible interval on stint degradation plots without heavy MCMC sampling.

## Data Sources

- **TracingInsights 2026 Snapshot (`data/2026/Australian Grand Prix/Race/VER/laptimes.json`):** Local JSON lap-timing dataset for Max Verstappen (`VER`) at the 2026 Australian Grand Prix Race, cached in memory via `@st.cache_data` for fast, deterministic, offline runs.

## Libraries

| Library | Used For |
|---------|----------|
| `pandas` | Tabular lap loading, cleaning, grouping, and stint summaries |
| `numpy` | Numerical arrays, linear algebra for NIG posterior updates, and summary math |
| `scipy` | Linear regression (`scipy.stats.linregress`) and Student-t quantiles for credible bands |
| `matplotlib` | Core figure rendering for lap progression, stint degradation, and strategy charts |
| `seaborn` | Styled scatter/line/bar plots and compound/stint color palettes |
| `streamlit` | Multi-page web dashboard, sidebar filters, tables, and session caching |
| `uv` | Fast environment and dependency management (`pyproject.toml` / `uv.lock`) |

## Setup

1. **Install `uv`** (if not already installed):
   - macOS / Linux:
     ```bash
     curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
     ```
   - Windows (PowerShell):
     ```powershell
     irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex
     ```

2. **Clone this repository and install dependencies:**
   ```bash
   git clone <your-repo-url> Box-Box
   cd Box-Box
   uv sync

```

## Data Setup (TracingInsights Sparse Checkout)

`data/2026/` is kept as a separate local Git repository (and ignored in `.gitignore`) so its nested `.git` history stays isolated from `Box-Box`.

To reproduce the local snapshot in `data/2026/Australian Grand Prix`, run:

```bash
mkdir -p data/2026 && cd data/2026
git clone --filter=blob:none --no-checkout [https://github.com/TracingInsights/2026.git](https://github.com/TracingInsights/2026.git) .
git sparse-checkout set "Australian Grand Prix"
git checkout

```

*(Note: Ensure the file ends up at `data/2026/Australian Grand Prix/Race/VER/laptimes.json` as expected by `app.py`.)*

## Run & Smoke Tests

Start the multi-page Streamlit dashboard (with hot reload enabled):

```bash
uv run streamlit run app.py

```

The app opens in your browser; use the sidebar to move between the overview and the `pace`, `stints`, and `strategy` pages.

You can also run the backend modules directly via `uv run` smoke tests on the `VER` race dataset:

* **Lap cleaning check:**
```bash
uv run python -c "import json, pandas as pd; from box_box.cleaner import clean_laps; raw = pd.DataFrame(json.load(open('data/2026/Australian Grand Prix/Race/VER/laptimes.json'))); cleaned, stats = clean_laps(raw); print(stats); print(cleaned[['lap', 'time', 'compound', 'stint', 'life', 'pos']].head(10))"

```


* **Stint aggregation check:**
```bash
uv run python -c "import json, pandas as pd; from box_box.cleaning import clean_laps, build_stints_tyres; raw = pd.DataFrame(json.load(open('data/2026/Australian Grand Prix/Race/VER/laptimes.json'))); cleaned, _ = clean_laps(raw); print(build_stints_tyres(cleaned))"

```


* **Stint degradation & driver pace summary check:**
```bash
uv run python -c "import json, pandas as pd; from box_box.cleaning import clean_laps; from box_box.analysis import compute_stint_degradation, summarize_driver_pace; raw = pd.DataFrame(json.load(open('data/2026/Australian Grand Prix/Race/VER/laptimes.json'))); cleaned, _ = clean_laps(raw); print(compute_stint_degradation(cleaned)); print(summarize_driver_pace(cleaned))"

```


* **End-to-end plot generation smoke test (cross-platform temp directory):**
```bash
uv run python -c "import json, tempfile; from pathlib import Path; import pandas as pd; from box_box.cleaning import clean_laps; from box_box.analysis import compute_stint_degradation; from box_box.plots import plot_pace_progression, plot_stint_degradation, plot_strategy_comparison; raw = pd.DataFrame(json.load((Path('data') / '2026' / 'Australian Grand Prix' / 'Race' / 'VER' / 'laptimes.json').open())); cleaned, _ = clean_laps(raw); deg = compute_stint_degradation(cleaned); stints = cleaned.groupby(['stint', 'compound'], as_index=False)['time'].agg(median_time='median', mean_time='mean', n_laps='count'); figs = [plot_pace_progression(cleaned), plot_stint_degradation(cleaned, deg), plot_stint_degradation(cleaned, deg, use_bayesian_band=True), plot_strategy_comparison(stints)]; out = Path(tempfile.gettempdir()); [f.savefig(out / f'test_plot_{i}.png') for i, f in enumerate(figs)]; print('Saved', len(figs), 'plots to', out)"

```



## Project Structure

```text
Box-Box/
├── app.py                  # Streamlit entry point, cached pipeline & overview KPIs
├── pages/
│   ├── pace.py             # Green-flag lap pace progression page
│   ├── stints.py           # Stint tyre degradation & NIG credible band page
│   └── strategy.py         # Stint pace & compound strategy page
├── src/
│   └── box_box/            # Backend package (no Streamlit code here)
│       ├── __init__.py
│       ├── loader.py       # Local race data loading helpers
│       ├── cleaning.py     # Green-flag filtering & cleaning audit trail
│       ├── analysis.py     # Linear regression degradation & stint summaries
│       └── plots.py        # Matplotlib/Seaborn & NIG credible-band plots
├── data/
│   └── 2026/               # Git-ignored sparse checkout of TracingInsights/2026
├── pyproject.toml
├── uv.lock
└── README.md

```

Everything in `src/box_box/` is plain Python reusable across scripts or notebooks, while `app.py` and `pages/` focus on displaying results and interactive controls.

## Notes on Reproducibility & Scope

* **Offline & Deterministic:** Reads local JSON files directly without network requests during analysis or rendering, using `@st.cache_data` so reruns stay fast and repeatable.
* **Focused Scope:** Intentionally scoped to single-driver (`VER`) lap-level pace, stint degradation, and stint strategy on the 2026 Australian GP race snapshot so filtering rules and statistical outputs remain clear and explainable.
