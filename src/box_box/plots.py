"""
This code implements src/box_box/plots.py using matplotlib, seaborn, numpy,
pandas, and scipy to generate green-flag pace progressions, stint tyre-degradation
plots (with optional light Bayesian credible bands), and strategy comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

COMPOUND_COLORS: dict[str, str] = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD400",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067FF",
}


def _resolve_compound_palette(compounds: list[str]) -> dict[str, str]:
    """Return a clean color mapping for tyre compounds."""
    fallback = sns.color_palette("Set2", max(len(compounds), 1))
    palette: dict[str, str] = {}
    for idx, comp in enumerate(compounds):
        key = str(comp).upper()
        palette[comp] = COMPOUND_COLORS.get(key, f"#{int(fallback[idx % len(fallback)][0] * 255):02x}{int(fallback[idx % len(fallback)][1] * 255):02x}{int(fallback[idx % len(fallback)][2] * 255):02x}")
    return palette


def _bayesian_credible_band(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    credible_level: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a light conjugate Normal-Inverse-Gamma posterior mean and credible band.

    Uses a weakly informative prior centered at zero slope and the sample mean of y,
    then updates analytically in closed form without heavy MCMC sampling.
    """
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    p = X.shape[1]

    # Weakly informative conjugate NIG prior: beta_0 = [mean(y), 0], V_0 = 1e4 * I_p
    beta_0 = np.array([float(np.mean(y)), 0.0])
    prec_0 = np.eye(p) * 1e-4
    a_0 = 2.0
    b_0 = 2.0 * max(float(np.var(y, ddof=1)) if n > 1 else 1.0, 1e-3)

    prec_n = prec_0 + X.T @ X
    cov_unscaled = np.linalg.inv(prec_n)
    beta_n = cov_unscaled @ (prec_0 @ beta_0 + X.T @ y)

    residuals = y - X @ beta_n
    quad_prior = (beta_n - beta_0).T @ prec_0 @ (beta_n - beta_0)
    a_n = a_0 + n / 2.0
    b_n = b_0 + 0.5 * (float(residuals.T @ residuals) + float(quad_prior))
    df_post = 2.0 * a_n
    scale_post_sq = b_n / a_n

    X_grid = np.column_stack([np.ones_like(x_grid), x_grid])
    mean_pred = X_grid @ beta_n
    # Posterior predictive uncertainty for a new lap at each grid point
    pred_var = scale_post_sq * (1.0 + np.sum((X_grid @ cov_unscaled) * X_grid, axis=1))
    t_crit = float(stats.t.ppf(0.5 + credible_level / 2.0, df=df_post))
    half_width = t_crit * np.sqrt(np.maximum(pred_var, 0.0))

    return mean_pred, mean_pred - half_width, mean_pred + half_width


def plot_pace_progression(
    clean_laps_df: pd.DataFrame,
    title: str = "Green-Flag Lap Time Progression by Tyre Compound",
) -> plt.Figure:
    """Plot lap times across race laps colored by tyre compound."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if clean_laps_df.empty or not {"lap", "time"}.issubset(clean_laps_df.columns):
        ax.set_title(title)
        ax.text(0.5, 0.5, "No clean lap data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    df = clean_laps_df.dropna(subset=["lap", "time"]).sort_values("lap").copy()
    has_compound = "compound" in df.columns and df["compound"].notna().any()
    has_driver = "drv" in df.columns and df["drv"].nunique(dropna=True) > 1

    if has_compound:
        compounds = df["compound"].astype(str).unique().tolist()
        palette = _resolve_compound_palette(compounds)
        sns.scatterplot(
            data=df,
            x="lap",
            y="time",
            hue="compound",
            style="drv" if has_driver else None,
            palette=palette,
            s=55,
            edgecolor="#222222",
            linewidth=0.5,
            ax=ax,
        )
    else:
        sns.scatterplot(data=df, x="lap", y="time", s=55, ax=ax)

    ax.plot(df["lap"], df["time"], color="#888888", alpha=0.35, linewidth=1.0, zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Race Lap")
    ax.set_ylabel("Lap Time (s)")
    if ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    return fig


def plot_stint_degradation(
    clean_laps_df: pd.DataFrame,
    degradation_df: pd.DataFrame | None = None,
    use_bayesian_band: bool = False,
    credible_level: float = 0.90,
    title: str = "Stint Tyre Degradation (Lap Time vs. Tyre Age)",
) -> plt.Figure:
    """Plot lap time against tyre age per stint with linear regression (or optional Bayesian band)."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))

    required = {"stint", "life", "time"}
    if clean_laps_df.empty or not required.issubset(clean_laps_df.columns):
        ax.set_title(title)
        ax.text(0.5, 0.5, "No stint degradation data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    df = clean_laps_df.dropna(subset=["stint", "life", "time"]).sort_values(["stint", "life"]).copy()
    stints = sorted(df["stint"].unique())
    colors = sns.color_palette("tab10", max(len(stints), 1))

    for idx, stint_val in enumerate(stints):
        grp = df[df["stint"] == stint_val].sort_values("life")
        x = grp["life"].to_numpy(dtype=float)
        y = grp["time"].to_numpy(dtype=float)
        color = colors[idx % len(colors)]

        compound_label = ""
        if "compound" in grp.columns and grp["compound"].notna().any():
            compound_label = f" ({grp['compound'].iloc[0]})"

        ax.scatter(x, y, color=color, alpha=0.75, s=45, label=f"Stint {int(stint_val)}{compound_label}")

        if len(x) >= 2 and np.ptp(x) > 0:
            x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 50)
            if use_bayesian_band:
                y_fit, y_low, y_high = _bayesian_credible_band(x, y, x_grid, credible_level=credible_level)
                ax.plot(x_grid, y_fit, color=color, linestyle="--", linewidth=1.8)
                ax.fill_between(
                    x_grid,
                    y_low,
                    y_high,
                    color=color,
                    alpha=0.16,
                    label=f"Stint {int(stint_val)} {int(credible_level * 100)}% Credible Band",
                )
            else:
                slope = None
                intercept = None
                if degradation_df is not None and not degradation_df.empty and "stint" in degradation_df.columns:
                    match = degradation_df[degradation_df["stint"] == stint_val]
                    if not match.empty and {"slope_sec_per_lap", "intercept_sec"}.issubset(match.columns):
                        slope = float(match["slope_sec_per_lap"].iloc[0])
                        intercept = float(match["intercept_sec"].iloc[0])

                if slope is None or intercept is None:
                    reg = stats.linregress(x, y)
                    slope = float(reg.slope)
                    intercept = float(reg.intercept)

                ax.plot(
                    x_grid,
                    intercept + slope * x_grid,
                    color=color,
                    linewidth=1.8,
                    label=f"Stint {int(stint_val)} Slope: {slope:+.4f} s/lap",
                )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tyre Age / Life (laps)")
    ax.set_ylabel("Lap Time (s)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_strategy_comparison(
    stints_df: pd.DataFrame,
    title: str = "Pit Strategy & Stint Pace Comparison",
) -> plt.Figure:
    """Plot stint lengths and pace across stints/compounds to compare pit strategies."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if stints_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No stint strategy data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    y_col = next((c for c in ["median_time", "mean_time", "avg_time"] if c in stints_df.columns), None)
    x_col = "stint" if "stint" in stints_df.columns else stints_df.columns[0]

    if y_col is None:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Missing stint pace column", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    df = stints_df.dropna(subset=[y_col]).copy()
    has_compound = "compound" in df.columns and df["compound"].notna().any()
    has_driver = "drv" in df.columns and df["drv"].nunique(dropna=True) > 1

    if has_compound:
        compounds = df["compound"].astype(str).unique().tolist()
        palette = _resolve_compound_palette(compounds)
        sns.barplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="compound",
            palette=palette,
            edgecolor="#222222",
            linewidth=0.6,
            ax=ax,
        )
    else:
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)

    if has_driver:
        sns.stripplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="drv",
            dodge=True,
            color="#111111",
            size=5,
            ax=ax,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Stint")
    ax.set_ylabel(f"{y_col.replace('_', ' ').title()} (s)")
    if ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    return fig