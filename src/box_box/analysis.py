"""
This code implements src/box_box/analysis.py using numpy, pandas, and scipy.stats.linregress to 
calculate linear tyre-degradation slopes per stint and summarize green-flag race pace across drivers. 
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_stint_degradation(
    clean_laps_df: pd.DataFrame,
    min_laps: int = 4,
) -> pd.DataFrame:
    """Estimate linear tyre degradation (seconds per lap of tyre age) per stint.

    Parameters
    ----------
    clean_laps_df : pd.DataFrame
        Cleaned lap-level DataFrame output by ``clean_laps``.
    min_laps : int, default 4
        Minimum number of clean laps in a stint segment required to fit a regression.

    Returns
    -------
    pd.DataFrame
        One row per stint segment with regression slope (sec/lap), intercept,
        R-squared, p-value, and sample size.
    """
    required = {"stint", "compound", "life", "time"}
    missing = required - set(clean_laps_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for degradation analysis: {sorted(missing)}")

    if clean_laps_df.empty:
        return pd.DataFrame(
            columns=[
                "drv",
                "team",
                "stint",
                "compound",
                "n_laps",
                "slope_sec_per_lap",
                "intercept_sec",
                "r_squared",
                "p_value",
            ]
        )

    df = clean_laps_df.dropna(subset=["life", "time"]).sort_values(["stint", "lap"]).copy()
    segment_id = (
        (df["stint"] != df["stint"].shift())
        | (df["compound"].astype(str) != df["compound"].astype(str).shift())
    ).cumsum()

    rows: list[dict[str, object]] = []
    for _, grp in df.groupby(segment_id, sort=False):
        if len(grp) < min_laps:
            continue

        x = grp["life"].to_numpy(dtype=float)
        y = grp["time"].to_numpy(dtype=float)

        # Guard against zero variance in tyre age
        if np.ptp(x) == 0:
            continue

        reg = stats.linregress(x, y)
        first_row = grp.iloc[0]
        rows.append(
            {
                "drv": first_row["drv"] if "drv" in grp.columns else pd.NA,
                "team": first_row["team"] if "team" in grp.columns else pd.NA,
                "stint": int(first_row["stint"]),
                "compound": str(first_row["compound"]),
                "n_laps": int(len(grp)),
                "slope_sec_per_lap": round(float(reg.slope), 4),
                "intercept_sec": round(float(reg.intercept), 3),
                "r_squared": round(float(reg.rvalue ** 2), 4),
                "p_value": round(float(reg.pvalue), 4),
            }
        )

    return pd.DataFrame(rows)


def summarize_driver_pace(clean_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize green-flag pace per driver using numpy/pandas metrics.

    Parameters
    ----------
    clean_laps_df : pd.DataFrame
        Cleaned lap-level DataFrame output by ``clean_laps``.

    Returns
    -------
    pd.DataFrame
        Driver pace summary sorted from fastest median green-flag lap time to slowest.
    """
    required = {"time"}
    missing = required - set(clean_laps_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for driver pace summary: {sorted(missing)}")

    if clean_laps_df.empty:
        return pd.DataFrame(
            columns=["drv", "team", "clean_laps", "mean_time", "median_time", "std_time", "best_time"]
        )

    group_cols = [col for col in ["drv", "team"] if col in clean_laps_df.columns]
    if not group_cols:
        times = clean_laps_df["time"].to_numpy(dtype=float)
        return pd.DataFrame(
            [
                {
                    "clean_laps": int(len(times)),
                    "mean_time": round(float(np.mean(times)), 3),
                    "median_time": round(float(np.median(times)), 3),
                    "std_time": round(float(np.std(times, ddof=1)) if len(times) > 1 else 0.0, 3),
                    "best_time": round(float(np.min(times)), 3),
                }
            ]
        )

    rows: list[dict[str, object]] = []
    for keys, grp in clean_laps_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        times = grp["time"].to_numpy(dtype=float)
        row: dict[str, object] = dict(zip(group_cols, keys))
        row.update(
            {
                "clean_laps": int(len(times)),
                "mean_time": round(float(np.mean(times)), 3),
                "median_time": round(float(np.median(times)), 3),
                "std_time": round(float(np.std(times, ddof=1)) if len(times) > 1 else 0.0, 3),
                "best_time": round(float(np.min(times)), 3),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("median_time", ascending=True).reset_index(drop=True)