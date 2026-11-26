from __future__ import annotations

import pandas as pd


def clean_laps(
    raw_laps: pd.DataFrame,
    *,
    drop_lap_one: bool = True,
    max_lap_ratio: float = 1.15,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter and normalize lap-level records into green-flag race laps.

    Parameters
    ----------
    raw_laps : pd.DataFrame
        DataFrame built from a driver's ``laptimes.json`` dictionary.
    drop_lap_one : bool, default True
        Whether to exclude lap 1 (standing start and grid launch dynamics).
    max_lap_ratio : float, default 1.15
        Upper multiplier relative to median green-flag lap time used to drop
        anomalous slow laps (such as SC/VSC/incidents not captured by status).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Cleaned laps DataFrame and a summary dictionary of row counts per step.
    """
    df = raw_laps.copy()
    counts = {"initial": len(df)}

    # Normalize stringified 'None' tokens into real NaN values
    none_tokens = {"None", "none", "NaN", "nan", ""}
    df = df.replace(list(none_tokens), pd.NA)

    # Keep green-flag status ('1') when present
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip() == "1"].copy()
    counts["after_green_flag"] = len(df)

    # Drop pit-in and pit-out laps
    for pit_col in ("pin", "pout"):
        if pit_col in df.columns:
            df = df[df[pit_col].isna()].copy()
    counts["after_pit_filter"] = len(df)

    # Optionally drop lap 1
    if drop_lap_one and "lap" in df.columns:
        df["lap"] = pd.to_numeric(df["lap"], errors="coerce")
        df = df[df["lap"] > 1].copy()
    counts["after_lap1_filter"] = len(df)

    # Cast timing and sector columns to numeric
    for num_col in ("time", "s1", "s2", "s3", "life", "pos", "stint"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df = df.dropna(subset=["lap", "time"]).copy()
    df["lap"] = df["lap"].astype(int)
    counts["after_valid_time"] = len(df)

    # Drop extreme slow outliers relative to median green-flag pace
    if not df.empty and max_lap_ratio > 0:
        median_lap = float(df["time"].median())
        df = df[df["time"] <= median_lap * max_lap_ratio].copy()
    counts["final_clean_laps"] = len(df)

    df = df.sort_values("lap").reset_index(drop=True)
    return df, counts

def build_stints_tyres(clean_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned green-flag laps into a stint and tyre summary table.

    Parameters
    ----------
    clean_laps_df : pd.DataFrame
        Cleaned lap-level DataFrame output by ``clean_laps``.

    Returns
    -------
    pd.DataFrame
        One row per contiguous stint/compound segment with lap bounds, stint
        length, tyre age range (start life, end life), and pace summary metrics.
    """
    required = {"stint", "compound", "lap", "time"}
    missing = required - set(clean_laps_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for stint aggregation: {sorted(missing)}")

    if clean_laps_df.empty:
        return pd.DataFrame(
            columns=[
                "drv",
                "team",
                "stint",
                "compound",
                "start_lap",
                "end_lap",
                "clean_laps",
                "start_life",
                "end_life",
                "mean_time",
                "median_time",
                "best_time",
            ]
        )

    df = clean_laps_df.sort_values("lap").copy()

    # Split if stint number changes or compound changes mid-stint
    segment_id = (
        (df["stint"] != df["stint"].shift())
        | (df["compound"].astype(str) != df["compound"].astype(str).shift())
    ).cumsum()

    grouped = df.groupby(segment_id, sort=False)
    rows: list[dict[str, object]] = []

    for _, grp in grouped:
        first_row = grp.iloc[0]
        row: dict[str, object] = {
            "drv": first_row["drv"] if "drv" in grp.columns else pd.NA,
            "team": first_row["team"] if "team" in grp.columns else pd.NA,
            "stint": int(first_row["stint"]),
            "compound": str(first_row["compound"]),
            "start_lap": int(grp["lap"].min()),
            "end_lap": int(grp["lap"].max()),
            "clean_laps": int(len(grp)),
            "start_life": int(grp["life"].min()) if "life" in grp.columns and grp["life"].notna().any() else pd.NA,
            "end_life": int(grp["life"].max()) if "life" in grp.columns and grp["life"].notna().any() else pd.NA,
            "mean_time": round(float(grp["time"].mean()), 3),
            "median_time": round(float(grp["time"].median()), 3),
            "best_time": round(float(grp["time"].min()), 3),
        }
        rows.append(row)

    return pd.DataFrame(rows)