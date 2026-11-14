"""Shared analytics: region aggregation, historical baseline and anomaly flags."""
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def aggregate_region(grid: pd.DataFrame, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame:
    """Sum harmonized hotspot counts of all grid cells whose centre lies in the bbox."""
    sel = grid[grid.cell_lat.between(lat_min, lat_max) & grid.cell_lon.between(lon_min, lon_max)]
    if sel.empty:
        return pd.DataFrame(columns=["year", "month", "sensor_source", "hotspot_count", "modis_raw", "viirs_raw"])
    return (sel.groupby(["year", "month"], as_index=False)
               .agg(sensor_source=("sensor_source", "first"),
                    hotspot_count=("hotspot_count", "sum"),
                    modis_raw=("modis_raw", "sum"),
                    viirs_raw=("viirs_raw", "sum"))
               .round({"hotspot_count": 1}))


def estimate_factor(monthly: pd.DataFrame):
    """VIIRS-per-MODIS factor k over the months where both sensors report (None if no overlap)."""
    ov = monthly[monthly.viirs_raw > 0]
    return None if ov.modis_raw.sum() == 0 else ov.viirs_raw.sum() / ov.modis_raw.sum()


def add_baseline_and_anomaly(monthly: pd.DataFrame, baseline_start: int, baseline_end: int,
                             iqr_k: float = 1.5) -> pd.DataFrame:
    """Add per-calendar-month baseline median/IQR (over the baseline years) and an anomaly flag.

    flag = 'high' if count > Q3 + k*spread, 'low' if count < Q1 - k*spread, else 'normal'.
    """
    df = monthly.copy()
    base = df[df.year.between(baseline_start, baseline_end)]
    stats = (base.groupby("month")["hotspot_count"]
                 .agg(baseline_historical_median="median",
                      baseline_q1=lambda s: s.quantile(0.25),
                      baseline_q3=lambda s: s.quantile(0.75))
                 .reset_index())
    df = df.merge(stats, on="month", how="left")
    # floor the spread so near-empty months don't flag every single hotspot
    spread = ((df.baseline_q3 - df.baseline_q1)
              .clip(lower=1.0)
              .combine(df.baseline_historical_median * 0.1, max))
    df["anomaly_flag"] = "normal"
    df.loc[df.hotspot_count > df.baseline_q3 + iqr_k * spread, "anomaly_flag"] = "high"
    df.loc[df.hotspot_count < df.baseline_q1 - iqr_k * spread, "anomaly_flag"] = "low"
    df["ratio_to_median"] = (df.hotspot_count / df.baseline_historical_median.where(
        df.baseline_historical_median > 0)).round(2)
    return df.sort_values(["year", "month"]).reset_index(drop=True)
