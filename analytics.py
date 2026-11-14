"""Shared analytics: region aggregation, MODIS/VIIRS harmonization, validation, baseline + anomalies."""
import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def aggregate_region(grid: pd.DataFrame, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame:
    """Sum RAW MODIS and VIIRS counts of all grid cells whose centre lies in the bbox."""
    sel = grid[grid.cell_lat.between(lat_min, lat_max) & grid.cell_lon.between(lon_min, lon_max)]
    if sel.empty:
        return pd.DataFrame(columns=["year", "month", "modis_raw", "viirs_raw"])
    return (sel.groupby(["year", "month"], as_index=False)
               .agg(modis_raw=("modis_raw", "sum"), viirs_raw=("viirs_raw", "sum")))


def harmonize(monthly: pd.DataFrame, method: str = "global") -> pd.DataFrame:
    """Convert to MODIS-equivalent counts.

    Before VIIRS exists: MODIS as-is. From the first VIIRS month: VIIRS / k, where
      method='global'  -> k = sum(VIIRS)/sum(MODIS) over all overlap months
      method='seasonal'-> k per calendar month (captures season-dependent sensitivity)
    """
    df = monthly.sort_values(["year", "month"]).reset_index(drop=True).copy()
    idx = df.year * 12 + df.month
    has_v = df.viirs_raw > 0
    if not has_v.any():
        df["k_used"] = np.nan
        df["sensor_source"] = "MODIS"
        df["hotspot_count"] = df.modis_raw.astype(float)
        return df
    era = idx >= idx[has_v].min()
    ov = df[era & (df.modis_raw > 0)]
    k_global = ov.viirs_raw.sum() / ov.modis_raw.sum() if len(ov) else np.nan
    k = pd.Series(k_global, index=df.index)
    if method == "seasonal":
        km = ov.groupby("month").apply(lambda g: g.viirs_raw.sum() / g.modis_raw.sum(), include_groups=False)
        k = df.month.map(km).fillna(k_global)
    df["k_used"] = np.where(era, k, np.nan)
    df["sensor_source"] = np.where(era, "VIIRS (harmonized)", "MODIS")
    df["hotspot_count"] = np.where(era, df.viirs_raw / k, df.modis_raw).round(1)
    return df


def validation_stats(df: pd.DataFrame):
    """Cross-sensor agreement over the overlap period (None if there is no overlap)."""
    ov = df[(df.viirs_raw > 0) & (df.modis_raw > 0)]
    if len(ov) < 6:
        return None
    r = float(np.corrcoef(np.log1p(ov.modis_raw), np.log1p(ov.viirs_raw))[0, 1])
    solid = ov[ov.modis_raw >= 5]
    pct = ((solid.viirs_raw / solid.k_used - solid.modis_raw).abs() / solid.modis_raw).median() * 100 \
        if len(solid) else np.nan
    return {"months": len(ov), "r": r, "k": float(ov.viirs_raw.sum() / ov.modis_raw.sum()),
            "median_pct_error": float(pct)}


def add_baseline_and_anomaly(monthly: pd.DataFrame, baseline_start: int, baseline_end: int,
                             iqr_k: float = 1.5) -> pd.DataFrame:
    """Per-calendar-month baseline median/IQR over the baseline years + anomaly flag.

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
    spread = ((df.baseline_q3 - df.baseline_q1).clip(lower=1.0)
              .combine(df.baseline_historical_median * 0.1, max))
    df["anomaly_flag"] = "normal"
    df.loc[df.hotspot_count > df.baseline_q3 + iqr_k * spread, "anomaly_flag"] = "high"
    df.loc[df.hotspot_count < df.baseline_q1 - iqr_k * spread, "anomaly_flag"] = "low"
    df["ratio_to_median"] = (df.hotspot_count / df.baseline_historical_median.where(
        df.baseline_historical_median > 0)).round(2)
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def peak_month(df: pd.DataFrame):
    """Calendar month with the highest baseline median (typical peak burning month)."""
    b = df.drop_duplicates("month")
    return int(b.loc[b.baseline_historical_median.idxmax(), "month"]) if len(b) else None


def top_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    hi = df[df.anomaly_flag == "high"].sort_values("ratio_to_median", ascending=False).head(n)
    return hi[["year", "month", "hotspot_count", "baseline_historical_median", "ratio_to_median", "sensor_source"]]


def brief(row, region_label: str, peak: int, month_names=MONTH_NAMES) -> str:
    """Plain-language situation summary for non-technical users."""
    m, y = month_names[int(row.month) - 1], int(row.year)
    ratio = row.ratio_to_median
    if row.anomaly_flag == "high":
        level = f"**unusually high** burning activity, {ratio:.1f}x the historical median"
    elif row.anomaly_flag == "low":
        level = f"**unusually low** burning activity, {ratio:.1f}x the historical median"
    else:
        level = f"**normal** burning activity ({ratio:.1f}x the historical median)"
    peak_txt = f" The region's typical peak month is {month_names[peak - 1]}." if peak else ""
    return (f"{m} {y} in {region_label} shows {level}: about {row.hotspot_count:,.0f} harmonized hotspots "
            f"vs a typical {row.baseline_historical_median:,.0f}.{peak_txt}")
