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
    cols = [c for c in ("modis_raw", "viirs_raw", "modis_terra", "modis_aqua") if c in sel.columns]
    return sel.groupby(["year", "month"], as_index=False)[cols].sum()


def bridge_aqua(df: pd.DataFrame) -> pd.DataFrame:
    """Make MODIS continuous across Aqua's launch (July 2002).

    From Aqua's first month: modis_raw = Terra + Aqua. Before it: Terra x s, where s = (Terra+Aqua)/Terra per
    calendar month, estimated over the Aqua era. Without separate Terra/Aqua columns nothing is changed.
    """
    df = df.copy()
    df["terra_scale"] = np.nan
    if not {"modis_terra", "modis_aqua"}.issubset(df.columns) or not (df.modis_aqua > 0).any():
        return df
    idx = df.year * 12 + df.month
    aq = idx >= idx[df.modis_aqua > 0].min()
    ref = df[aq & (df.modis_terra > 0)]
    k_g = (ref.modis_terra + ref.modis_aqua).sum() / ref.modis_terra.sum()
    s = ref.groupby("month")[["modis_terra", "modis_aqua"]].sum()
    scale = df.month.map((s.modis_terra + s.modis_aqua) / s.modis_terra).fillna(k_g)
    df["modis_raw"] = np.where(aq, df.modis_terra + df.modis_aqua, df.modis_terra * scale)
    df["terra_scale"] = np.where(aq, np.nan, scale)
    return df


def harmonize(monthly: pd.DataFrame, method: str = "global") -> pd.DataFrame:
    """Convert to MODIS-equivalent (Terra+Aqua) counts.

    Terra-only months before Aqua are scaled up (see bridge_aqua). Before VIIRS exists: MODIS as-is. From the
    first VIIRS month: VIIRS / k, where
      method='global'  -> k = sum(VIIRS)/sum(MODIS) over all overlap months
      method='seasonal'-> k per calendar month (captures season-dependent sensitivity)
    """
    df = bridge_aqua(monthly.sort_values(["year", "month"]).reset_index(drop=True))
    idx = df.year * 12 + df.month
    has_v = df.viirs_raw > 0
    modis_label = np.where(df.terra_scale.notna(), "MODIS Terra (scaled to Terra+Aqua)",
                           "MODIS Terra+Aqua" if "modis_aqua" in df.columns and (df.modis_aqua > 0).any() else "MODIS")
    if not has_v.any():
        df["k_used"] = np.nan
        df["sensor_source"] = modis_label
        df["hotspot_count"] = df.modis_raw.astype(float).round(1)
        return df
    era = idx >= idx[has_v].min()
    ov = df[era & (df.modis_raw > 0)]
    k_global = ov.viirs_raw.sum() / ov.modis_raw.sum() if len(ov) else np.nan
    k = pd.Series(k_global, index=df.index)
    if method == "seasonal":
        km = ov.groupby("month").apply(lambda g: g.viirs_raw.sum() / g.modis_raw.sum(), include_groups=False)
        k = df.month.map(km).fillna(k_global)
    df["k_used"] = np.where(era, k, np.nan)
    df["sensor_source"] = np.where(era, "VIIRS (harmonized)", modis_label)
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
    """Calendar month with the highest typical (median) harmonized count over the whole record."""
    g = df.groupby("month").hotspot_count.median()
    return int(g.idxmax()) if len(g) else None


def top_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    hi = df[df.severity.isin(["high", "extreme"])].sort_values("ratio_to_median", ascending=False).head(n)
    return hi[["year", "month", "hotspot_count", "hotspot_lo", "hotspot_hi", "baseline_historical_median", "ratio_to_median", "severity", "robust"]]


def brief(row, region_label: str, peak: int, month_names=MONTH_NAMES) -> str:
    """Plain-language situation summary for non-technical users."""
    m, y = month_names[int(row.month) - 1], int(row.year)
    ratio = row.ratio_to_median
    if row.anomaly_flag in ("high", "extreme"):
        level = f"**unusually high** burning activity, {ratio:.1f}x the historical median"
    elif row.anomaly_flag == "low":
        level = f"**unusually low** burning activity, {ratio:.1f}x the historical median"
    else:
        level = f"**normal** burning activity ({ratio:.1f}x the historical median)"
    peak_txt = f" The region's typical peak month is {month_names[peak - 1]}." if peak else ""
    if pd.notna(row.get("pctile")):
        rank = "all" if row.pctile >= 100 else f"{row.pctile:.0f}% of"
        peak_txt = f" That is higher than {rank} baseline {m} values." + peak_txt
    return (f"{m} {y} in {region_label} shows {level}: about {row.hotspot_count:,.0f} harmonized hotspots "
            f"vs a typical {row.baseline_historical_median:,.0f}.{peak_txt}")



# ---------- 1. out-of-sample harmonization error ----------
def loyo_error(df, method="seasonal", min_modis=5):
    """Leave-one-year-out error of VIIRS/k vs the MODIS record on overlap months.
    Returns (median, p80) relative error as fractions."""
    ov = df[(df.viirs_raw > 0) & (df.modis_raw > 0)]
    errs = []
    for y, te in ov.groupby("year"):
        tr = ov[ov.year != y]
        k = tr.viirs_raw.sum() / tr.modis_raw.sum()
        if method == "seasonal":
            s = tr.groupby("month")[["viirs_raw", "modis_raw"]].sum()
            k = te.month.map(s.viirs_raw / s.modis_raw).fillna(k)
        te = te[te.modis_raw >= min_modis]
        errs.append((te.viirs_raw / k - te.modis_raw).abs() / te.modis_raw)
    e = pd.concat(errs) if errs else pd.Series(dtype=float)
    return (float(e.median()), float(e.quantile(0.8))) if len(e) else (np.nan, np.nan)

def add_uncertainty(df, err, err_terra=0.0):
    """+/- band: `err` on VIIRS-harmonized months, `err_terra` on Terra-only months scaled to Terra+Aqua.
    Months measured directly by MODIS Terra+Aqua get no band."""
    src = df.sensor_source.astype(str)
    e = np.where(src.str.startswith("VIIRS"), err, np.where(src.str.startswith("MODIS Terra ("), err_terra, 0.0))
    return df.assign(err=e, hotspot_lo=df.hotspot_count * (1 - e), hotspot_hi=df.hotspot_count * (1 + e))


def terra_scale_error(df, min_count=5):
    """Leave-one-year-out error of scaling Terra-only to Terra+Aqua over the Aqua era (NaN if not applicable)."""
    if not {"modis_terra", "modis_aqua"}.issubset(df.columns) or not (df.modis_aqua > 0).any():
        return np.nan
    idx = df.year * 12 + df.month
    d = df[(idx >= idx[df.modis_aqua > 0].min()) & (df.modis_terra > 0)]
    errs = []
    for y, te in d.groupby("year"):
        tr = d[d.year != y]
        s = tr.groupby("month")[["modis_terra", "modis_aqua"]].sum()
        k = te.month.map((s.modis_terra + s.modis_aqua) / s.modis_terra).fillna(
            (tr.modis_terra + tr.modis_aqua).sum() / tr.modis_terra.sum())
        actual = te.modis_terra + te.modis_aqua
        keep = actual >= min_count
        errs.append(((te.modis_terra * k - actual).abs() / actual)[keep])
    e = pd.concat(errs) if errs else pd.Series(dtype=float)
    return float(e.median()) if len(e) else np.nan


def loyo_error_levels(df, method="seasonal", min_modis=5):
    """Leave-one-year-out median error of VIIRS/k vs MODIS for monthly, quarterly and annual totals."""
    ov = df[(df.viirs_raw > 0) & (df.modis_raw > 0)].copy()
    parts = []
    for y, te in ov.groupby("year"):
        tr = ov[ov.year != y]
        k = tr.viirs_raw.sum() / tr.modis_raw.sum()
        if method == "seasonal":
            s = tr.groupby("month")[["viirs_raw", "modis_raw"]].sum()
            k = te.month.map(s.viirs_raw / s.modis_raw).fillna(k)
        parts.append(te.assign(pred=te.viirs_raw / k))
    if not parts:
        return {}
    p = pd.concat(parts)
    p["q"] = (p.month - 1) // 3
    out = {}
    for name, keys in (("monthly", ["year", "month"]), ("quarterly", ["year", "q"]), ("annual", ["year"])):
        g = p.groupby(keys)
        a = g[["pred", "modis_raw"]].sum()
        n = g.month.nunique() if name != "monthly" else None
        if name == "quarterly":
            a = a[n.reindex(a.index) == 3]
        if name == "annual":
            a = a[n.reindex(a.index) == 12]
        a = a[a.modis_raw >= (min_modis if name == "monthly" else 3 * min_modis)]
        if len(a):
            out[name] = float(((a.pred - a.modis_raw).abs() / a.modis_raw).median())
    return out


# ---------- 2. severity tiers, raw or log scale, uncertainty-aware ----------
def classify(df, b0, b1, scale="raw", k=1.5):
    tf = np.log1p if scale == "log" else (lambda s: s)
    x = tf(df.hotspot_count)
    base = x[df.year.between(b0, b1)]
    q = base.groupby(df.month[base.index]).quantile([.25, .75]).unstack()
    q1, q3 = df.month.map(q[0.25]), df.month.map(q[0.75])
    iqr = (q3 - q1).clip(lower=0.2 if scale == "log" else 1.0)
    sev = pd.Series("normal", index=df.index)
    sev[x > q3 + k * iqr] = "high"
    sev[x > q3 + 3 * iqr] = "extreme"          # Tukey "far-out" fence
    sev[x < q1 - k * iqr] = "low"
    robust = (sev.isin(["high", "extreme"]) & (tf(df.hotspot_lo) > q3 + k * iqr)) | \
             ((sev == "low") & (tf(df.hotspot_hi) < q1 - k * iqr))
    return df.assign(severity=sev, robust=robust)

# ---------- 3. percentile rank ----------
def add_percentile(df, b0, b1):
    ref = {m: np.sort(g.hotspot_count.to_numpy()) for m, g in df[df.year.between(b0, b1)].groupby("month")}
    p = [100 * np.searchsorted(ref[m], v, side="right") / len(ref[m]) if m in ref else np.nan
         for m, v in zip(df.month, df.hotspot_count)]
    return df.assign(pctile=p)


# ---------- 4. one-stop analysis with a FIXED or ROLLING baseline ----------
def analyze(df, mode="fixed", b0=2001, b1=2020, window=10, scale="raw", k=1.5, min_years=7):
    """Baseline stats, severity tier, robustness and percentile for every month.

    mode='fixed'   : reference = same calendar month in years b0..b1
    mode='rolling' : reference = same calendar month in the `window` years BEFORE each month
                     (needs >= min_years of history, otherwise the month is 'no baseline')
    severity: normal / high (> Q3+1.5*IQR) / extreme (> Q3+3*IQR) / low (< Q1-1.5*IQR), on raw or log scale.
    robust : the flag still holds at the unfavourable end of the harmonization error band.
    """
    tf = np.log1p if scale == "log" else (lambda a: np.asarray(a, dtype=float))
    df = df.sort_values(["year", "month"]).reset_index(drop=True).copy()
    n = len(df)
    med, q1, q3, cnt, pct = (np.full(n, np.nan) for _ in range(5))
    sev = np.array(["no baseline"] * n, dtype=object)
    robust = np.zeros(n, dtype=bool)
    need = 3 if mode == "fixed" else min_years
    hc, lo, hi = (df[c].to_numpy(float) for c in ("hotspot_count", "hotspot_lo", "hotspot_hi"))
    yrs, mos = df.year.to_numpy(), df.month.to_numpy()
    for i in range(n):
        yr_ok = (yrs >= b0) & (yrs <= b1) if mode == "fixed" else (yrs >= yrs[i] - window) & (yrs < yrs[i])
        ref = hc[(mos == mos[i]) & yr_ok]
        if len(ref) < need:
            continue
        med[i], (q1[i], q3[i]), cnt[i] = np.median(ref), np.percentile(ref, [25, 75]), len(ref)
        rx, x = tf(ref), float(tf(hc[i]))
        t1, t3 = np.percentile(rx, [25, 75])
        iqr = max(t3 - t1, 0.2 if scale == "log" else 1.0)
        s = "normal"
        if x > t3 + k * iqr:
            s = "extreme" if x > t3 + 3 * iqr else "high"
        elif x < t1 - k * iqr:
            s = "low"
        sev[i] = s
        robust[i] = ((s in ("high", "extreme")) and float(tf(lo[i])) > t3 + k * iqr) or \
                    (s == "low" and float(tf(hi[i])) < t1 - k * iqr)
        pct[i] = 100 * np.searchsorted(np.sort(ref), hc[i], side="right") / len(ref)
    out = df.assign(baseline_historical_median=med, baseline_q1=q1, baseline_q3=q3, baseline_n=cnt,
                    pctile=pct, severity=sev, robust=robust)
    out["ratio_to_median"] = (out.hotspot_count / out.baseline_historical_median.where(
        out.baseline_historical_median > 0)).round(2)
    out["anomaly_flag"] = out.severity
    return out


def annual_trend(df):
    """Annual harmonized totals (complete years only) and the linear trend in % of the mean per year."""
    n = df.groupby("year").month.nunique()
    a = df[df.year.isin(n[n == 12].index)].groupby("year").hotspot_count.sum()
    if len(a) < 5:
        return a, None
    slope = np.polyfit(a.index.to_numpy(float), a.to_numpy(), 1)[0]
    return a, float(slope / a.mean() * 100)
