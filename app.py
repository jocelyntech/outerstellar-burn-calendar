"""Team Outerstellar - Harmonized Burning Activity Calendar.  Run: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import (MONTH_NAMES, add_uncertainty, aggregate_region, analyze, annual_trend, brief, harmonize,
                       loyo_error, loyo_error_levels, peak_month, terra_scale_error, top_anomalies,
                       validation_stats)
from generate_mock_data import GRID_CSV, main as build_mock_data

REAL_CSV = Path("data/grid_monthly_real.csv")  # created by build_grid_from_firms.py
st.set_page_config(page_title="Harmonized Burning Activity Calendar", page_icon="🔥", layout="wide")

PRESETS = {  # name: (lat_min, lat_max, lon_min, lon_max); "Entire dataset" is filled in from the data
    "Entire dataset": None,
    "Peninsular Malaysia": (1.0, 7.0, 99.0, 105.0),
    "Sabah and Sarawak (Borneo)": (0.5, 7.5, 108.5, 119.5),
    "Mainland SE Asia (TH/LA/MM/VN/KH)": (8.0, 24.0, 93.0, 110.0),
    "Sumatra": (-6.0, 6.0, 94.0, 109.0),
    "Kalimantan (Borneo)": (-4.5, 4.5, 108.0, 119.0),
    "Custom": None,
}
FLAG_COLORS = {"extreme": "#7f0000", "high": "#d62728", "low": "#1f77b4", "normal": "#9aa0a6",
               "no baseline": "#d9d9d9"}


# ---------- data ----------
def _data_version():
    """Changes whenever data/grid_monthly_real.csv is added or replaced, so the cache refreshes."""
    return (REAL_CSV.stat().st_mtime_ns, REAL_CSV.stat().st_size) if REAL_CSV.exists() else None


@st.cache_data(show_spinner="Loading gridded hotspot data...")
def load_grid(version):
    """Real FIRMS-derived grid if present, otherwise the mock grid (built on first run)."""
    if REAL_CSV.exists():
        return pd.read_csv(REAL_CSV), False
    if not GRID_CSV.exists():
        build_mock_data()
    return pd.read_csv(GRID_CSV), True


def cell_size(grid):
    """Grid cell size in degrees, inferred from the data."""
    diffs = np.diff(np.sort(grid.cell_lat.unique()))
    return float(diffs[diffs > 1e-9].min()) if len(diffs[diffs > 1e-9]) else 1.0


def year_range_slider(label, years, default=None, container=st.sidebar):
    lo, hi = int(years.min()), int(years.max())
    if lo == hi:
        container.caption(f"{label}: {lo} (only year available)")
        return lo, hi
    d = default or (lo, hi)
    d = (max(lo, d[0]), min(hi, d[1]))
    if d[0] >= d[1]:
        d = (lo, hi)
    return container.slider(label, lo, hi, d)


# ---------- sidebar ----------
def sidebar_controls(grid):
    years = grid.year.drop_duplicates()
    ext = (float(np.floor(grid.cell_lat.min() - 1)), float(np.ceil(grid.cell_lat.max() + 1)),
           float(np.floor(grid.cell_lon.min() - 1)), float(np.ceil(grid.cell_lon.max() + 1)))
    st.sidebar.header("1. Region of interest")
    def covered(box):  # preset box must lie inside the data extent and contain data
        return box is None or (box[0] >= ext[0] and box[1] <= ext[1] and box[2] >= ext[2]
                               and box[3] <= ext[3] and len(aggregate_region(grid, *box)) > 0)
    options = [p for p, box in PRESETS.items() if covered(box)]
    qp = st.query_params
    hidden = [p for p, b in PRESETS.items() if b and p not in options]
    if hidden:
        st.sidebar.caption("Not covered by the loaded data: " + ", ".join(hidden))
    # stable widget identity: set the initial value once via session_state (a changing `index=` made
    # Streamlit treat the widget as new and silently discard every second click)
    if st.session_state.get("preset") not in options:
        st.session_state["preset"] = qp.get("preset") if qp.get("preset") in options else options[0]
    preset = st.sidebar.selectbox("Preset", options, key="preset")
    d = PRESETS[preset] or ext
    custom = preset == "Custom"
    c1, c2 = st.sidebar.columns(2)
    kk = f"{preset}"
    lat_min = c1.number_input("Lat min", ext[0], ext[1], d[0], 0.5, key=f"a{kk}", disabled=not custom)
    lat_max = c2.number_input("Lat max", ext[0], ext[1], d[1], 0.5, key=f"b{kk}", disabled=not custom)
    lon_min = c1.number_input("Lon min", ext[2], ext[3], d[2], 0.5, key=f"c{kk}", disabled=not custom)
    lon_max = c2.number_input("Lon max", ext[2], ext[3], d[3], 0.5, key=f"d{kk}", disabled=not custom)

    st.sidebar.header("2. Harmonization")
    if st.session_state.get("method") not in ("global", "seasonal"):
        st.session_state["method"] = qp.get("method") if qp.get("method") in ("global", "seasonal") else "seasonal"
    method = st.sidebar.radio("Cross-sensor factor", ["global", "seasonal"], horizontal=True, key="method",
                              help="global: one VIIRS/MODIS factor. seasonal: a factor per calendar month.")
    scale = st.sidebar.radio("Detection scale", ["raw", "log"], horizontal=True,
                             help="log treats a 2x jump and a 2x drop symmetrically, so quiet months can be flagged too.")

    st.sidebar.header("3. Baseline & selection")
    bmode = st.sidebar.radio("Baseline type", ["Fixed", "Rolling"], horizontal=True,
                             help="Fixed: compare every month with one set of years (stable, and it exposes "
                                  "long-term change). Rolling: compare each month with the same month in the "
                                  "previous N years (needs several years of history, so it is noisier).")
    if bmode == "Rolling":
        window = st.sidebar.slider("Years to look back", 8, 15, 10)
        base = (2001, 2020)
    else:
        window = 10
        base = year_range_slider("Baseline years (historical median)", years, (2001, 2020))
    latest = int(years.max())
    sel_year = st.sidebar.selectbox("Year", sorted(years, reverse=True))
    last_m = int(grid[grid.year == latest].month.max())
    sel_month = st.sidebar.selectbox("Month", range(1, 13), index=last_m - 1,
                                     format_func=lambda m: MONTH_NAMES[m - 1])
    yr = year_range_slider("Calendar year range", years)
    return preset, (lat_min, lat_max, lon_min, lon_max), method, scale, (bmode.lower(), base, window), (sel_year, sel_month), yr


# ---------- charts ----------
def calendar_heatmap(df, mode):
    if mode == "Hotspot count":
        z = df.pivot(index="year", columns="month", values="hotspot_count")
        scale, zmid, zmax, title = "YlOrRd", None, None, "Hotspots"
    else:
        z = df.pivot(index="year", columns="month", values="ratio_to_median")
        scale, zmid, zmax, title = "RdBu_r", 1.0, 3.0, "x median"
    flags = df.pivot(index="year", columns="month", values="anomaly_flag").reindex_like(z)
    src = df.pivot(index="year", columns="month", values="sensor_source").reindex_like(z)
    z = z.reindex(columns=range(1, 13))
    flags, src = flags.reindex(columns=range(1, 13)), src.reindex(columns=range(1, 13))
    fig = go.Figure(go.Heatmap(
        z=z.values, x=MONTH_NAMES, y=z.index, colorscale=scale, zmid=zmid, zmin=0 if zmid else None,
        zmax=zmax, colorbar=dict(title=title), xgap=2, ygap=2,
        customdata=np.dstack([flags.values, src.values]),
        hovertemplate="%{y} %{x}<br>value: %{z}<br>flag: %{customdata[0]}<br>source: %{customdata[1]}<extra></extra>"))
    mk = df[df.anomaly_flag.isin(["high", "extreme", "low"])]
    fig.add_trace(go.Scatter(x=[MONTH_NAMES[m - 1] for m in mk.month], y=mk.year, mode="markers", showlegend=False,
                             hoverinfo="skip", marker=dict(size=7, color="black", line=dict(width=1, color="white"),
                             symbol=["triangle-down" if f == "low" else "triangle-up" for f in mk.anomaly_flag])))
    fig.update_yaxes(autorange="reversed", dtick=1, type="category")
    fig.update_xaxes(side="top")
    fig.update_layout(height=max(420, 22 * len(z) + 120), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def month_across_years(df, month, sel_year):
    m = df[df.month == month].sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(m.year) + list(m.year[::-1]),
                             y=list(m.baseline_q3) + list(m.baseline_q1[::-1]),
                             fill="toself", fillcolor="rgba(120,120,120,.18)", line=dict(width=0),
                             name="Baseline IQR", hoverinfo="skip"))
    fig.add_trace(go.Bar(x=m.year, y=m.hotspot_count, name="Hotspots",
                         marker_color=[FLAG_COLORS[f] if y == sel_year else "#f2a65a"
                                       for f, y in zip(m.anomaly_flag, m.year)]))
    fig.add_trace(go.Scatter(x=m.year, y=m.baseline_historical_median, mode="lines", name="Baseline median",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(title=f"{MONTH_NAMES[month - 1]} hotspots by year (selected year coloured by flag)",
                      xaxis_title="Year", yaxis_title="Harmonized hotspots", height=420,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def year_vs_baseline(df, year):
    y = df[df.year == year].sort_values("month")
    base = y.dropna(subset=["baseline_q1", "baseline_q3"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[MONTH_NAMES[m - 1] for m in base.month], y=base.baseline_q3,
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[MONTH_NAMES[m - 1] for m in base.month], y=base.baseline_q1, fill="tonexty",
                             line=dict(width=0), fillcolor="rgba(120,120,120,.2)", name="Baseline IQR",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[MONTH_NAMES[m - 1] for m in base.month], y=base.baseline_historical_median,
                             name="Baseline median", line=dict(color="black", dash="dash")))
    fig.add_trace(go.Scatter(x=[MONTH_NAMES[m - 1] for m in y.month], y=y.hotspot_count, name=str(year),
                             mode="lines+markers", line=dict(color="#d62728"),
                             marker=dict(color=[FLAG_COLORS[f] for f in y.anomaly_flag], size=9)))
    fig.update_layout(title=f"{year} vs historical baseline", yaxis_title="Harmonized hotspots", height=420,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def trend_chart(annual):
    x = annual.index.to_numpy(float)
    fig = go.Figure(go.Bar(x=annual.index, y=annual.values, name="Annual total", marker_color="#f2a65a"))
    fig.add_trace(go.Scatter(x=annual.index, y=np.polyval(np.polyfit(x, annual.values, 1), x),
                             name="Linear trend", line=dict(color="black", dash="dash")))
    fig.update_layout(height=340, yaxis_title="Harmonized hotspots / year", xaxis_title="Year (complete years)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    return fig


def harmonization_chart(df):
    x = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    v = df.viirs_raw.where(df.viirs_raw > 0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Raw counts: the two sensors disagree on scale",
                                        "Harmonized (MODIS-equivalent): one consistent series"))
    fig.add_trace(go.Scatter(x=x, y=df.modis_raw, name="MODIS raw (1 km, Terra+Aqua)", line=dict(color="#e67e22")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=v, name="VIIRS raw (375 m)", line=dict(color="#8e44ad")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df.hotspot_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=df.hotspot_lo, fill="tonexty", fillcolor="rgba(192,57,43,.18)", line=dict(width=0),
                             name="Typical harmonization error", hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=df.hotspot_count, name="Harmonized", line=dict(color="#c0392b")), 2, 1)
    if (df.viirs_raw > 0).any():
        first = df[df.viirs_raw > 0].iloc[0]
        fig.add_vline(x=pd.Timestamp(year=int(first.year), month=int(first.month), day=1),
                      line_dash="dot", line_color="gray")
    fig.update_layout(height=560, legend=dict(orientation="h", y=-0.12), margin=dict(t=60))
    fig.update_yaxes(title_text="Hotspots", row=1, col=1)
    fig.update_yaxes(title_text="Hotspots", row=2, col=1)
    return fig


def region_map(grid, bbox, year, month, k, tscale=np.nan):
    lat0, lat1, lon0, lon1 = bbox
    c = grid[(grid.year == year) & (grid.month == month)
             & grid.cell_lat.between(lat0, lat1) & grid.cell_lon.between(lon0, lon1)].copy()
    # MODIS-equivalent counts (Terra-only months before Aqua are scaled to Terra+Aqua)
    c["val"] = c.viirs_raw / k if pd.notna(k) else c.modis_raw * (tscale if pd.notna(tscale) else 1.0)
    fig = go.Figure(go.Scattergeo(
        lat=c.cell_lat, lon=c.cell_lon, mode="markers",
        marker=dict(size=np.clip(c.val * 1.2 + 4, 4, 40), color=c.val, colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Hotspots"), line=dict(width=0.3, color="black")),
        hovertemplate="%{lat}, %{lon}<br>%{marker.color:.1f} hotspots<extra></extra>"))
    fig.update_geos(lataxis_range=[lat0 - 1, lat1 + 1], lonaxis_range=[lon0 - 1, lon1 + 1],
                    showland=True, landcolor="#2b2b2b", showocean=True, oceancolor="#1a2a3a",
                    showcountries=True, countrycolor="#777")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    return fig


# ---------- main ----------
def main():
    st.title("🔥 Harmonized Burning Activity Calendar")
    grid, is_mock = load_grid(_data_version())
    st.caption("Team Outerstellar · NASA Space Apps · MODIS (1 km, 2000→) + VIIRS (375 m, 2012→) harmonized "
               "to MODIS-equivalent counts · " + ("**MOCK DATA**" if is_mock else "**NASA FIRMS data**"))
    if is_mock:
        st.warning("Demo mode: showing simulated data because `data/grid_monthly_real.csv` was not found. "
                   "Add that file (made by `build_grid_from_firms.py` from FIRMS archive downloads) to use real detections.")

    preset, bbox, method, scale, (bmode, (b0, b1), window), (sel_year, sel_month), (y0, y1) = sidebar_controls(grid)
    st.query_params.update(preset=preset, method=method)
    monthly_raw = aggregate_region(grid, *bbox)
    if monthly_raw.empty:
        st.warning(f"No grid cells inside that bounding box (cells are {cell_size(grid):g}°). Enlarge the box.")
        return
    harmonized = harmonize(monthly_raw, method)
    med_err, p80_err = loyo_error(harmonized, method)
    err_ta = terra_scale_error(harmonized)
    levels = loyo_error_levels(harmonized, method)
    harmonized = add_uncertainty(harmonized, 0 if np.isnan(med_err) else med_err,
                                 0 if np.isnan(err_ta) else err_ta)
    full = analyze(harmonized, bmode, b0, b1, window, scale)
    base_desc = (f"the same month in the previous {window} years (needs at least 7)" if bmode == "rolling"
                 else f"the same month in {b0}-{b1}")
    # partial-month guard: the current UTC month is incomplete, so never flag it
    now = pd.Timestamp.now(tz="UTC")
    partial = (full.year == now.year) & (full.month == now.month)
    full.loc[partial, ["anomaly_flag", "severity"]] = "normal"
    full["partial"] = partial
    view = full[full.year.between(y0, y1)]
    peak = peak_month(full)

    row = full[(full.year == sel_year) & (full.month == sel_month)]
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4 = st.columns([1, 1, 1, 1.4])
        m1.metric(f"{MONTH_NAMES[sel_month - 1]} {sel_year}", f"{r.hotspot_count:,.0f}")
        m2.metric("Baseline median", f"{r.baseline_historical_median:,.0f}" if pd.notna(r.baseline_historical_median) else "n/a")
        m3.metric("vs median", f"{r.ratio_to_median}x" if pd.notna(r.ratio_to_median) else "n/a")
        m4.metric("Anomaly", r.anomaly_flag.capitalize())
        if pd.notna(r.ratio_to_median):
            label = preset if preset not in ("Custom", "Entire dataset") else "the selected region"
            st.info("**Situation brief:** " + brief(r, label, peak))
        st.caption(f"Sensor source for this month: {r.sensor_source} · Baseline = {base_desc}")
    else:
        st.info("No data for the selected month.")

    st.subheader("Burning activity calendar")
    mode = st.radio("Colour by", ["Hotspot count", "Anomaly vs baseline (ratio to median)"], horizontal=True)
    st.plotly_chart(calendar_heatmap(view, mode), width="stretch")

    st.subheader("Selected period vs historical baseline")
    t1, t2 = st.tabs(["Selected month across years", "Selected year across months"])
    t1.plotly_chart(month_across_years(view, sel_month, sel_year), width="stretch")
    t2.plotly_chart(year_vs_baseline(full, sel_year), width="stretch")

    st.subheader("Long-term trend")
    annual, slope = annual_trend(harmonized.assign(hotspot_count=harmonized.hotspot_count))
    if slope is not None:
        word = "falling" if slope < 0 else "rising"
        st.caption(f"Annual harmonized hotspots are {word} by about **{abs(slope):.1f}% of the average per year** "
                   "(complete years only). With a fixed baseline this trend makes recent months look low; "
                   "the rolling baseline is an alternative view that compares each month with its recent past.")
        st.plotly_chart(trend_chart(annual), width="stretch")
    else:
        st.caption("Not enough complete years in this region for a trend estimate.")

    st.subheader("Why harmonize? MODIS vs VIIRS")
    v = validation_stats(harmonized)
    if v:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overlap months", v["months"])
        c2.metric("VIIRS / MODIS factor k", f"{v['k']:.2f}")
        c3.metric("Cross-sensor correlation (log)", f"{v['r']:.2f}")
        c4.metric("Out-of-sample error (leave-one-year-out)",
                  f"{med_err:.0%}" if pd.notna(med_err) else "n/a", help=f"80% of months within {p80_err:.0%}")
        lv = " · ".join(f"{k} {v:.0%}" for k, v in levels.items())
        st.caption("Validation over the months where both sensors report (leave-one-year-out). "
                   f"Median error by time scale: {lv}; random counting noise averages out in longer totals."
                   + (f" Terra-only months before Aqua (Nov 2000-Jun 2002) are scaled to Terra+Aqua with "
                      f"≈{err_ta:.0%} error." if pd.notna(err_ta) else ""))
    else:
        st.caption("Not enough overlapping MODIS/VIIRS months in this region to validate the harmonization.")
    st.plotly_chart(harmonization_chart(view), width="stretch")

    st.subheader("Most unusual months in this region")
    top = top_anomalies(full)
    if len(top):
        st.dataframe(top, width="stretch", hide_index=True)
    else:
        st.caption("No high anomalies found.")

    st.subheader(f"Map: {MONTH_NAMES[sel_month - 1]} {sel_year} intensity by {cell_size(grid):g}° cell")
    k_sel = row.iloc[0].k_used if not row.empty else np.nan
    t_sel = row.iloc[0].terra_scale if not row.empty else np.nan
    st.plotly_chart(region_map(grid, bbox, sel_year, sel_month, k_sel, t_sel), width="stretch")

    st.subheader("Export")
    if not row.empty and pd.notna(row.iloc[0].ratio_to_median):
        st.download_button("📄 One-page situation brief (Markdown)", ("# Burning activity brief\n\n" + brief(row.iloc[0], preset, peak)
                           + f"\n\n_Harmonization error ≈ ±{med_err:.0%} (leave-one-year-out). Source: NASA FIRMS MODIS + VIIRS._").encode(),
                           file_name=f"brief_{sel_year}-{sel_month:02d}.md")
    out = view.assign(lat_min=bbox[0], lat_max=bbox[1], lon_min=bbox[2], lon_max=bbox[3],
                      baseline_start=b0, baseline_end=b1, harmonization=method)
    st.download_button("⬇️ Download filtered results (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"burning_calendar_{y0}-{y1}.csv", mime="text/csv")
    with st.expander("Preview table"):
        st.dataframe(out, width="stretch")
    with st.expander("Method, assumptions and limitations"):
        st.markdown(
            "- **Pre-VIIRS:** MODIS counts as-is. **From the first VIIRS month:** VIIRS ÷ *k*, with *k* = Σ VIIRS / "
            "Σ MODIS over the overlap (*global*) or per calendar month (*seasonal*).\n"
            "- **Baseline:** *Rolling* compares each month with the same calendar month in the previous N years "
            "(default 10); *Fixed* uses one set of years. Flags: *high* above Q3 + 1.5×IQR, *extreme* above "
            "Q3 + 3×IQR, *low* below Q1 − 1.5×IQR, optionally on a log scale so quiet months can be flagged too. "
            "*Robust* means the flag still holds at the unfavourable end of the harmonization error band.\n"
            "- **Consistency:** MODIS = Terra + Aqua (Aqua starts July 2002; earlier Terra-only months are scaled "
            "to Terra+Aqua per calendar month), VIIRS = Suomi-NPP only, low-confidence detections removed. "
            "VIIRS (13:30 overpass) agrees better with Terra+Aqua than with Terra alone (10:30).\n"
            "- **Limits:** counts are detections, not burned area; cloud/smoke and overpass time affect detection; "
            "a hotspot count ratio (*k*) assumes similar fire sizes between sensors.")


if __name__ == "__main__":
    main()
