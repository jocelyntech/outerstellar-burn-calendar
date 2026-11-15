"""Team Outerstellar - Harmonized Burning Activity Calendar.  Run: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from burn_analytics import (MONTH_NAMES, add_uncertainty, aggregate_region, analyze, annual_trend, attach_coverage,
                       brief, harmonize, loyo_error, loyo_error_by_size, loyo_error_levels, peak_month,
                       terra_scale_error, top_anomalies, validation_stats)
from generate_mock_data import GRID_CSV, main as build_mock_data

REAL_CSV = Path("data/grid_monthly_real.csv")  # created by build_grid_from_firms.py
COVERAGE_CSV = Path("data/coverage.csv")        # same script: share of days each satellite reported
st.set_page_config(page_title="Harmonized Burning Activity Calendar", page_icon="🔥", layout="wide")

PRESETS = {  # name: ((lat_min, lat_max, lon_min, lon_max) or None = whole data extent, country or None)
    "Entire dataset": (None, None),
    "Malaysia (all)": (None, "Malaysia"),
    "Indonesia (all)": (None, "Indonesia"),
    "Peninsular Malaysia": ((1.0, 7.0, 99.0, 105.0), "Malaysia"),
    "Sabah and Sarawak (Borneo)": ((0.5, 7.5, 108.5, 119.5), "Malaysia"),
    "Sumatra": ((-6.0, 6.0, 94.0, 109.0), "Indonesia"),
    "Kalimantan (Borneo)": ((-4.5, 4.5, 108.0, 119.0), "Indonesia"),
    "Java": ((-9.0, -5.5, 105.0, 115.0), "Indonesia"),
    "Papua": ((-10.0, 0.5, 130.0, 141.5), "Indonesia"),
    "Mainland SE Asia (TH/LA/MM/VN/KH)": ((8.0, 24.0, 93.0, 110.0), None),
    "Custom": (None, None),
}
PALETTE = {"heat": "YlOrRd", "div": "RdBu_r"}   # swapped for colour-blind-safe ramps by a sidebar toggle
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
    pend = st.session_state.pop("pending_aoi", None)   # box drawn on the map in the previous run
    if pend:
        st.session_state["aoi"] = pend
        st.session_state["aoi_v"] = st.session_state.get("aoi_v", 0) + 1   # new widget keys -> new default values
        st.session_state["preset"] = "Custom"
    st.sidebar.header("1. Region of interest")
    countries = sorted(grid.country.unique()) if "country" in grid.columns else []

    def covered(item):  # preset box must lie inside the data extent and contain data
        box, ctry = item
        if ctry and countries and ctry not in countries:
            return False
        if ctry and not countries:
            ctry = None  # grid without country labels (mock data): ignore the country part
        if box is None:
            return True
        return (box[0] >= ext[0] and box[1] <= ext[1] and box[2] >= ext[2] and box[3] <= ext[3]
                and len(aggregate_region(grid, *box, ctry)) > 0)
    options = [p for p, item in PRESETS.items() if covered(item)]
    qp = st.query_params
    hidden = [p for p, (b, _) in PRESETS.items() if b and p not in options]
    if hidden:
        st.sidebar.caption("Not covered by the loaded data: " + ", ".join(hidden))
    # stable widget identity: set the initial value once via session_state (a changing `index=` made
    # Streamlit treat the widget as new and silently discard every second click)
    if st.session_state.get("preset") not in options:
        st.session_state["preset"] = qp.get("preset") if qp.get("preset") in options else options[0]
    preset = st.sidebar.selectbox("Preset", options, key="preset")
    box, country = PRESETS[preset]
    custom = preset == "Custom"
    d = box or (st.session_state.get("aoi") if custom else None) or ext
    c1, c2 = st.sidebar.columns(2)
    kk = f"{preset}{st.session_state.get('aoi_v', 0)}"
    lat_min = c1.number_input("Lat min", ext[0], ext[1], d[0], 0.5, key=f"a{kk}", disabled=not custom)
    lat_max = c2.number_input("Lat max", ext[0], ext[1], d[1], 0.5, key=f"b{kk}", disabled=not custom)
    lon_min = c1.number_input("Lon min", ext[2], ext[3], d[2], 0.5, key=f"c{kk}", disabled=not custom)
    lon_max = c2.number_input("Lon max", ext[2], ext[3], d[3], 0.5, key=f"d{kk}", disabled=not custom)

    if custom and countries:
        country = st.sidebar.selectbox("Country", ["All"] + countries)
    if country == "All" or not countries:
        country = None
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
    last_m = int(grid[grid.year == latest].month.max())
    qy, qm = qp.get("year", ""), qp.get("month", "")
    st.session_state.setdefault("sel_year", int(qy) if qy.isdigit() and int(qy) in set(years) else latest)
    st.session_state.setdefault("sel_month", int(qm) if qm.isdigit() and 1 <= int(qm) <= 12 else last_m)
    sel_year = st.sidebar.selectbox("Year", sorted(years, reverse=True), key="sel_year")
    sel_month = st.sidebar.selectbox("Month", range(1, 13), key="sel_month",
                                     format_func=lambda m: MONTH_NAMES[m - 1])
    yr = year_range_slider("Calendar year range", years)
    return preset, (lat_min, lat_max, lon_min, lon_max, country), method, scale, (bmode.lower(), base, window), (sel_year, sel_month), yr


# ---------- charts ----------
def calendar_heatmap(df, mode):
    if mode == "Hotspot count":
        z = df.pivot(index="year", columns="month", values="hotspot_count")
        scale, zmid, zmax, title = PALETTE["heat"], None, None, "Hotspots"
    else:
        z = df.pivot(index="year", columns="month", values="ratio_to_median")
        scale, zmid, zmax, title = PALETTE["div"], 1.0, 3.0, "x median"
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


def map_cells(grid, bbox, year, month, k, tscale=np.nan):
    lat0, lat1, lon0, lon1 = bbox
    c = grid[(grid.year == year) & (grid.month == month)
             & grid.cell_lat.between(lat0, lat1) & grid.cell_lon.between(lon0, lon1)].copy()
    # MODIS-equivalent counts (Terra-only months before Aqua are scaled to Terra+Aqua)
    c["val"] = c.viirs_raw / k if pd.notna(k) else c.modis_raw * (tscale if pd.notna(tscale) else 1.0)
    return c.reset_index(drop=True)


def region_map(c, bbox):
    lat0, lat1, lon0, lon1 = bbox
    fig = go.Figure(go.Scattergeo(
        lat=c.cell_lat, lon=c.cell_lon, mode="markers",
        marker=dict(size=np.clip(c.val * 1.2 + 4, 4, 40), color=c.val, colorscale=PALETTE["heat"], showscale=True,
                    colorbar=dict(title="Hotspots"), line=dict(width=0.3, color="black")),
        hovertemplate="%{lat}, %{lon}<br>%{marker.color:.1f} hotspots<extra></extra>"))
    fig.update_geos(lataxis_range=[lat0 - 1, lat1 + 1], lonaxis_range=[lon0 - 1, lon1 + 1],
                    showland=True, landcolor="#2b2b2b", showocean=True, oceancolor="#1a2a3a",
                    showcountries=True, countrycolor="#777")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), dragmode="select")
    return fig


def rank_of(full, year, month):
    """(rank, n_years): 1 = highest count for that calendar month in the record."""
    s = full[(full.month == month) & full.hotspot_count.notna()].set_index("year").hotspot_count
    if year not in s.index or len(s) < 8:
        return None
    return int(s.rank(ascending=False, method="min")[year]), len(s)


def fire_season(full, b0, b1, cover=0.8):
    """Fewest calendar months that hold >= `cover` of the typical yearly burning (baseline-years median)."""
    med = full[full.year.between(b0, b1)].groupby("month").hotspot_count.median()
    if med.sum() <= 0:
        return None
    share = (med / med.sum()).sort_values(ascending=False)
    n = int((share.cumsum() < cover).sum()) + 1
    return sorted(share.index[:n]), float(share.iloc[:n].sum())


def fmt_months(ms):
    names = [MONTH_NAMES[m - 1] for m in ms]
    return f"{names[0]}\u2013{names[-1]}" if ms == list(range(ms[0], ms[0] + len(ms))) else ", ".join(names)


def jump_to(year, month):
    st.session_state["sel_year"], st.session_state["sel_month"] = int(year), int(month)


def cell_anomaly(gview, full, bbox, year, month, b0, b1, min_count=5):
    """Per-cell harmonized count for (year, month) vs that cell's own median for the same calendar month."""
    lat0, lat1, lon0, lon1 = bbox
    c = gview[gview.cell_lat.between(lat0, lat1) & gview.cell_lon.between(lon0, lon1) & (gview.month == month)]
    c = c.merge(full[["year", "month", "k_used", "terra_scale", "viirs_era"]], on=["year", "month"], how="left")
    c["val"] = np.where(c.viirs_era.fillna(False), c.viirs_raw / c.k_used,
                        np.where(c.terra_scale.notna(), c.modis_terra * c.terra_scale, c.modis_raw))
    base = (c[c.year.between(b0, b1)].groupby(["cell_lat", "cell_lon"]).val.median().rename("base").reset_index())
    now = c[c.year == year][["cell_lat", "cell_lon", "val"]].merge(base, how="outer").fillna(0)
    now = now[(now.val >= min_count) | (now.base >= min_count)].copy()   # hide cells too quiet to judge
    now["log2_ratio"] = np.log2((now.val + 1) / (now.base + 1))         # +1 keeps 0-count cells finite
    return now


def anomaly_map(cells, bbox, cell):
    lat0, lat1, lon0, lon1 = bbox
    fig = go.Figure(go.Scattergeo(
        lat=cells.cell_lat, lon=cells.cell_lon, mode="markers",
        marker=dict(symbol="square", size=11, color=cells.log2_ratio.clip(-3, 3), colorscale=PALETTE["div"], cmid=0,
                    cmin=-3, cmax=3, line=dict(width=0.3, color="#222"),
                    colorbar=dict(title="vs cell median", tickvals=[-3, -1.58, 0, 1.58, 3],
                                  ticktext=["1/8x", "1/3x", "normal", "3x", "8x"])),
        customdata=np.c_[cells.val.round(1), cells.base.round(1)],
        hovertemplate="%{lat}, %{lon}<br>now %{customdata[0]} | typical %{customdata[1]}<extra></extra>"))
    fig.update_geos(lataxis_range=[lat0 - 1, lat1 + 1], lonaxis_range=[lon0 - 1, lon1 + 1], showland=True,
                    landcolor="#2b2b2b", showocean=True, oceancolor="#1a2a3a", showcountries=True, countrycolor="#777")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def k_stability_chart(df):
    """VIIRS/MODIS ratio per year (full VIIRS-era months): how constant is the harmonization factor?"""
    d = df[df.viirs_era & (df.modis_raw > 0)]
    g = d.groupby("year")[["viirs_raw", "modis_raw"]].sum()
    g = g[g.modis_raw >= 50]                      # tiny totals give meaningless ratios
    if len(g) < 3:
        return None
    k = g.viirs_raw / g.modis_raw
    fig = go.Figure(go.Bar(x=k.index, y=k.values, marker_color="#8e44ad", name="k per year"))
    fig.add_hline(y=d.viirs_raw.sum() / d.modis_raw.sum(), line_dash="dash", line_color="black",
                  annotation_text="overall k")
    fig.update_layout(height=300, yaxis_title="VIIRS / MODIS", xaxis_title="Year", margin=dict(t=20), showlegend=False)
    return fig


def season_tracker(full, year, b0, b1):
    """Cumulative harmonized hotspots Jan->month for `year` vs the baseline-years median and the worst complete year."""
    p = full[~full.get("partial", False)].pivot(index="year", columns="month", values="hotspot_count").reindex(columns=range(1, 13))
    if year not in p.index:
        return None
    last = int(p.loc[year].last_valid_index() or 0)
    if last == 0:
        return None
    cum = p.fillna(0).cumsum(axis=1)
    base = cum[cum.index.isin(range(b0, b1 + 1)) & p.notna().all(axis=1)]
    if len(base) < 3:
        return None
    complete = p.dropna().index.difference([year])
    worst = int(cum.loc[complete, 12].idxmax()) if len(complete) else None
    x = [MONTH_NAMES[m - 1] for m in range(1, 13)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=base.quantile(.75), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=base.quantile(.25), fill="tonexty", fillcolor="rgba(120,120,120,.2)",
                             line=dict(width=0), name="Baseline IQR", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=base.median(), name="Baseline median", line=dict(color="black", dash="dash")))
    if worst:
        fig.add_trace(go.Scatter(x=x, y=cum.loc[worst], name=f"Worst complete year ({worst})",
                                 line=dict(color="#8e44ad", dash="dot")))
    fig.add_trace(go.Scatter(x=x[:last], y=cum.loc[year].iloc[:last], name=str(year), mode="lines+markers",
                             line=dict(color="#d62728", width=3)))
    fig.update_layout(height=380, yaxis_title="Cumulative harmonized hotspots", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=20))
    now, norm = float(cum.loc[year, last]), float(base.median()[last])
    return fig, last, now, norm, worst, (float(cum.loc[worst, last]) if worst else None)


@st.cache_data(show_spinner="Comparing regions...")
def region_table(version, method, scale, bmode, b0, b1, window, year, month):
    """Selected month for every preset region: count, ratio to median, flag, rank among same months."""
    grid, _ = load_grid(version)
    countries = set(grid.country.unique()) if "country" in grid.columns else set()
    ext = (grid.cell_lat.min() - 1, grid.cell_lat.max() + 1, grid.cell_lon.min() - 1, grid.cell_lon.max() + 1)
    rows = []
    for name, (box, ctry) in PRESETS.items():
        if name == "Custom" or (ctry and countries and ctry not in countries):
            continue
        bb = box or ext
        m = aggregate_region(grid, *bb, ctry if countries else None)
        if m.empty:
            continue
        h = add_uncertainty(harmonize(m, method), 0.0, 0.0)
        f = analyze(h, bmode, b0, b1, window, scale)
        r = f[(f.year == year) & (f.month == month)]
        if r.empty or pd.isna(r.iloc[0].baseline_historical_median):
            continue
        r = r.iloc[0]
        rk = rank_of(f, year, month)
        rows.append({"Region": name, "Hotspots": round(r.hotspot_count), "Typical (median)": round(r.baseline_historical_median),
                     "x median": r.ratio_to_median, "Level": r.anomaly_flag.capitalize(),
                     "Rank for this month": f"#{rk[0]} of {rk[1]}" if rk else "n/a"})
    return pd.DataFrame(rows).sort_values("x median", ascending=False, na_position="last") if rows else pd.DataFrame()


# ---------- main ----------
def main():
    st.title("🔥 Harmonized Burning Activity Calendar")
    grid, is_mock = load_grid(_data_version())
    st.caption("Team Outerstellar · NASA Space Apps · MODIS (1 km, 2000→) + VIIRS (375 m, 2012→) harmonized "
               "to MODIS-equivalent counts · " + ("**MOCK DATA**" if is_mock else "**NASA FIRMS data**"))
    if is_mock:
        st.warning("Demo mode: showing simulated data because `data/grid_monthly_real.csv` was not found. "
                   "Add that file (made by `build_grid_from_firms.py` from FIRMS archive downloads) to use real detections.")

    last_y = int(grid.year.max())
    last_mo = int(grid[grid.year == last_y].month.max())
    src_countries = " + ".join(sorted(grid.country.unique())) if "country" in grid.columns else "all"
    st.caption(f"Data through **{MONTH_NAMES[last_mo - 1]} {last_y}** "
               f"(NASA FIRMS archive: MODIS C6.1 and Suomi-NPP VIIRS C2, {src_countries})." if not is_mock else "")
    preset, bbox, method, scale, (bmode, (b0, b1), window), (sel_year, sel_month), (y0, y1) = sidebar_controls(grid)
    st.sidebar.header("4. Display")
    if st.sidebar.checkbox("Colour-blind-safe palette", help="Swaps the red/blue ramps for cividis / purple-orange."):
        PALETTE.update(heat="cividis", div="PuOr_r")
    else:
        PALETTE.update(heat="YlOrRd", div="RdBu_r")
    st.query_params.update(preset=preset, method=method, year=str(sel_year), month=str(sel_month))
    bbox, country = bbox[:4], bbox[4]
    gview = grid if not country else grid[grid.country == country]
    monthly_raw = aggregate_region(grid, *bbox, country)
    if monthly_raw.empty:
        st.warning(f"No grid cells inside that bounding box (cells are {cell_size(grid):g}°). Enlarge the box.")
        return
    harmonized = harmonize(monthly_raw, method)
    med_err, p80_err = loyo_error(harmonized, method)
    err_ta = terra_scale_error(harmonized)
    err_ta80 = terra_scale_error(harmonized, q=0.8)
    levels = loyo_error_levels(harmonized, method)
    by_size = loyo_error_by_size(harmonized, method)
    nz = lambda x: 0.0 if pd.isna(x) else float(x)
    harmonized = add_uncertainty(harmonized, nz(med_err), nz(err_ta),
                                 err_robust=nz(p80_err), err_terra_robust=nz(err_ta80))
    has_cov = (not is_mock) and COVERAGE_CSV.exists()
    if has_cov:
        harmonized = attach_coverage(harmonized, pd.read_csv(COVERAGE_CSV))
    full = analyze(harmonized, bmode, b0, b1, window, scale)
    if has_cov:  # a flag in a month with missing satellite days is never "robust"
        full["robust"] = full.robust & ~full.data_gap
    base_desc = (f"the same month in the previous {window} years (needs at least 7)" if bmode == "rolling"
                 else f"the same month in {b0}-{b1}")
    # partial-month guard: never flag (a) the current UTC month, or (b) the newest month in the data when the
    # coverage check shows the satellite(s) reported on < 90% of its days (e.g. a top-up that stops mid-month)
    now = pd.Timestamp.now(tz="UTC")
    partial = (full.year == now.year) & (full.month == now.month)
    if has_cov:
        ym = full.year * 12 + full.month
        partial = partial | ((ym == ym.max()) & full.data_gap.fillna(False).astype(bool))
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
        pc = r.pctile if pd.notna(r.pctile) else np.nan
        if r.anomaly_flag == "extreme":
            st.error(f"**EXTREME burning activity** for {MONTH_NAMES[sel_month - 1]} {sel_year}"
                     + (" \u2013 the flag holds even with a pessimistic (80th-percentile) harmonization error." if r.robust else "."))
        elif r.anomaly_flag == "high":
            st.warning(f"**HIGH burning activity** for {MONTH_NAMES[sel_month - 1]} {sel_year}"
                       + (" \u2013 the flag holds even with a pessimistic (80th-percentile) harmonization error." if r.robust else "."))
        elif r.anomaly_flag == "low":
            st.info(f"**Unusually LOW burning activity** for {MONTH_NAMES[sel_month - 1]} {sel_year}.")
        elif r.anomaly_flag == "normal" and pd.notna(pc) and pc >= 85:
            st.warning(f"**ELEVATED burning activity**: higher than {pc:.0f}% of baseline years for this month. "
                       "Not a statistical outlier, because the baseline itself contains several very active years.")
        elif r.anomaly_flag == "normal":
            st.success(f"**NORMAL burning activity** for {MONTH_NAMES[sel_month - 1]} {sel_year}.")
    else:
        st.info("No data for the selected month.")

    if not row.empty:
        rk, season = rank_of(full, sel_year, sel_month), fire_season(full, b0, b1)
        if rk:
            st.markdown(f"**{MONTH_NAMES[sel_month - 1]} {sel_year} is the #{rk[0]} most active "
                        f"{MONTH_NAMES[sel_month - 1]} out of {rk[1]} years on record** for this region.")
        if season:
            st.caption(f"Typical fire season ({b0}-{b1}): **{fmt_months(season[0])}**, "
                       f"holding {season[1]:.0%} of a normal year's burning.")
    big = full[~full.partial].nlargest(4, "hotspot_count")
    if len(big):
        st.caption("Jump to the busiest months on record for this region:")
        cols = st.columns(len(big))
        for c, (_, b) in zip(cols, big.iterrows()):
            c.button(f"{MONTH_NAMES[int(b.month) - 1]} {int(b.year)} ({b.hotspot_count:,.0f})",
                     on_click=jump_to, args=(b.year, b.month), key=f"jump{int(b.year)}{int(b.month)}")

    st.subheader("Burning activity calendar")
    mode = st.radio("Colour by", ["Hotspot count", "Anomaly vs baseline (ratio to median)"], horizontal=True)
    cal_fig = calendar_heatmap(view, mode)
    st.plotly_chart(cal_fig, width="stretch")

    st.subheader(f"Is {sel_year} ahead of a normal season?")
    tr = season_tracker(full, sel_year, b0, b1)
    if tr:
        tfig, last_m_, now_, norm_, worst_, worst_v = tr
        st.plotly_chart(tfig, width="stretch")
        pct_ = (now_ / norm_ - 1) * 100 if norm_ > 0 else np.nan
        msg = (f"By the end of {MONTH_NAMES[last_m_ - 1]}, {sel_year} has accumulated **{now_:,.0f}** hotspots: "
               + (f"**{abs(pct_):.0f}% {'above' if pct_ >= 0 else 'below'}** the normal {b0}-{b1} total ({norm_:,.0f})" if pd.notna(pct_) else ""))
        if worst_v:
            msg += f"; the worst complete year, {worst_}, had {worst_v:,.0f} by the same point."
        st.caption(msg)
    else:
        st.caption("Not enough complete years in this region for a season tracker.")

    st.subheader(f"Which regions are unusual in {MONTH_NAMES[sel_month - 1]} {sel_year}?")
    rt = region_table(_data_version(), method, scale, bmode, b0, b1, window, sel_year, sel_month)
    if len(rt):
        st.dataframe(rt, width="stretch", hide_index=True)
        st.caption("Every preset region compared with its own history for this month (sorted by x median). "
                   "Overlapping regions (e.g. Sumatra and Indonesia all) are shown separately, not added up.")
    else:
        st.caption("No preset region has a baseline for this month.")

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
        size_txt = " · ".join(f"{k} hotspots: {v:.0%}" for k, v in by_size.items())
        st.caption("Validation over the months where both sensors report (leave-one-year-out). "
                   + (f"Error by month size: {size_txt} (quiet months are noisy). " if size_txt else "") +
                   f"Median error by time scale: {lv}; random counting noise averages out in longer totals."
                   + (f" Terra-only months before Aqua (Nov 2000-Jun 2002) are scaled to Terra+Aqua with "
                      f"≈{err_ta:.0%} error." if pd.notna(err_ta) else ""))
    else:
        st.caption("Not enough overlapping MODIS/VIIRS months in this region to validate the harmonization.")
    st.plotly_chart(harmonization_chart(view), width="stretch")
    kfig = k_stability_chart(harmonized)
    if kfig is not None:
        with st.expander("How stable is the cross-sensor factor k? (transparency check)"):
            st.plotly_chart(kfig, width="stretch")
            st.caption("If k were perfectly constant every bar would match the dashed line. The spread is why the "
                       "harmonization error band is shown, and why small regions should use the *global* factor.")

    if has_cov:
        gaps = harmonized[harmonized.data_gap][["year", "month", "sensor_source", "coverage"]]
        with st.expander(f"Data-quality notes: {len(gaps)} months in this region with missing satellite days"):
            st.caption("Share of calendar days on which the satellite(s) behind the month reported any detection "
                       "(whole-dataset check). Low values are likely outages or partial months, not real changes.")
            st.dataframe(gaps.sort_values("coverage"), width="stretch", hide_index=True)

    st.subheader("Most unusual months in this region")
    top = top_anomalies(full)
    if len(top):
        st.dataframe(top, width="stretch", hide_index=True)
    else:
        st.caption("No high anomalies found.")

    st.subheader(f"Map: {MONTH_NAMES[sel_month - 1]} {sel_year} intensity by {cell_size(grid):g}° cell")
    k_sel = row.iloc[0].k_used if not row.empty else np.nan
    t_sel = row.iloc[0].terra_scale if not row.empty else np.nan
    mt1, mt2 = st.tabs(["Where is it burning?", "Where is it unusual?"])
    mcells = map_cells(gview, bbox, sel_year, sel_month, k_sel, t_sel)
    ev = mt1.plotly_chart(region_map(mcells, bbox), width="stretch", on_select="rerun",
                          selection_mode=("box", "lasso"), key=f"map_{bbox}_{sel_year}_{sel_month}")
    mt1.caption("Tip: drag a box on this map to analyse that area as a custom region.")
    try:   # box/lasso select -> new custom region (guarded: a failure here must never break the app)
        pts = (ev.selection.points if ev and ev.selection else []) or []
        idx = [p.get("point_index", p.get("point_number")) for p in pts]
        sub = mcells.iloc[[i for i in idx if i is not None and 0 <= i < len(mcells)]]
        if len(sub):
            h = cell_size(grid) / 2
            new = (float(sub.cell_lat.min() - h), float(sub.cell_lat.max() + h),
                   float(sub.cell_lon.min() - h), float(sub.cell_lon.max() + h))
            if any(abs(a - b) > 1e-6 for a, b in zip(new, bbox)):
                st.session_state["pending_aoi"] = new
                st.rerun()
    except st.errors.StreamlitAPIException:
        raise
    except Exception:
        pass
    cells = cell_anomaly(gview, full, bbox, sel_year, sel_month, b0, b1)
    if len(cells):
        mt2.plotly_chart(anomaly_map(cells, bbox, cell_size(grid)), width="stretch")
        mt2.caption("Red = more burning than that same cell normally has in this month; blue = less. "
                    "Cells with fewer than 5 hotspots (now and typically) are hidden as too noisy.")
    else:
        mt2.caption("No cells with enough detections to compare.")

    st.subheader("Export")
    if not row.empty and pd.notna(row.iloc[0].ratio_to_median):
        st.download_button("📄 One-page situation brief (Markdown)", ("# Burning activity brief\n\n" + brief(row.iloc[0], preset, peak)
                           + f"\n\n_Harmonization error ≈ ±{med_err:.0%} (leave-one-year-out). Source: NASA FIRMS MODIS + VIIRS._").encode(),
                           file_name=f"brief_{sel_year}-{sel_month:02d}.md")
    out = view.assign(country=country or "All", lat_min=bbox[0], lat_max=bbox[1], lon_min=bbox[2], lon_max=bbox[3],
                      baseline_start=b0, baseline_end=b1, harmonization=method)
    st.download_button("🖼️ Download the calendar as an interactive HTML chart", cal_fig.to_html(include_plotlyjs="cdn"),
                       file_name=f"burning_calendar_{y0}-{y1}.html", mime="text/html")
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
            "*Robust* means the flag still holds after shifting the count by the 80th-percentile harmonization "
            f"error (≈{p80_err:.0%} here; the shaded band in the charts shows the typical, median error).\n"
            "- **Consistency:** MODIS = Terra + Aqua (Aqua starts July 2002; earlier Terra-only months are scaled "
            "to Terra+Aqua per calendar month), VIIRS = Suomi-NPP only, low-confidence detections removed. "
            "VIIRS (13:30 overpass) agrees better with Terra+Aqua than with Terra alone (10:30).\n"
            "- **Limits:** counts are detections, not burned area; cloud/smoke and overpass time affect detection; "
            "a hotspot count ratio (*k*) assumes similar fire sizes between sensors.\n"
            "- **Orbit drift:** MODIS on Terra and Aqua can drift from its original overpass time in the 2020s, which "
            "may shift detection rates; treat comparisons of recent years with early years with extra care.\n"
            "- **Sensors:** VIIRS is Suomi-NPP only, kept for one consistent sensor; adding NOAA-20/21 would raise "
            "later-year counts. *k* is not perfectly constant (see the stability check above).\n"
            "- **Coverage:** FIRMS files were cut by country polygon (Indonesia, Malaysia), so Brunei, Timor-Leste and "
            "other neighbours are not counted; Borneo totals are slightly low. "
            f"Data end {MONTH_NAMES[last_mo - 1]} {last_y}; near-real-time top-up is future work.")


if __name__ == "__main__":
    main()
