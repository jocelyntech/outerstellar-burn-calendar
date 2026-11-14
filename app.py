"""Team Outerstellar - Harmonized Burning Activity Calendar (prototype).  Run: streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from analytics import MONTH_NAMES, add_baseline_and_anomaly, aggregate_region, estimate_factor
from generate_mock_data import GRID_CSV, main as build_data

st.set_page_config(page_title="Harmonized Burning Activity Calendar", page_icon="🔥", layout="wide")

PRESETS = {  # name: (lat_min, lat_max, lon_min, lon_max)
    "Whole Southeast Asia (mock domain)": (-10.0, 24.0, 93.0, 127.0),
    "Mainland SE Asia (TH/LA/MM/VN/KH)": (8.0, 24.0, 93.0, 110.0),
    "Sumatra": (-6.0, 6.0, 94.0, 109.0),
    "Kalimantan (Borneo)": (-4.5, 4.5, 108.0, 119.0),
    "Custom": None,
}
FLAG_COLORS = {"high": "#d62728", "low": "#1f77b4", "normal": "#9aa0a6"}


# ---------- data ----------
@st.cache_data(show_spinner="Loading gridded hotspot data...")
def load_grid() -> pd.DataFrame:
    if not GRID_CSV.exists():
        build_data()  # writes data/mock_grid_monthly.csv
    return pd.read_csv(GRID_CSV)


# ---------- sidebar ----------
def sidebar_controls(years):
    st.sidebar.header("1. Region of interest")
    preset = st.sidebar.selectbox("Preset", list(PRESETS))
    d = PRESETS[preset] or PRESETS["Whole Southeast Asia (mock domain)"]
    custom = preset == "Custom"
    c1, c2 = st.sidebar.columns(2)
    k = preset  # new widget keys per preset so values reset when the preset changes
    lat_min = c1.number_input("Lat min", -11.0, 25.0, d[0], 0.5, key=f"a{k}", disabled=not custom)
    lat_max = c2.number_input("Lat max", -11.0, 25.0, d[1], 0.5, key=f"b{k}", disabled=not custom)
    lon_min = c1.number_input("Lon min", 92.0, 128.0, d[2], 0.5, key=f"c{k}", disabled=not custom)
    lon_max = c2.number_input("Lon max", 92.0, 128.0, d[3], 0.5, key=f"d{k}", disabled=not custom)

    st.sidebar.header("2. Baseline & selection")
    b0, b1 = st.sidebar.slider("Baseline years (historical median)", int(years.min()), int(years.max()),
                               (2001, 2020))
    sel_year = st.sidebar.selectbox("Year", sorted(years, reverse=True))
    sel_month = st.sidebar.selectbox("Month", range(1, 13), index=9, format_func=lambda m: MONTH_NAMES[m - 1])
    y0, y1 = st.sidebar.slider("Calendar year range", int(years.min()), int(years.max()),
                               (int(years.min()), int(years.max())))
    return (lat_min, lat_max, lon_min, lon_max), (b0, b1), (sel_year, sel_month), (y0, y1)


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
    fig.update_layout(title=f"{MONTH_NAMES[month - 1]} hotspots by year (selected year outlined by flag colour)",
                      xaxis_title="Year", yaxis_title="Harmonized hotspots", height=420,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def year_vs_baseline(df, year):
    y = df[df.year == year].sort_values("month")
    base = df.drop_duplicates("month").sort_values("month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=base.baseline_q3, line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=base.baseline_q1, fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(120,120,120,.2)", name="Baseline IQR", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=base.baseline_historical_median, name="Baseline median",
                             line=dict(color="black", dash="dash")))
    fig.add_trace(go.Scatter(x=[MONTH_NAMES[m - 1] for m in y.month], y=y.hotspot_count, name=str(year),
                             mode="lines+markers", line=dict(color="#d62728"),
                             marker=dict(color=[FLAG_COLORS[f] for f in y.anomaly_flag], size=9)))
    fig.update_layout(title=f"{year} vs historical baseline", yaxis_title="Harmonized hotspots", height=420,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def harmonization_chart(df):
    """Top: raw MODIS vs raw VIIRS (different scales). Bottom: harmonized single series."""
    x = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    v = df.viirs_raw.where(df.viirs_raw > 0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Raw counts: the two sensors disagree on scale",
                                        "Harmonized (MODIS-equivalent): one consistent series"))
    fig.add_trace(go.Scatter(x=x, y=df.modis_raw, name="MODIS raw (1 km)", line=dict(color="#e67e22")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=v, name="VIIRS raw (375 m)", line=dict(color="#8e44ad")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df.hotspot_count, name="Harmonized", line=dict(color="#c0392b")), 2, 1)
    fig.add_vline(x=pd.Timestamp("2012-01-01"), line_dash="dot", line_color="gray")
    fig.update_layout(height=560, legend=dict(orientation="h", y=-0.12), margin=dict(t=60))
    fig.update_yaxes(title_text="Hotspots", row=1, col=1)
    fig.update_yaxes(title_text="Hotspots", row=2, col=1)
    return fig


def region_map(grid, bbox, year, month):
    lat0, lat1, lon0, lon1 = bbox
    c = grid[(grid.year == year) & (grid.month == month)
             & grid.cell_lat.between(lat0, lat1) & grid.cell_lon.between(lon0, lon1)]
    fig = go.Figure(go.Scattergeo(
        lat=c.cell_lat, lon=c.cell_lon, mode="markers",
        marker=dict(size=np.clip(c.hotspot_count * 1.2 + 4, 4, 40), color=c.hotspot_count,
                    colorscale="YlOrRd", showscale=True, colorbar=dict(title="Hotspots"),
                    line=dict(width=0.3, color="black")),
        hovertemplate="%{lat}, %{lon}<br>%{marker.color:.1f} hotspots<extra></extra>"))
    fig.update_geos(lataxis_range=[lat0 - 1, lat1 + 1], lonaxis_range=[lon0 - 1, lon1 + 1],
                    showland=True, landcolor="#2b2b2b", showocean=True, oceancolor="#1a2a3a",
                    showcountries=True, countrycolor="#777")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    return fig


# ---------- main ----------
def main():
    st.title("🔥 Harmonized Burning Activity Calendar")
    st.caption("Team Outerstellar · NASA Space Apps · MODIS (1 km, 2000→) + VIIRS (375 m, 2012→) "
               "harmonized to MODIS-equivalent counts · **MOCK DATA**")

    grid = load_grid()
    bbox, (b0, b1), (sel_year, sel_month), (y0, y1) = sidebar_controls(grid.year.drop_duplicates())

    monthly = aggregate_region(grid, *bbox)
    if monthly.empty:
        st.warning("No grid cells inside that bounding box (cells are 2°). Enlarge the box.")
        return
    full = add_baseline_and_anomaly(monthly, b0, b1)
    view = full[full.year.between(y0, y1)]

    row = full[(full.year == sel_year) & (full.month == sel_month)]
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4 = st.columns([1, 1, 1, 1.4])
        m1.metric(f"{MONTH_NAMES[sel_month - 1]} {sel_year}", f"{r.hotspot_count:,.0f}")
        m2.metric("Baseline median", f"{r.baseline_historical_median:,.0f}")
        m3.metric("vs median", f"{r.ratio_to_median}x" if pd.notna(r.ratio_to_median) else "n/a")
        m4.metric("Anomaly", r.anomaly_flag.capitalize())
        st.caption(f"Sensor source for this month: {r.sensor_source}")
    else:
        st.info("No data for the selected month (the record starts Nov 2000).")

    st.subheader("Burning activity calendar")
    mode = st.radio("Colour by", ["Hotspot count", "Anomaly vs baseline (ratio to median)"], horizontal=True)
    st.plotly_chart(calendar_heatmap(view, mode), width="stretch")

    st.subheader("Selected period vs historical baseline")
    t1, t2 = st.tabs(["Selected month across years", "Selected year across months"])
    t1.plotly_chart(month_across_years(view, sel_month, sel_year), width="stretch")
    t2.plotly_chart(year_vs_baseline(full, sel_year), width="stretch")

    st.subheader("Why harmonize? MODIS vs VIIRS")
    k = estimate_factor(monthly)
    if k:
        st.caption(f"Over the overlap period (2012→) VIIRS reports about **{k:.1f}x** more detections than MODIS "
                   f"in this region (finer 375 m pixels). Dividing VIIRS by k removes the artificial jump in 2012.")
    st.plotly_chart(harmonization_chart(view), width="stretch")

    st.subheader(f"Map: {MONTH_NAMES[sel_month - 1]} {sel_year} hotspot intensity by 2° cell")
    st.plotly_chart(region_map(grid, bbox, sel_year, sel_month), width="stretch")

    st.subheader("Export")
    out = view.assign(lat_min=bbox[0], lat_max=bbox[1], lon_min=bbox[2], lon_max=bbox[3],
                      baseline_start=b0, baseline_end=b1)
    st.download_button("⬇️ Download filtered results (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"burning_calendar_{y0}-{y1}.csv", mime="text/csv")
    with st.expander("Preview table"):
        st.dataframe(out, width="stretch")
    with st.expander("Harmonization method"):
        st.markdown("- **2000–2011:** MODIS counts as-is.\n"
                    "- **2012→:** VIIRS counts ÷ *k*, where *k* = Σ VIIRS / Σ MODIS over the overlap period "
                    "(~5.8 in the mock data; ~5.8 in the NRT sample files too).\n"
                    "- **Baseline:** per-calendar-month median and IQR over the chosen baseline years; "
                    "a month is flagged *high/low* beyond 1.5×IQR from Q3/Q1.")


if __name__ == "__main__":
    main()
