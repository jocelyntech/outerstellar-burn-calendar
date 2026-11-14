"""Generate MOCK harmonized MODIS + VIIRS monthly hotspot data (Nov 2000 -> Oct 2026).

Outputs (in ./data):
  mock_grid_monthly.csv       gridded (2-degree cells) monthly counts -> used by the app for any bbox
  mock_harmonized_monthly.csv region-wide series: year, month, sensor_source, hotspot_count,
                              baseline_historical_median, anomaly_flag

Harmonization (MODIS-equivalent units):
  * Nov 2000 - Dec 2011 : MODIS counts as-is
  * Jan 2012 onward     : VIIRS counts / k, where k = sum(VIIRS)/sum(MODIS) over the overlap period
Run:  python generate_mock_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from burn_analytics import add_baseline_and_anomaly

DATA_DIR = Path("data")
GRID_CSV = DATA_DIR / "mock_grid_monthly.csv"
REGION_CSV = DATA_DIR / "mock_harmonized_monthly.csv"

SEED = 42
FIRST_MONTH, LAST_MONTH = "2000-11", "2026-10"   # Terra MODIS record starts Nov 2000
VIIRS_START = "2012-01"                           # VIIRS 375 m record starts 2012
LAT_RANGE, LON_RANGE, CELL = (-10, 24), (93, 127), 2.0
VIIRS_PER_MODIS = 5.8      # ~ratio seen in the NRT sample files (16,972 VIIRS vs 2,912 MODIS / 7 days)
EL_NINO_BOOST = {2002: .4, 2004: .2, 2006: .6, 2009: .2, 2015: 1.0, 2019: .7, 2023: .6}


def _circ_gauss(month, peak, width):
    d = np.minimum(abs(month - peak), 12 - abs(month - peak))
    return np.exp(-0.5 * (d / width) ** 2)


def seasonality(lat, month):
    """Mainland SE Asia peaks ~Mar; southern Indonesia peaks ~Sep; blended in between."""
    w_north = np.clip((lat + 2) / 10, 0, 1)
    return w_north * _circ_gauss(month, 3, 1.3) + (1 - w_north) * _circ_gauss(month, 9, 1.5) + 0.03


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = np.arange(LAT_RANGE[0] + CELL / 2, LAT_RANGE[1], CELL)
    lons = np.arange(LON_RANGE[0] + CELL / 2, LON_RANGE[1], CELL)
    cells = pd.DataFrame([(a, o) for a in lats for o in lons], columns=["cell_lat", "cell_lon"])
    land = rng.random(len(cells)) > 0.4                           # crude land/sea mask
    cells["base"] = rng.lognormal(1.8, 1.1, len(cells)) * np.where(land, 1.0, 0.03)

    months = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
    t = pd.DataFrame({"year": months.year, "month": months.month})
    t["boost"] = t.year.map(EL_NINO_BOOST).fillna(0.0)
    t["trend"] = 1 + 0.01 * (t.year - 2000)
    t["has_viirs"] = months >= pd.Period(VIIRS_START, "M")

    g = cells.merge(t, how="cross")
    seas = seasonality(g.cell_lat.to_numpy(), g.month.to_numpy())
    lam = g.base * seas * (1 + g.boost * seas) * g.trend * rng.lognormal(0, 0.25, len(g))
    g["modis_raw"] = rng.poisson(lam)                                         # 1 km sensor
    g["viirs_raw"] = np.where(g.has_viirs, rng.poisson(lam * VIIRS_PER_MODIS), 0)  # 375 m sensor

    # estimate cross-sensor factor on the overlap, then convert VIIRS to MODIS-equivalent
    ov = g[g.has_viirs]
    k = ov.viirs_raw.sum() / ov.modis_raw.sum()
    g["sensor_source"] = np.where(g.has_viirs, "VIIRS (harmonized)", "MODIS")
    g["hotspot_count"] = np.where(g.has_viirs, g.viirs_raw / k, g.modis_raw).round(2)
    print(f"Estimated VIIRS/MODIS factor k = {k:.2f}")
    return g[["year", "month", "cell_lat", "cell_lon", "sensor_source",
              "modis_raw", "viirs_raw", "hotspot_count"]]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    grid = generate()
    grid.to_csv(GRID_CSV, index=False)
    region = (grid.groupby(["year", "month"], as_index=False)
                  .agg(sensor_source=("sensor_source", "first"), hotspot_count=("hotspot_count", "sum"))
                  .round({"hotspot_count": 1}))
    region = add_baseline_and_anomaly(region, 2001, 2020)
    region[["year", "month", "sensor_source", "hotspot_count",
            "baseline_historical_median", "anomaly_flag"]].to_csv(REGION_CSV, index=False)
    print(f"Wrote {GRID_CSV} ({len(grid):,} rows) and {REGION_CSV} ({len(region)} rows)")


if __name__ == "__main__":
    main()
