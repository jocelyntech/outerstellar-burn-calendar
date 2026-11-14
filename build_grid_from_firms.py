"""Convert REAL NASA FIRMS archive CSVs into the gridded monthly file the app reads.

Usage:
  python build_grid_from_firms.py --modis modis_*.csv --viirs viirs_*.csv
  Label each file with its country using Label=path (FIRMS downloads are cut by country polygon):
  python build_grid_from_firms.py --modis Malaysia=m1.csv Malaysia=m2.csv Indonesia=m3.csv --viirs Malaysia=v1.csv Indonesia=v2.csv
Output: data/grid_monthly_real.csv  (the app uses it automatically instead of the mock data)
Duplicate detections (same place, date, time, satellite) are removed across ALL files; the first file wins.

Consistency choices (important for a harmonized record):
  * MODIS : Terra and Aqua are counted separately (columns modis_terra, modis_aqua; modis_raw = their sum).
            Aqua only exists from July 2002, so the app scales the Terra-only months before that to
            Terra+Aqua-equivalent. VIIRS agrees much better with Terra+Aqua than with Terra alone.
  * VIIRS : Suomi-NPP only ('N') by default -> runs since 2012; NOAA-20/21 would add extra detections later.
  * Low-confidence detections are dropped (MODIS confidence < 30, VIIRS 'l'/'low').
Use --modis-sat T / --viirs-sat all to restrict or widen the satellites.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CELL = 1.0   # grid cell size in degrees (override with --cell)
OUT = Path("data/grid_monthly_real.csv")


def split_label(arg):
    """'Malaysia=path/to/file.csv' -> ('Malaysia', path); a plain path -> ('All', path)."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        if label and "/" not in label and "\\" not in label and ":" not in label:
            return label, path
    return "All", arg


def load(files, kind, sat, cell=CELL, cov_sink=None):
    need = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence"]  # skip heavy columns
    frames = []
    for arg in files:
        label, path = split_label(arg)
        f = pd.read_csv(path, usecols=need, dtype={"satellite": "category", "confidence": str,
                                                   "latitude": "float32", "longitude": "float32"})
        frames.append(f.assign(country=label))
    d = pd.concat(frames, ignore_index=True)
    d["satellite"] = d["satellite"].astype(str)
    before = len(d)
    d = d.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    if before != len(d):
        print(f"  {kind}: removed {before - len(d):,} duplicate detections ({before:,} -> {len(d):,})")
    if sat != "all":
        s_ = d.satellite.str.strip().str.upper()
        # MODIS archive says 'Terra'/'Aqua', NRT says 'T'/'A' -> match first letter.
        # VIIRS: exact match ('N' = Suomi-NPP, 'N20', 'N21'); archive may also say 'NPP'/'SNPP'.
        if kind == "modis":
            d = d[s_.str[0] == sat.upper()[0]]
        else:
            d = d[s_.isin({sat.upper()} | ({"N", "NPP", "SNPP", "SUOMI-NPP"} if sat.upper() == "N" else set()))]
    if cov_sink is not None:  # which days each satellite reported anything (before the confidence filter)
        sensor = d.satellite.str.strip().str.upper().str[0].map({"T": "terra", "A": "aqua"}) if kind == "modis" \
            else pd.Series("viirs", index=d.index)
        cov_sink.append(pd.DataFrame({"sensor": sensor, "date": pd.to_datetime(d.acq_date)}).dropna().drop_duplicates())
    conf = d.confidence.str.lower()
    if kind == "modis":
        d = d[pd.to_numeric(conf, errors="coerce") >= 30]
    else:
        d = d[~conf.isin(["l", "low"])]
    dt = pd.to_datetime(d.acq_date)
    return pd.DataFrame({"year": dt.dt.year, "month": dt.dt.month, "country": d.country, "sat": d.satellite.str.strip().str.upper().str[0],
                         "cell_lat": np.floor(d.latitude / cell) * cell + cell / 2,
                         "cell_lon": np.floor(d.longitude / cell) * cell + cell / 2})


KEYS = ["year", "month", "cell_lat", "cell_lon", "country"]


def count(df, name):
    return df.groupby(KEYS).size().rename(name).reset_index()


def count_modis(df):
    """Per-cell monthly MODIS counts, Terra and Aqua kept apart."""
    c = (df.groupby(KEYS + ["sat"]).size()
           .unstack("sat", fill_value=0).reindex(columns=["T", "A"], fill_value=0)
           .rename(columns={"T": "modis_terra", "A": "modis_aqua"}).reset_index())
    c["modis_raw"] = c.modis_terra + c.modis_aqua
    return c


def coverage_table(parts):
    """Share of calendar days per sensor-month with at least one detection anywhere in the data.
    A first/last partial month (e.g. VIIRS Jan 2012, starting on the 20th) shows up as low coverage."""
    d = pd.concat(parts).drop_duplicates()
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    g = d.groupby(["sensor", "year", "month"]).date.nunique().rename("days_with_data").reset_index()
    g["days_in_month"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1)).dt.days_in_month
    g["coverage"] = (g.days_with_data / g.days_in_month).round(3)
    return g.sort_values(["sensor", "year", "month"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", nargs="+", required=True)
    ap.add_argument("--viirs", nargs="+", required=True)
    ap.add_argument("--modis-sat", default="all")
    ap.add_argument("--viirs-sat", default="N")
    ap.add_argument("--cell", type=float, default=CELL, help="grid cell size in degrees")
    a = ap.parse_args()
    cov = []
    m = count_modis(load(a.modis, "modis", a.modis_sat, a.cell, cov))
    v = count(load(a.viirs, "viirs", a.viirs_sat, a.cell, cov), "viirs_raw")
    g = m.merge(v, how="outer", on=KEYS).fillna(0)
    num = ["modis_terra", "modis_aqua", "modis_raw", "viirs_raw"]
    g[num] = g[num].astype(int)
    OUT.parent.mkdir(exist_ok=True)
    g.to_csv(OUT, index=False)
    cv = coverage_table(cov)
    cv.to_csv(OUT.parent / "coverage.csv", index=False)
    bad = cv[cv.coverage < 0.9]
    print(f"Wrote {OUT.parent / 'coverage.csv'}: {len(bad)} sensor-months with < 90% of days reporting")
    print(f"Wrote {OUT}: {len(g):,} cell-months, {g.year.min()}-{g.year.max()}, "
          f"MODIS={g.modis_raw.sum():,} (Terra {g.modis_terra.sum():,}, Aqua {g.modis_aqua.sum():,}) VIIRS={g.viirs_raw.sum():,}")


if __name__ == "__main__":
    main()
