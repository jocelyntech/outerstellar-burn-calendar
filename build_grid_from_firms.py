"""Convert REAL NASA FIRMS archive CSVs into the gridded monthly file the app reads.

Usage:
  python build_grid_from_firms.py --modis modis_*.csv --viirs viirs_*.csv
Output: data/grid_monthly_real.csv  (the app uses it automatically instead of the mock data)

Consistency choices (important for a harmonized record):
  * MODIS : Terra only ('T' / 'Terra') by default -> Terra runs since 2000; Aqua only from mid-2002 would add a step.
  * VIIRS : Suomi-NPP only ('N') by default -> runs since 2012; NOAA-20/21 would add extra detections later.
  * Low-confidence detections are dropped (MODIS confidence < 30, VIIRS 'l'/'low').
Use --modis-sat all / --viirs-sat all to keep every satellite.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CELL = 2.0
OUT = Path("data/grid_monthly_real.csv")


def load(files, kind, sat):
    frames = [pd.read_csv(f, dtype={"satellite": str, "confidence": str}) for f in files]
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    if sat != "all":
        s_ = d.satellite.str.strip().str.upper()
        # MODIS archive says 'Terra'/'Aqua', NRT says 'T'/'A' -> match first letter.
        # VIIRS: exact match ('N' = Suomi-NPP, 'N20', 'N21'); archive may also say 'NPP'/'SNPP'.
        if kind == "modis":
            d = d[s_.str[0] == sat.upper()[0]]
        else:
            d = d[s_.isin({sat.upper()} | ({"N", "NPP", "SNPP", "SUOMI-NPP"} if sat.upper() == "N" else set()))]
    conf = d.confidence.str.lower()
    if kind == "modis":
        d = d[pd.to_numeric(conf, errors="coerce") >= 30]
    else:
        d = d[~conf.isin(["l", "low"])]
    dt = pd.to_datetime(d.acq_date)
    return pd.DataFrame({"year": dt.dt.year, "month": dt.dt.month,
                         "cell_lat": np.floor(d.latitude / CELL) * CELL + CELL / 2,
                         "cell_lon": np.floor(d.longitude / CELL) * CELL + CELL / 2})


def count(df, name):
    return df.groupby(["year", "month", "cell_lat", "cell_lon"]).size().rename(name).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", nargs="+", required=True)
    ap.add_argument("--viirs", nargs="+", required=True)
    ap.add_argument("--modis-sat", default="T")
    ap.add_argument("--viirs-sat", default="N")
    a = ap.parse_args()
    m = count(load(a.modis, "modis", a.modis_sat), "modis_raw")
    v = count(load(a.viirs, "viirs", a.viirs_sat), "viirs_raw")
    g = m.merge(v, how="outer", on=["year", "month", "cell_lat", "cell_lon"]).fillna(0)
    g[["modis_raw", "viirs_raw"]] = g[["modis_raw", "viirs_raw"]].astype(int)
    OUT.parent.mkdir(exist_ok=True)
    g.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(g):,} cell-months, {g.year.min()}-{g.year.max()}, "
          f"MODIS={g.modis_raw.sum():,} VIIRS={g.viirs_raw.sum():,}")


if __name__ == "__main__":
    main()
