# 🔥 Outerstellar Burn Calendar

### Harmonized Burning Activity Calendar for Indonesia and Malaysia, 2000 → 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Data](https://img.shields.io/badge/Data-NASA%20FIRMS-0B3D91)
![Hackathon](https://img.shields.io/badge/NASA%20Space%20Apps-2026-E03C31)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

**Outerstellar Burn Calendar** is a Streamlit web app that merges two decades of NASA satellite fire detections into **one consistent monthly timeline** (November 2000 to October 2026). Pick a region, pick a month, and see at a glance whether burning is normal, unusually high or unusually low compared with history.

Built by **Team Outerstellar** for the **2026 NASA Space Apps Challenge**, challenge *"Harmonization of MODIS and VIIRS Hot Spots"*.

> 🔗 **Live app:** _add your Streamlit Community Cloud link here_

---

## Why this exists

Fires and peatland burning in Southeast Asia cause haze that harms health, ecosystems and the climate. Satellites have tracked active fires for more than 20 years, but the record is split across two sensors that cannot be compared directly:

| | MODIS (Terra, Aqua) | VIIRS (Suomi-NPP) |
|---|---|---|
| Resolution | ~1 km | 375 m |
| Record | Nov 2000 →, Aqua from Jul 2002 | 20 Jan 2012 → |
| Raw counts | Fewer detections | About 4.5× more detections for the same fires |

As monitoring moves from MODIS to VIIRS, a raw count that "jumps" in 2012 says more about the sensor than about the fires. This project converts VIIRS to **MODIS-equivalent detections**, so the sensor of origin no longer matters and the real patterns, trends and anomalies stand out.

---

## ✨ Key Features

### Data processing
- **Builds a monthly 1° grid** from raw NASA FIRMS archive and near-real-time CSVs.
- **Removes duplicate detections** across all files (same place, date, time and satellite), which handles overlaps along country borders.
- **Filters low-confidence detections** (MODIS confidence < 30, VIIRS "low").
- **Keeps Terra and Aqua separate**, so the Terra-only period before Aqua launched can be bridged correctly.
- **Writes a coverage report** (`coverage.csv`) showing which satellite-months have missing days.

### Analytics
- **MODIS ↔ VIIRS harmonization** with a choice of a *global* or *seasonal* conversion factor *k*.
- **Aqua bridging:** Terra-only months (Nov 2000 – Jun 2002) are scaled up to Terra+Aqua equivalents.
- **Anomaly detection against a baseline:** *fixed* years or a *rolling* window, on a raw or log scale. Months are tiered as `normal`, `high`, `extreme` or `low` (Tukey fences), with a percentile rank.
- **Honest uncertainty:** leave-one-year-out validation gives monthly, quarterly and annual error, plus an uncertainty band on every harmonized month. A flag only counts as **robust** if it survives a pessimistic (80th-percentile) harmonization error and the month has no missing satellite days.
- **Trend and season tools:** annual totals with a linear trend, typical fire season, and a season tracker ("is this year ahead of a normal season?").

### App and visualization
- **Burning activity calendar:** a year × month heatmap, coloured by hotspot count or by ratio to the historical median.
- **Region presets:** Entire dataset, Malaysia, Indonesia, Peninsular Malaysia, Sabah and Sarawak, Sumatra, Kalimantan, Java, Papua, plus a **Custom** box. Presets outside the loaded data are hidden automatically.
- **Interactive maps:** *Where is it burning?* and *Where is it unusual?* per 1° cell. Drag a box or lasso on the map to analyse that area as a custom region.
- **Region comparison:** every preset compared with its own history for the selected month.
- **Transparency tools:** sensor-agreement statistics, a stability check of *k*, and a data-quality table of months with satellite outages.
- **Accessibility:** a colour-blind-safe palette toggle (cividis / purple-orange).
- **Exports:** filtered results as **CSV**, the calendar as an **interactive HTML chart**, and a **one-page situation brief** (Markdown).
- **Shareable views:** the preset, method, year and month are kept in the page URL.
- **Demo mode:** if no real data is present, the app generates simulated data and shows a clear *MOCK DATA* banner.

### Dataset at a glance
| | |
|---|---|
| Period | Nov 2000 → Oct 2026 (312 months) |
| Area | Indonesia and Malaysia, 416 one-degree cells |
| Grid size | 75,234 cell-month records |
| MODIS detections kept | 1,681,353 (Terra 563,506 + Aqua 1,117,847) |
| VIIRS detections kept | 4,110,069 (Suomi-NPP) |

---

## 📁 Project Structure

```
outerstellar-burn-calendar/
├── app.py                    # Streamlit app (UI, charts, maps, exports)
├── burn_analytics.py         # Harmonization, validation, baselines, anomalies
├── build_grid_from_firms.py  # Raw NASA FIRMS CSVs  ->  monthly 1° grid
├── generate_mock_data.py     # Simulated data for demos and testing
├── requirements.txt          # Python dependencies
└── data/
    ├── grid_monthly_real.csv # Real grid (made by build_grid_from_firms.py)
    └── coverage.csv          # Per-satellite data coverage (same script)
```

| File | What it does |
|---|---|
| **`app.py`** | The dashboard. Loads `data/grid_monthly_real.csv` (or builds mock data if it is missing), applies the sidebar choices, and renders the calendar, trend charts, maps, comparison table and downloads. |
| **`burn_analytics.py`** | The analytics engine shared by the app and the mock-data script: region aggregation (`aggregate_region`), Aqua bridging (`bridge_aqua`), MODIS/VIIRS harmonization (`harmonize`), validation (`loyo_error`, `loyo_error_levels`, `terra_scale_error`), uncertainty bands, baselines and anomaly tiers (`analyze`), trends, coverage checks and the plain-language brief. |
| **`build_grid_from_firms.py`** | Command-line tool. Reads FIRMS archive and NRT CSVs, removes duplicates across all files, drops low-confidence detections, bins into 1° cells and writes `data/grid_monthly_real.csv` and `data/coverage.csv`. |
| **`generate_mock_data.py`** | Creates simulated gridded data (seasonality, El Niño-style boosts, a synthetic VIIRS/MODIS ratio) so the app can be demoed without downloading anything. Writes `data/mock_grid_monthly.csv` and `data/mock_harmonized_monthly.csv`. |
| **`requirements.txt`** | `streamlit`, `pandas`, `numpy`, `plotly`. |

### How the harmonization works

All numbers are in **MODIS-equivalent detections** (Terra + Aqua).

1. **Nov 2000 – Jun 2002:** only Terra exists, so Terra counts are scaled up to Terra+Aqua using a per-calendar-month factor measured in the Aqua era.
2. **Jul 2002 – Jan 2012:** MODIS Terra + Aqua as measured.
3. **Feb 2012 onward:** VIIRS ÷ *k*, where *k* = Σ VIIRS / Σ MODIS over the overlap period (*global*), or per calendar month (*seasonal*). January 2012 stays on MODIS because VIIRS only started on the 20th.

---

## 🛠️ Prerequisites & Installation

**Requirements:** Python 3.10 or newer, and Git.

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3. Install the dependencies**
```bash
pip install -r requirements.txt
```

**4. Check the install**
```bash
python -c "import streamlit, pandas, numpy, plotly; print('OK')"
```

---

## 🚀 Usage

Run every command from the repository root (the scripts use the relative `data/` folder).

### Option A: try the app with simulated data (no downloads)

```bash
python generate_mock_data.py
streamlit run app.py
```

`generate_mock_data.py` writes the mock files into `data/`. The app also builds them automatically on first launch if no real data is found. A *MOCK DATA* banner is shown so simulated numbers are never mistaken for real ones.

### Option B: build the real grid from NASA FIRMS data

**1. Download the data.** On the [NASA FIRMS Download page](https://firms.modaps.eosdis.nasa.gov/download/), request archive CSVs for your countries, for both products: **MODIS C6.1** and **VIIRS S-NPP (375 m)**. Include the near-real-time (NRT) files if you want the latest months. Unzip them into a folder such as `raw/`.

> Raw FIRMS files are large (hundreds of MB). Keep them out of Git.

**2. Build the grid.** Label each file with its country using `Country=path`. FIRMS downloads are cut by country polygon, and labelling lets the app filter by country. Pass **all** files in **one** command so duplicates are removed across every file:

```bash
python build_grid_from_firms.py \
  --modis Malaysia=raw/fire_archive_M-C61_846191.csv Malaysia=raw/fire_nrt_M-C61_846191.csv \
          Indonesia=raw/fire_archive_M-C61_846960.csv Indonesia=raw/fire_nrt_M-C61_846960.csv \
  --viirs Malaysia=raw/fire_archive_SV-C2_846192.csv Malaysia=raw/fire_nrt_SV-C2_846192.csv \
          Indonesia=raw/fire_archive_SV-C2_846969.csv Indonesia=raw/fire_nrt_SV-C2_846969.csv
```
*(On Windows, put the command on one line.)*

Replace the file names with your own. This writes:
- `data/grid_monthly_real.csv` (columns: `year, month, cell_lat, cell_lon, country, modis_terra, modis_aqua, modis_raw, viirs_raw`)
- `data/coverage.csv` (share of days each satellite reported, per month)

| Option | Default | Meaning |
|---|---|---|
| `--modis-sat` | `all` | Restrict MODIS to `T` (Terra) or `A` (Aqua). |
| `--viirs-sat` | `N` | Suomi-NPP only, for one consistent sensor. Use `all` to include NOAA-20/21. |
| `--cell` | `1.0` | Grid cell size in degrees. |

### Launch the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually <http://localhost:8501>). If `data/grid_monthly_real.csv` exists, the app uses it automatically and refreshes whenever the file changes.

**Typical workflow in the app:**
1. Choose a **region** (preset, custom box, or drag a box on the map).
2. Choose the **harmonization method** (global or seasonal *k*) and the **baseline** (fixed or rolling).
3. Pick a **year and month** to see its status, rank and brief.
4. Explore the calendar, season tracker, region comparison and maps.
5. **Export** a CSV, an HTML chart or a one-page brief.

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub, including `data/grid_monthly_real.csv` and `data/coverage.csv` (about 3 MB, well below GitHub's 25 MB browser-upload limit).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app, select the repository and set the main file to `app.py`.

---

## ⚠️ Limitations

- **Hotspot counts are not burned area** or fire size. The result is an index of burning activity.
- **Cloud, thick smoke and overpass time** affect what satellites can see.
- **The conversion factor *k* is not perfectly constant.** That is why every harmonized month carries an uncertainty band. Use the *global* factor for small regions.
- **The first 20 months (Nov 2000 – Jun 2002) are estimates** scaled up from Terra alone, so they are less certain.
- **MODIS orbit drift** since about 2021 may shift detection rates, so compare recent years with early years with care.
- **Country-cut downloads:** Brunei, Timor-Leste and other neighbours are not counted, so Borneo totals are slightly low.
- **Satellite outages** undercount some months; these are listed in the app's data-quality notes.
- The most recent months come from near-real-time data and may be revised.
- This tool supports awareness and does **not** replace on-the-ground monitoring.

---

## 🤝 Contributing

Contributions are welcome. This section is a placeholder; adapt it to your team's workflow.

1. Fork the repository and create a branch: `git checkout -b feature/my-improvement`
2. Make your changes and test locally with `streamlit run app.py`.
3. Commit with a clear message and push your branch.
4. Open a pull request describing what changed and why.

Please open an issue first for large changes, and keep the data pipeline (`build_grid_from_firms.py`) and the analytics (`burn_analytics.py`) backward compatible with the grid file format.

## 📄 License

_License to be decided._ Add a `LICENSE` file (for example MIT or Apache-2.0) and update the badge at the top of this README.

---

## 🙏 Acknowledgements

- **NASA FIRMS / Earthdata** for the MODIS Collection 6.1 and VIIRS (Suomi-NPP) Collection 2 active-fire data.
- **NASA Space Apps Challenge 2026**, challenge *"Harmonization of MODIS and VIIRS Hot Spots"*.
- Built by **Team Outerstellar**.
