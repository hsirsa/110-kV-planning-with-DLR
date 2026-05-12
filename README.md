# 110 kV Network Planning with Dynamic Line Rating

Power flow study of the SimBench HV1 110 kV mixed network with IEEE 738 Dynamic Line Rating (DLR), electric boiler loads, and seasonal analysis. Weather data is sourced from the Aitolahti station near Tampere, Finland.

## What it does

- **Seasonal power flow** — runs four independent time-series power flows (winter, spring, summer, autumn) using 15-minute resolution profiles from the SimBench dataset.
- **Dynamic Line Rating** — applies IEEE 738 thermal ampacity to every line in the network using real ambient temperature, wind speed, wind angle, and solar irradiance, then compares DLR utilisation against the static rating.
- **Electric boilers** — adds two flexible electric boiler loads (50 MW at HV1 Bus 67, 100 MW at HV1 Bus 35) driven by time-varying price-based profiles before the power flow runs.
- **Subnet focus** — produces detailed voltage, current, DLR, and generation-mix analysis for a focused 8-bus subnet inside the HV1 network.
- **Cross-season comparison** — aggregates seasonal peak currents, loading percentages, and DLR benefit into side-by-side summary tables and plots.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd 110-kV-planning-with-DLR

# Install all dependencies and the dlr package
make install
```

To also enable ERA5 weather downloads via the Copernicus CDS API:

```bash
make install-weather
```

## Before running studies

The seasonal power flow requires a preprocessed weather file at:

```
weather_tampere_dlr/Aitolahti_Tampere_Finland_2023_dlr_weather_15min.csv
```

Download and process it with:

```bash
make weather
```

This fetches hourly ERA5 data for 2023 from the Copernicus CDS API (requires a `~/.cdsapirc` credentials file) and falls back automatically to the Open-Meteo public archive if CDS is not configured. The output is interpolated to 15-minute intervals and saved per location.

You also need the electric boiler load profiles in the project root:

```
electric_boiler_50MW_newprice.csv
electric_boiler_100MW_newprice.csv
```

## Running studies

### Seasonal DLR study (main workflow)

Run all four seasons in sequence:

```bash
make run
# or
uv run dlr run
```

Run a single season:

```bash
make run-winter
uv run dlr run --season spring
uv run dlr run --season summer --output-root my_results/
```

Results are written to `results_1year_seasons/<season>/` with the following structure:

```
results_1year_seasons/
├── winter/
│   ├── subnet_focus/
│   │   ├── subnet_bus_vm_pu.csv
│   │   ├── subnet_line_dlr_ka.csv
│   │   ├── subnet_line_loading_comparison.csv
│   │   ├── subnet_bus_generation_mix.csv
│   │   ├── subnet_timeseries.png
│   │   ├── subnet_topology_results.png
│   │   └── ...
│   └── hv1_network/
│       ├── hv1_line_peak_summary.csv
│       ├── hv1_line_dlr_ka.csv
│       └── hv1_line_peak_overview.png
├── spring/ ...
├── summer/ ...
├── autumn/ ...
├── seasonal_subnet_line_peak_comparison.csv
├── seasonal_hv1_line_peak_comparison.csv
└── seasonal_subnet_dlr_summary.csv
```

Rebuild the cross-season summary tables from existing results without re-running the power flow:

```bash
make summary
```

### Baseline power flow studies

```bash
make run-day    # 1-day study (96 × 15 min steps) → results_1day/
make run-year   # Full 1-year study, no seasons, no DLR → results_1year/
```

## Network topology

Export topology CSVs (bus summary, line summary, parallel lines, connections) and a network diagram:

```bash
make topology   # geographic coordinates → HV1_export/
make diagram    # BFS tree layout, PNG + PDF → hv1_diagram_like_image/
```

## Code quality

```bash
make lint       # ruff check (read-only)
make format     # ruff check --fix + ruff format
```

## Cleaning up

```bash
make clean-results   # delete all generated output directories
make clean           # delete the virtual environment
```

## Project structure

```
src/dlr/
├── cli.py          # Click entry point — all commands live here
├── config.py       # Grid code, constants, paths, season definitions
├── network.py      # Bus lookup, coordinate helpers
├── boiler.py       # Electric boiler profile loading and injection
├── powerflow.py    # robust_runpp, OutputWriter setup
├── export.py       # Named-result CSV export
├── subnet.py       # Subnet topology and conductor property builders
├── dlr_calc.py     # IEEE 738 ampacity (scalar + vectorised NumPy batch)
├── analysis.py     # Time-series aggregation, DLR benefit, voltage summaries
├── plots.py        # All matplotlib figures
├── seasons.py      # Seasonal orchestration: run, export, cross-season tables
├── study.py        # Baseline 1-day and 1-year power flow runners
├── topology.py     # Full-network topology export and diagrams
└── weather.py      # ERA5 / Open-Meteo weather download
```

## CLI reference

```
dlr run                          Run all four seasonal DLR studies
dlr run --season <name>          Run one season (winter|spring|summer|autumn)
dlr run --season winter \
    --output-root path/          Override output directory

dlr summary                      Rebuild cross-season tables from saved results
dlr run-day                      1-day baseline power flow
dlr run-year                     1-year baseline power flow (no DLR)

dlr topology                     Export topology CSVs + geo-coordinate diagram
dlr diagram                      Export BFS tree-layout diagram (PNG + PDF)
dlr weather                      Download ERA5 weather data
dlr weather --year 2022 \
    --output-dir my_weather/     Override year and output directory
```

Pass `--help` to any command for full options:

```bash
uv run dlr run --help
```
