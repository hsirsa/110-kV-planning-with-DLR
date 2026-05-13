"""ERA5 weather downloader for DLR studies.

Primary source: Copernicus CDS (requires a ~/.cdsapirc key file).
Automatic fallback: Open-Meteo ERA5 archive (no credentials needed).

Install optional dependencies:  uv sync --extra weather-cds
"""

from __future__ import annotations

import calendar
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim

try:
    import cdsapi
    import xarray as xr
except Exception:
    cdsapi = None
    xr = None


YEAR = 2023
DEFAULT_OUT_DIR = Path("weather_tampere_dlr")

PLACES = [
    "Teisko, Tampere, Finland",
    "Kämmenniemi, Tampere, Finland",
    "Aitolahti, Tampere, Finland",
    "Terälahti, Tampere, Finland",
]

FALLBACK_COORDS: dict[str, tuple[float, float]] = {
    "Teisko, Tampere, Finland": (61.70, 23.90),
    "Kämmenniemi, Tampere, Finland": (61.63, 23.84),
    "Aitolahti, Tampere, Finland": (61.55, 23.89),
    "Terälahti, Tampere, Finland": (61.58, 24.03),
}


def _safe_name(s: str) -> str:
    return s.replace(",", "").replace(" ", "_").replace("ä", "a").replace("ö", "o").replace("å", "a")


def _geocode_places(places: list[str]) -> dict[str, tuple[float, float]]:
    geolocator = Nominatim(user_agent="dlr-weather-tampere")
    out: dict[str, tuple[float, float]] = {}
    for place in places:
        loc = None
        try:
            loc = geolocator.geocode(place, timeout=20)
        except Exception:
            pass
        out[place] = (float(loc.latitude), float(loc.longitude)) if loc else FALLBACK_COORDS[place]
    return out


def _uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u**2 + v**2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


def _from_cds_era5(client, place: str, lat: float, lon: float, year: int, out_dir: Path) -> pd.DataFrame:
    if xr is None:
        raise RuntimeError("xarray is not available.")
    frames = []
    sname = _safe_name(place)
    for month in range(1, 13):
        ndays = calendar.monthrange(year, month)[1]
        req = {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_solar_radiation_downwards",
            ],
            "year": [f"{year:04d}"],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, ndays + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [lat, lon, lat, lon],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        nc_file = out_dir / f"{sname}_{year}_{month:02d}.nc"
        for attempt in range(1, 4):
            try:
                client.retrieve("reanalysis-era5-single-levels", req, str(nc_file))
                break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"CDS failed for {place}, month {month:02d}: {e}") from e
                time.sleep(15 * attempt)
        ds = xr.open_dataset(nc_file)
        wind_speed, wind_dir = _uv_to_speed_dir(ds["u10"].to_series().values, ds["v10"].to_series().values)
        frames.append(
            pd.DataFrame(
                {
                    "time_utc": pd.to_datetime(ds["t2m"].to_series().index, utc=True),
                    "ambient_temp_c": ds["t2m"].to_series().values - 273.15,
                    "wind_speed_mps": wind_speed,
                    "wind_angle_deg": wind_dir,
                    "solar_wm2": np.maximum(ds["ssrd"].to_series().values / 3600.0, 0.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _from_open_meteo(lat: float, lon: float, year: int) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,wind_u_component_10m,wind_v_component_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90)
    r.raise_for_status()
    h = r.json().get("hourly", {})
    if not h:
        raise RuntimeError("Open-Meteo response has no 'hourly' data.")
    wind_speed = h.get("wind_speed_10m", h.get("windspeed_10m"))
    wind_dir = h.get("wind_direction_10m", h.get("winddirection_10m"))
    if wind_speed is None or all(v is None for v in wind_speed):
        u = np.array(h["wind_u_component_10m"], dtype=float)
        v = np.array(h["wind_v_component_10m"], dtype=float)
        wind_speed, wind_dir = _uv_to_speed_dir(u, v)
        wind_speed, wind_dir = wind_speed.tolist(), wind_dir.tolist()
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(h["time"], utc=True),
            "ambient_temp_c": pd.to_numeric(h["temperature_2m"], errors="coerce"),
            "wind_speed_mps": pd.to_numeric(wind_speed, errors="coerce"),
            "wind_angle_deg": pd.to_numeric(wind_dir, errors="coerce"),
            "solar_wm2": pd.to_numeric(h["shortwave_radiation"], errors="coerce").clip(lower=0.0),
        }
    )
    cols = ["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]
    df[cols] = df[cols].interpolate(limit_direction="both")
    return df


def _validate(df: pd.DataFrame, place: str):
    required = ["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{place}: missing columns {missing}")
    null_frac = df[required].isna().mean()
    for c in required:
        if null_frac[c] > 0.01:
            raise ValueError(f"{place}: column '{c}' has >1% nulls ({null_frac[c]:.3%})")


def run_weather_download(year: int = YEAR, out_dir: Path = DEFAULT_OUT_DIR, places: list[str] = PLACES):
    """Download ERA5 weather data for the given places and year."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    coords = _geocode_places(places)
    cds_client = None
    if cdsapi is not None:
        try:
            cds_client = cdsapi.Client(timeout=300)
        except Exception:
            pass
    print("Primary source:", "CDS ERA5" if cds_client else "Open-Meteo ERA5 (no CDS config)")

    for place, (lat, lon) in coords.items():
        print(f"\nProcessing {place} ({lat:.4f}, {lon:.4f})")
        source_used = "CDS"
        try:
            if cds_client is None:
                raise RuntimeError("CDS client unavailable")
            df = _from_cds_era5(cds_client, place, lat, lon, year, out_dir)
        except Exception as e:
            print(f"  CDS failed: {e} — switching to Open-Meteo fallback")
            source_used = "OPEN_METEO"
            df = _from_open_meteo(lat, lon, year)

        start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
        end = pd.Timestamp(f"{year}-12-31T23:00:00Z")
        df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()
        df["location_name"] = place
        df["lat"] = lat
        df["lon"] = lon
        df["source"] = source_used

        _validate(df, place)
        out_csv = out_dir / f"{_safe_name(place)}_{year}_dlr_weather.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv} ({len(df)} rows)")

    print(f"\nDone. Output folder: {out_dir.resolve()}")
