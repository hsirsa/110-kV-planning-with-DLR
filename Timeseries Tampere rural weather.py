# dlr_weather_tampere_full.py
# Full version: ERA5 from CDS (preferred) + automatic Open-Meteo fallback
# with robust wind/solar handling for DLR studies.
#
# Compatible with Python 3.12
# pip install cdsapi xarray netcdf4 pandas numpy geopy requests

from __future__ import annotations
import calendar
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim

# Optional imports for CDS route
try:
    import cdsapi
    import xarray as xr
except Exception:
    cdsapi = None
    xr = None


# ---------------------------
# User settings
# ---------------------------
YEAR = 2023
OUT_DIR = Path("weather_tampere_dlr")
OUT_DIR.mkdir(exist_ok=True)

PLACES = [
    "Teisko, Tampere, Finland",
    "Kämmenniemi, Tampere, Finland",
    "Aitolahti, Tampere, Finland",
    "Terälahti, Tampere, Finland",
]

# Used if geocoding fails
FALLBACK_COORDS = {
    "Teisko, Tampere, Finland": (61.70, 23.90),
    "Kämmenniemi, Tampere, Finland": (61.63, 23.84),
    "Aitolahti, Tampere, Finland": (61.55, 23.89),
    "Terälahti, Tampere, Finland": (61.58, 24.03),
}


# ---------------------------
# Utilities
# ---------------------------
def safe_name(s: str) -> str:
    return (
        s.replace(",", "")
        .replace(" ", "_")
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("å", "a")
    )


def geocode_places(places: list[str]) -> dict[str, tuple[float, float]]:
    geolocator = Nominatim(user_agent="dlr-weather-tampere")
    out: dict[str, tuple[float, float]] = {}
    for place in places:
        loc = None
        try:
            loc = geolocator.geocode(place, timeout=20)
        except Exception:
            pass
        if loc is not None:
            out[place] = (float(loc.latitude), float(loc.longitude))
        else:
            out[place] = FALLBACK_COORDS[place]
    return out


def try_create_cds_client():
    if cdsapi is None:
        return None
    try:
        return cdsapi.Client(timeout=300)
    except Exception:
        return None


def era5_month_request(year: int, month: int, lat: float, lon: float) -> dict:
    ndays = calendar.monthrange(year, month)[1]
    return {
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
        "area": [lat, lon, lat, lon],  # [N, W, S, E]
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u ** 2 + v ** 2)
    # Meteorological direction: degrees from which wind blows
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


def from_cds_era5(client, place: str, lat: float, lon: float) -> pd.DataFrame:
    if xr is None:
        raise RuntimeError("xarray is not available for CDS processing.")

    frames = []
    sname = safe_name(place)

    for month in range(1, 13):
        req = era5_month_request(YEAR, month, lat, lon)
        nc_file = OUT_DIR / f"{sname}_{YEAR}_{month:02d}.nc"

        for attempt in range(1, 4):
            try:
                client.retrieve("reanalysis-era5-single-levels", req, str(nc_file))
                break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"CDS failed for {place}, month {month:02d}: {e}") from e
                time.sleep(15 * attempt)

        ds = xr.open_dataset(nc_file)
        t2m = ds["t2m"].to_series()
        u10 = ds["u10"].to_series()
        v10 = ds["v10"].to_series()
        ssrd = ds["ssrd"].to_series()  # J/m^2 over hour

        wind_speed, wind_dir = uv_to_speed_dir(u10.values, v10.values)

        frame = pd.DataFrame({
            "time_utc": pd.to_datetime(t2m.index, utc=True),
            "ambient_temp_c": t2m.values - 273.15,
            "wind_speed_mps": wind_speed,
            "wind_angle_deg": wind_dir,
            "solar_wm2": np.maximum(ssrd.values / 3600.0, 0.0),  # J/m² per hour -> W/m²
        })
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def from_open_meteo_era5(lat: float, lon: float) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{YEAR}-01-01",
        "end_date": f"{YEAR}-12-31",
        "hourly": ",".join([
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_u_component_10m",
            "wind_v_component_10m",
            "shortwave_radiation",
        ]),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        # Intentionally not forcing one model; reduces risk of partial null fields.
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError("Open-Meteo response has no 'hourly' data.")

    h = data["hourly"]

    temp = h.get("temperature_2m")
    if temp is None:
        raise RuntimeError("Open-Meteo returned no temperature_2m.")

    # Accept both naming styles
    wind_speed = h.get("wind_speed_10m", h.get("windspeed_10m"))
    wind_dir = h.get("wind_direction_10m", h.get("winddirection_10m"))
    u10 = h.get("wind_u_component_10m")
    v10 = h.get("wind_v_component_10m")

    # Build wind from u/v if speed missing or fully null
    if wind_speed is None or all(v is None for v in wind_speed):
        if u10 is None or v10 is None:
            raise RuntimeError("Open-Meteo returned no usable wind data.")
        u = np.array(u10, dtype=float)
        v = np.array(v10, dtype=float)
        ws, wd = uv_to_speed_dir(u, v)
        wind_speed = ws.tolist()
        wind_dir = wd.tolist()

    solar = h.get("shortwave_radiation")
    if solar is None:
        raise RuntimeError("Open-Meteo returned no shortwave_radiation.")

    df = pd.DataFrame({
        "time_utc": pd.to_datetime(h["time"], utc=True),
        "ambient_temp_c": pd.to_numeric(temp, errors="coerce"),
        "wind_speed_mps": pd.to_numeric(wind_speed, errors="coerce"),
        "wind_angle_deg": pd.to_numeric(wind_dir, errors="coerce"),
        "solar_wm2": pd.to_numeric(solar, errors="coerce"),
    })

    df["solar_wm2"] = df["solar_wm2"].clip(lower=0.0)
    df[["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]] = (
        df[["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]]
        .interpolate(limit_direction="both")
    )

    return df


def validate_df(df: pd.DataFrame, place: str):
    required = ["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{place}: missing columns {miss}")

    null_frac = df[required].isna().mean()
    print(f"[{place}] null fractions:\n{null_frac.to_string()}")

    # Hard fail if too many nulls remain
    for c in required:
        if null_frac[c] > 0.01:
            raise ValueError(f"{place}: column '{c}' has >1% nulls ({null_frac[c]:.3%})")


def main():
    coords = geocode_places(PLACES)
    cds_client = try_create_cds_client()
    print("Primary source:", "CDS ERA5" if cds_client else "Open-Meteo ERA5 fallback (no CDS config)")

    for place, (lat, lon) in coords.items():
        print(f"\nProcessing {place} ({lat:.4f}, {lon:.4f})")
        source_used = "CDS"

        try:
            if cds_client is not None:
                df = from_cds_era5(cds_client, place, lat, lon)
            else:
                raise RuntimeError("CDS client unavailable")
        except Exception as e:
            print(f"CDS failed: {e}")
            print("Switching to Open-Meteo ERA5 fallback...")
            source_used = "OPEN_METEO"
            df = from_open_meteo_era5(lat, lon)

        # Keep only target year exactly
        start = pd.Timestamp(f"{YEAR}-01-01T00:00:00Z")
        end = pd.Timestamp(f"{YEAR}-12-31T23:00:00Z")
        df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()

        # Add metadata
        df["location_name"] = place
        df["lat"] = lat
        df["lon"] = lon
        df["source"] = source_used

        validate_df(df, place)

        out_csv = OUT_DIR / f"{safe_name(place)}_{YEAR}_dlr_weather.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} ({len(df)} rows)")

    print(f"\nDone. Output folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
