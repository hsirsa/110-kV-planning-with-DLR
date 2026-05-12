import math

import numpy as np
import pandas as pd

from .config import (
    AITOLAHTI_WEATHER_CSV, CONDUCTOR_TEMP_C, DEFAULT_ABSORPTIVITY, DEFAULT_ALPHA,
    DEFAULT_EMISSIVITY, SIGMA, TIME_STEP_HOURS,
)


def load_aitolahti_weather(time_steps):
    if not AITOLAHTI_WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"Aitolahti weather file not found: {AITOLAHTI_WEATHER_CSV}. Run dlr_weather_tampere_full.py first."
        )
    weather_df = pd.read_csv(AITOLAHTI_WEATHER_CSV).reset_index(drop=True)
    weather_df["time_step"] = weather_df.index.astype(int)
    weather_df = weather_df[weather_df["time_step"].isin(time_steps)].copy()
    required = ["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]
    missing = [col for col in required if col not in weather_df.columns]
    if missing:
        raise ValueError(f"Aitolahti weather file is missing columns: {missing}")
    weather_df["hour_of_day"] = weather_df["time_step"] * TIME_STEP_HOURS
    return weather_df[["time_step", "time_utc", "hour_of_day", "ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]]


def ieee738_ampacity_ka(line_row, weather_row):
    conductor_temp_c = float(line_row.get("dlr_conductor_temp_c", CONDUCTOR_TEMP_C))
    ambient_temp_c = float(weather_row["ambient_temp_c"])
    delta_t = max(conductor_temp_c - ambient_temp_c, 0.1)
    diameter_m = max(float(line_row.get("diameter_m_est", 0.03)), 1e-4)
    wind_speed_mps = max(float(weather_row["wind_speed_mps"]), 0.01)
    wind_angle_deg = float(weather_row["wind_angle_deg"])
    line_azimuth_deg = float(line_row.get("line_azimuth_deg", 90.0))
    attack_angle_deg = abs(((wind_angle_deg - line_azimuth_deg + 90.0) % 180.0) - 90.0)
    attack_angle_rad = math.radians(attack_angle_deg)
    angle_factor = 1.194 - math.cos(attack_angle_rad) + 0.194 * math.cos(2.0 * attack_angle_rad) + 0.368 * math.sin(2.0 * attack_angle_rad)
    angle_factor = max(angle_factor, 0.2)

    film_temp_k = ((conductor_temp_c + ambient_temp_c) / 2.0) + 273.15
    air_density = 1.225 * 288.15 / film_temp_k
    air_viscosity = 1.458e-6 * film_temp_k ** 1.5 / (film_temp_k + 110.4)
    air_thermal_conductivity = 0.02424 + 7.477e-5 * (film_temp_k - 273.15)
    prandtl = 1006.0 * air_viscosity / air_thermal_conductivity
    reynolds = air_density * wind_speed_mps * diameter_m / air_viscosity

    q_natural = 3.645 * math.sqrt(max(air_density, 1e-9)) * diameter_m ** 0.75 * delta_t ** 1.25
    nusselt = 0.3 + (
        (0.62 * math.sqrt(max(reynolds, 0.0)) * prandtl ** (1.0 / 3.0))
        / ((1.0 + (0.4 / prandtl) ** (2.0 / 3.0)) ** 0.25)
    ) * ((1.0 + (reynolds / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0))
    heat_transfer_coeff = angle_factor * nusselt * air_thermal_conductivity / diameter_m
    q_forced = heat_transfer_coeff * math.pi * diameter_m * delta_t
    q_convective = max(q_natural, q_forced)

    emissivity = float(line_row.get("emissivity", DEFAULT_EMISSIVITY))
    absorptivity = float(line_row.get("absorptivity", DEFAULT_ABSORPTIVITY))
    conductor_temp_k = conductor_temp_c + 273.15
    ambient_temp_k = ambient_temp_c + 273.15
    q_radiative = math.pi * diameter_m * emissivity * SIGMA * (conductor_temp_k ** 4 - ambient_temp_k ** 4)
    q_solar = absorptivity * float(weather_row["solar_wm2"]) * diameter_m

    resistance_20 = float(line_row.get("resistance_20c_ohm_per_km", 0.0))
    alpha = float(line_row.get("alpha_per_c", DEFAULT_ALPHA))
    parallel_count = max(int(line_row.get("parallel_count", 1)), 1)
    resistance_operating = resistance_20 * (1.0 + alpha * (conductor_temp_c - 20.0)) / 1000.0 / parallel_count
    heat_balance = q_convective + q_radiative - q_solar
    if heat_balance <= 0.0 or resistance_operating <= 0.0:
        return 0.0
    return math.sqrt(heat_balance / resistance_operating) / 1000.0


def ieee738_ampacity_ka_batch(df):
    """Vectorized IEEE 738 DLR computation over a DataFrame of merged line+weather rows."""
    conductor_temp_c = df["dlr_conductor_temp_c"].to_numpy(dtype=float)
    ambient_temp_c = df["ambient_temp_c"].to_numpy(dtype=float)
    delta_t = np.maximum(conductor_temp_c - ambient_temp_c, 0.1)
    diameter_m = np.maximum(df["diameter_m_est"].to_numpy(dtype=float), 1e-4)
    wind_speed_mps = np.maximum(df["wind_speed_mps"].to_numpy(dtype=float), 0.01)
    wind_angle_deg = df["wind_angle_deg"].to_numpy(dtype=float)
    line_azimuth_deg = df["line_azimuth_deg"].to_numpy(dtype=float)

    attack_angle_rad = np.radians(np.abs(((wind_angle_deg - line_azimuth_deg + 90.0) % 180.0) - 90.0))
    angle_factor = np.maximum(
        1.194 - np.cos(attack_angle_rad) + 0.194 * np.cos(2.0 * attack_angle_rad) + 0.368 * np.sin(2.0 * attack_angle_rad),
        0.2,
    )

    film_temp_k = (conductor_temp_c + ambient_temp_c) / 2.0 + 273.15
    air_density = 1.225 * 288.15 / film_temp_k
    air_viscosity = 1.458e-6 * film_temp_k ** 1.5 / (film_temp_k + 110.4)
    air_thermal_conductivity = 0.02424 + 7.477e-5 * (film_temp_k - 273.15)
    prandtl = 1006.0 * air_viscosity / air_thermal_conductivity
    reynolds = air_density * wind_speed_mps * diameter_m / air_viscosity

    q_natural = 3.645 * np.sqrt(np.maximum(air_density, 1e-9)) * diameter_m ** 0.75 * delta_t ** 1.25
    nusselt = 0.3 + (
        (0.62 * np.sqrt(np.maximum(reynolds, 0.0)) * prandtl ** (1.0 / 3.0))
        / ((1.0 + (0.4 / prandtl) ** (2.0 / 3.0)) ** 0.25)
    ) * (1.0 + (reynolds / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    q_forced = angle_factor * nusselt * air_thermal_conductivity / diameter_m * np.pi * diameter_m * delta_t
    q_convective = np.maximum(q_natural, q_forced)

    emissivity = df["emissivity"].to_numpy(dtype=float)
    absorptivity = df["absorptivity"].to_numpy(dtype=float)
    conductor_temp_k = conductor_temp_c + 273.15
    ambient_temp_k = ambient_temp_c + 273.15
    q_radiative = np.pi * diameter_m * emissivity * SIGMA * (conductor_temp_k ** 4 - ambient_temp_k ** 4)
    q_solar = absorptivity * df["solar_wm2"].to_numpy(dtype=float) * diameter_m

    resistance_20 = df["resistance_20c_ohm_per_km"].to_numpy(dtype=float)
    alpha = df["alpha_per_c"].to_numpy(dtype=float)
    parallel_count = np.maximum(df["parallel_count"].to_numpy(dtype=int), 1).astype(float)
    resistance_operating = resistance_20 * (1.0 + alpha * (conductor_temp_c - 20.0)) / 1000.0 / parallel_count

    heat_balance = q_convective + q_radiative - q_solar
    valid = (heat_balance > 0.0) & (resistance_operating > 0.0)
    safe_r = np.where(valid, resistance_operating, 1.0)
    safe_hb = np.where(valid, heat_balance, 0.0)
    return np.where(valid, np.sqrt(safe_hb / safe_r) / 1000.0, 0.0)


def build_subnet_dlr_timeseries(line_summary, line_i_ka, weather_df):
    if line_summary.empty or line_i_ka.empty or weather_df.empty:
        return pd.DataFrame()

    weather_cols = ["hour_of_day", "ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]
    merged = line_i_ka.merge(
        weather_df.set_index("time_step")[weather_cols],
        left_on="time_step",
        right_index=True,
        how="inner",
    )
    prop_cols = [
        "line_index", "diameter_m_est", "line_azimuth_deg", "emissivity", "absorptivity",
        "resistance_20c_ohm_per_km", "alpha_per_c", "parallel_count", "dlr_conductor_temp_c",
    ]
    available_props = [c for c in prop_cols if c in line_summary.columns]
    merged = merged.merge(line_summary[available_props], on="line_index", how="left")

    if merged.empty:
        return pd.DataFrame()

    merged = merged.copy()
    merged["dlr_ka"] = ieee738_ampacity_ka_batch(merged)
    merged["actual_i_ka"] = merged["i_ka"]
    merged["dlr_utilization_percent"] = np.where(
        merged["dlr_ka"] > 0.0, 100.0 * merged["i_ka"] / merged["dlr_ka"], np.nan
    )
    out_cols = [
        "time_step", "hour_of_day", "line_index", "name",
        "ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2",
        "actual_i_ka", "dlr_ka", "dlr_utilization_percent",
    ]
    return merged[[c for c in out_cols if c in merged.columns]].reset_index(drop=True)


def build_loading_comparison_timeseries(line_loading, dlr_df):
    if line_loading.empty or dlr_df.empty:
        return pd.DataFrame()
    no_dlr = line_loading.copy()
    no_dlr["hour_of_day"] = no_dlr["time_step"] * TIME_STEP_HOURS
    no_dlr = no_dlr.rename(columns={"loading_percent": "loading_percent_without_dlr"})
    with_dlr = dlr_df[["time_step", "line_index", "name", "hour_of_day", "dlr_utilization_percent"]].copy()
    with_dlr = with_dlr.rename(columns={"dlr_utilization_percent": "loading_percent_with_dlr"})
    merged = no_dlr.merge(with_dlr, on=["time_step", "line_index", "name", "hour_of_day"], how="left")
    merged["dlr_benefit_percent_points"] = merged["loading_percent_without_dlr"] - merged["loading_percent_with_dlr"]
    return merged
