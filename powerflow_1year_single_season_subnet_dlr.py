import os
import warnings
from pathlib import Path

import pandas as pd
import simbench as sb
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries import run_timeseries

import powerflow_1year_seasons as base


SEASON_OUTPUT_ROOTS = {
    "winter": "results_1year_winter_subnet_dlr_only",
    "spring": "results_1year_spring_subnet_dlr_only",
    "summer": "results_1year_summer_subnet_dlr_only",
    "autumn": "results_1year_autumn_subnet_dlr_only",
}


def _normalize_boiler_profile(profile_csv):
    profile_df = pd.read_csv(profile_csv)
    if "p_mw" not in profile_df.columns:
        raise ValueError(f"Electric boiler profile must contain 'p_mw': {profile_csv}")
    if "time_step" not in profile_df.columns and "datetime_utc" not in profile_df.columns:
        raise ValueError(
            f"Electric boiler profile must contain either 'time_step' or 'datetime_utc': {profile_csv}"
        )

    normalized = profile_df.copy()
    if "datetime_utc" in normalized.columns:
        normalized["datetime_utc"] = pd.to_datetime(normalized["datetime_utc"], errors="coerce", utc=True)
        if normalized["datetime_utc"].isna().any():
            raise ValueError(f"Electric boiler profile contains invalid UTC timestamps: {profile_csv}")
        base_time = pd.Timestamp("2023-01-01 00:00:00Z")
        step_delta = pd.to_timedelta(base.TIME_STEP_HOURS, unit="h")
        derived_steps = (normalized["datetime_utc"] - base_time) / step_delta
        rounded_steps = derived_steps.round()
        if not ((derived_steps - rounded_steps).abs() < 1e-9).all():
            raise ValueError(
                f"Electric boiler profile timestamps are not aligned to {base.TIME_STEP_HOURS} h steps: {profile_csv}"
            )
        normalized["time_step"] = rounded_steps.astype(int)
        duplicate_mask = normalized["datetime_utc"].duplicated(keep=False)
        if duplicate_mask.any():
            raise ValueError(f"Electric boiler profile has duplicate datetime_utc rows: {profile_csv}")
    else:
        normalized["time_step"] = pd.to_numeric(normalized["time_step"], errors="coerce")
        if normalized["time_step"].isna().any():
            raise ValueError(f"Electric boiler profile contains invalid time_step values: {profile_csv}")
        rounded_steps = normalized["time_step"].round()
        if not ((normalized["time_step"] - rounded_steps).abs() < 1e-9).all():
            raise ValueError(f"Electric boiler profile time_step values must be integers: {profile_csv}")
        normalized["time_step"] = rounded_steps.astype(int)

    if normalized["time_step"].duplicated(keep=False).any():
        raise ValueError(f"Electric boiler profile has duplicate time_step rows: {profile_csv}")

    q_source = normalized["q_mvar"] if "q_mvar" in normalized.columns else pd.Series(0.0, index=normalized.index)
    scaling_source = normalized["scaling"] if "scaling" in normalized.columns else pd.Series(1.0, index=normalized.index)
    normalized["p_mw"] = pd.to_numeric(normalized["p_mw"], errors="coerce").fillna(0.0)
    normalized["q_mvar"] = pd.to_numeric(q_source, errors="coerce").fillna(0.0)
    normalized["scaling"] = pd.to_numeric(scaling_source, errors="coerce").fillna(1.0)

    if "in_service" in normalized.columns:
        in_service_raw = normalized["in_service"]
        in_service_numeric = pd.to_numeric(in_service_raw, errors="coerce")
        in_service = pd.Series(True, index=normalized.index)
        numeric_mask = in_service_numeric.notna()
        in_service.loc[numeric_mask] = in_service_numeric.loc[numeric_mask] != 0.0
        text_mask = ~numeric_mask
        if text_mask.any():
            in_service.loc[text_mask] = in_service_raw.loc[text_mask].astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes", "y", "on"]
            )
    else:
        in_service = pd.Series(True, index=normalized.index)

    normalized["p_mw"] = normalized["p_mw"] * normalized["scaling"] * in_service.astype(float)
    normalized["q_mvar"] = normalized["q_mvar"] * normalized["scaling"] * in_service.astype(float)
    return normalized.sort_values("time_step").reset_index(drop=True)


def add_electric_boiler_profile(net, abs_vals, profile_csv, bus_name, load_name):
    profile_csv = Path(profile_csv)
    if not profile_csv.exists():
        raise FileNotFoundError(f"Electric boiler profile not found: {profile_csv}")

    profile_df = _normalize_boiler_profile(profile_csv)
    target_steps = pd.Index(abs_vals[("load", "p_mw")].index.astype(int), name="time_step")
    profile_by_step = profile_df.set_index("time_step")

    aligned_profile_df = None
    if "datetime_utc" in profile_df.columns:
        profile_by_datetime = profile_df.set_index("datetime_utc")
        target_datetimes = pd.date_range(
            start=pd.Timestamp("2023-01-01 00:00:00Z"),
            periods=len(target_steps),
            freq=pd.to_timedelta(base.TIME_STEP_HOURS, unit="h"),
            tz="UTC",
            name="datetime_utc",
        )
        overlapping_datetimes = profile_by_datetime.index.intersection(target_datetimes)
        if not overlapping_datetimes.empty:
            aligned_profile_df = profile_by_datetime.loc[overlapping_datetimes, ["p_mw", "q_mvar"]].copy()
            aligned_profile_df = aligned_profile_df.reindex(target_datetimes).fillna(0.0)
            aligned_profile_df.index = target_steps

    if aligned_profile_df is None:
        overlapping_steps = profile_by_step.index.intersection(target_steps)
        if overlapping_steps.empty:
            if len(profile_by_step) < len(target_steps):
                raise ValueError(
                    f"Electric boiler profile {profile_csv} has {len(profile_by_step)} rows, "
                    f"but the SimBench study requires at least {len(target_steps)} time steps."
                )
            aligned_profile_df = profile_by_step.iloc[: len(target_steps)][["p_mw", "q_mvar"]].copy()
            aligned_profile_df.index = target_steps
        else:
            aligned_profile_df = profile_by_step.loc[overlapping_steps, ["p_mw", "q_mvar"]].copy()
            aligned_profile_df = aligned_profile_df.reindex(target_steps).fillna(0.0)

    boiler_bus_idx = base.get_bus_index_by_name(net, bus_name)
    initial_p = float(aligned_profile_df["p_mw"].iloc[0]) if not aligned_profile_df.empty else 0.0
    initial_q = float(aligned_profile_df["q_mvar"].iloc[0]) if not aligned_profile_df.empty else 0.0

    boiler_load_idx = base.pp.create_load(
        net,
        bus=boiler_bus_idx,
        p_mw=initial_p,
        q_mvar=initial_q,
        name=load_name,
    )

    load_p = abs_vals[("load", "p_mw")].copy()
    load_p[boiler_load_idx] = 0.0
    load_p.loc[target_steps, boiler_load_idx] = aligned_profile_df["p_mw"].to_numpy(dtype=float)
    abs_vals[("load", "p_mw")] = load_p

    if ("load", "q_mvar") in abs_vals:
        load_q = abs_vals[("load", "q_mvar")].copy()
    else:
        load_q = pd.DataFrame(0.0, index=load_p.index, columns=load_p.columns)
    load_q[boiler_load_idx] = 0.0
    load_q.loc[target_steps, boiler_load_idx] = aligned_profile_df["q_mvar"].to_numpy(dtype=float)
    abs_vals[("load", "q_mvar")] = load_q

    return boiler_load_idx


def add_configured_electric_boilers(net, abs_vals):
    boiler_indices = []
    for boiler_config in base.ELECTRIC_BOILER_CONFIGS:
        boiler_idx = add_electric_boiler_profile(
            net,
            abs_vals,
            boiler_config["profile_csv"],
            boiler_config["bus_name"],
            boiler_config["load_name"],
        )
        boiler_indices.append(boiler_idx)
    return boiler_indices


def _get_time_lookup_and_steps(season_name):
    temp_net = sb.get_simbench_net(base.GRID_CODE)
    temp_abs_vals = sb.get_absolute_values(temp_net, profiles_instead_of_study_cases=True)
    max_steps = base.available_time_steps(temp_abs_vals)
    weather_calendar = base.load_weather_calendar(max_steps)
    time_lookup = base.build_time_lookup(weather_calendar)
    season_time_steps = base.build_season_time_steps(weather_calendar)
    return time_lookup, season_time_steps[season_name]


def run_single_season_study(season_name, output_root=None):
    if season_name not in base.SEASON_MONTHS:
        raise ValueError(f"Unknown season '{season_name}'. Expected one of: {list(base.SEASON_MONTHS)}")

    warnings.filterwarnings("ignore", category=FutureWarning)
    output_path = output_root or SEASON_OUTPUT_ROOTS[season_name]
    time_lookup, time_steps = _get_time_lookup_and_steps(season_name)

    net = sb.get_simbench_net(base.GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    add_configured_electric_boilers(net, abs_vals)
    sb.apply_const_controllers(net, abs_vals)
    base.prepare_output_writer(net, time_steps, output_path)

    try:
        run_timeseries(
            net,
            time_steps=time_steps,
            run=base.robust_runpp,
            continue_on_divergence=True,
            verbose=True,
        )
    except LoadflowNotConverged as err:
        print(f"Season {season_name} stopped on a non-converged step: {err}")

    base.export_named_results(net, output_path)
    base.export_subnet_results_for_season(net, abs_vals, time_steps, output_path, time_lookup, season_name)

    print(
        f"Finished {season_name} study with {len(time_steps)} time steps. "
        f"Subnet comparison before/after DLR written to: {os.path.join(output_path, base.SUBNET_DIR_NAME)}"
    )
