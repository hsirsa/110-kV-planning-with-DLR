import json
import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
import simbench as sb

GRID_CODE = "1-HV-mixed--0-no_sw"
TIME_STEP_HOURS = 0.25
# These buses define the subnet that will be filtered from the full HV1 results.
SUBNET_BUS_NAMES = [
    "EHV Bus 143",
    "HV1 Bus 5",
    "HV1 Bus 47",
    "HV1 Bus 21",
    "HV1 Bus 49",
    "HV1 Bus 67",
    "HV1 Bus 19",
    "HV1 Bus 35",
]
CONTROLLED_HV_BUS_NAMES = [
    "HV1 Bus 5",
    "HV1 Bus 47",
    "HV1 Bus 21",
    "HV1 Bus 49",
    "HV1 Bus 67",
    "HV1 Bus 19",
    "HV1 Bus 35",
]
CONTROL_METRICS_TABLE = "control_metrics"
CONTROL_ACTIVATION_THRESHOLD_PU = 1.08
Q_DROOP_START_PU = 1.06
Q_DROOP_FULL_PU = 1.08
FINAL_VOLTAGE_LIMIT_PU = 1.10
EMERGENCY_CURTAILMENT_BUS_NAMES = ["HV1 Bus 19", "HV1 Bus 49", "HV1 Bus 21"]
EMERGENCY_CURTAILMENT_STEP_FRACTION = 0.05
EMERGENCY_CURTAILMENT_MAX_FRACTION = 0.50
MAX_OLTC_STEP_CHANGE_PER_STAGE = 2
CONTROL_STAGE_NONE = 0
CONTROL_STAGE_OLTC_Q = 1
CONTROL_STAGE_EMERGENCY = 2
SUBNET_DIR_NAME = "subnet_focus"
WEATHER_DIR = Path("weather_tampere_dlr")
AITOLAHTI_WEATHER_CSV = WEATHER_DIR / "Aitolahti_Tampere_Finland_2023_dlr_weather_15min.csv"
CONDUCTOR_TEMP_C = 75.0
DEFAULT_EMISSIVITY = 0.8
DEFAULT_ABSORPTIVITY = 0.8
DEFAULT_ALPHA = 0.00403
SIGMA = 5.670374419e-8

# External electric-boiler profiles that are added as new time-varying loads
# before the full HV1 time-series power flow is executed.
ELECTRIC_BOILER_CONFIGS = [
    {
        "profile_csv": Path(r"c:\Users\hsirsa\PycharmProjects\PythonProject4\electric_boiler_50MW_newprice.csv"),
        "bus_name": "HV1 Bus 67",
        "load_name": "Electric boiler 50 MW",
    },
    {
        "profile_csv": Path(r"c:\Users\hsirsa\PycharmProjects\PythonProject4\electric_boiler_100MW_newprice.csv"),
        "bus_name": "HV1 Bus 35",
        "load_name": "Electric boiler 100 MW",
    },
]


def robust_runpp(net, **kwargs):
    solver_options = [
        {
            "algorithm": "nr",
            "init": "dc",
            "check_connectivity": True,
            "calculate_voltage_angles": True,
            "max_iteration": 50,
            "tolerance_mva": 1e-6,
            "numba": False,
        },
        {
            "algorithm": "nr",
            "init": "flat",
            "check_connectivity": True,
            "calculate_voltage_angles": True,
            "max_iteration": 50,
            "tolerance_mva": 1e-6,
            "numba": False,
        },
        {
            "algorithm": "fdbx",
            "init": "flat",
            "check_connectivity": True,
            "calculate_voltage_angles": True,
            "max_iteration": 100,
            "tolerance_mva": 1e-6,
            "numba": False,
        },
    ]
    last_error = None
    for options in solver_options:
        current_options = options.copy()
        current_options.update(kwargs)
        try:
            pp.runpp(net, **current_options)
            return
        except Exception as err:
            last_error = err
    raise LoadflowNotConverged(f"All solver attempts failed. Last error: {last_error}")


def get_subnet_bus_indices(net):
    return list(get_subnet_bus_map(net, SUBNET_BUS_NAMES).values())


def get_controlled_hv_bus_indices(net):
    return list(get_subnet_bus_map(net, CONTROLLED_HV_BUS_NAMES).values())


def get_emergency_curtailment_bus_indices(net):
    return list(get_subnet_bus_map(net, EMERGENCY_CURTAILMENT_BUS_NAMES).values())


def subnet_max_voltage_pu(net, bus_indices):
    if "res_bus" not in net or net.res_bus.empty or "vm_pu" not in net.res_bus.columns:
        return float("nan")
    valid_indices = [idx for idx in bus_indices if idx in net.res_bus.index]
    if not valid_indices:
        return float("nan")
    return float(net.res_bus.loc[valid_indices, "vm_pu"].max())


def set_voltage_controller_targets(net, vm_target_pu):
    if "controller" not in net or net.controller is None or net.controller.empty:
        return 0
    updated = 0
    controller_column = "object" if "object" in net.controller.columns else None
    for idx in net.controller.index:
        controller_obj = net.controller.at[idx, controller_column] if controller_column else None
        if controller_obj is None:
            continue
        if hasattr(controller_obj, "vm_set_pu"):
            controller_obj.vm_set_pu = float(vm_target_pu)
            updated += 1
    return updated


def subnet_sgen_q_limit_mvar(p_mw):
    apparent_p = abs(float(p_mw))
    if apparent_p <= 0.0:
        return 0.0
    return apparent_p * math.tan(math.acos(0.95))


def apply_subnet_sgen_voltage_droop(net, subnet_bus_indices):
    if net.sgen.empty:
        return 0
    subnet_bus_set = set(int(idx) for idx in subnet_bus_indices)
    updated = 0
    droop_span = max(Q_DROOP_FULL_PU - Q_DROOP_START_PU, 1e-9)
    for sgen_idx in net.sgen.index:
        bus_idx = int(net.sgen.at[sgen_idx, "bus"])
        if bus_idx not in subnet_bus_set or bus_idx not in net.res_bus.index:
            continue
        vm_pu = float(net.res_bus.at[bus_idx, "vm_pu"])
        if vm_pu <= Q_DROOP_START_PU:
            q_mvar = 0.0
        else:
            droop = min(max((vm_pu - Q_DROOP_START_PU) / droop_span, 0.0), 1.0)
            q_limit = subnet_sgen_q_limit_mvar(net.sgen.at[sgen_idx, "p_mw"])
            q_mvar = -q_limit * droop
        net.sgen.at[sgen_idx, "q_mvar"] = float(q_mvar)
        updated += 1
    return updated


def get_total_absorbed_sgen_q_mvar(net, subnet_bus_indices):
    if net.sgen.empty:
        return 0.0
    subnet_mask = net.sgen["bus"].isin(subnet_bus_indices)
    return float((-net.sgen.loc[subnet_mask, "q_mvar"].clip(upper=0.0)).sum())


def get_controlled_oltc_trafo_indices(net, controlled_hv_bus_indices):
    if net.trafo.empty:
        return []
    controlled_set = set(int(idx) for idx in controlled_hv_bus_indices)
    indices = []
    for trafo_idx in net.trafo.index:
        row = net.trafo.loc[trafo_idx]
        tap_min = row.get("tap_min")
        tap_max = row.get("tap_max")
        tap_step_percent = row.get("tap_step_percent")
        if not bool(row.get("in_service", True)):
            continue
        if pd.isna(tap_min) or pd.isna(tap_max) or pd.isna(tap_step_percent):
            continue
        if float(tap_step_percent) == 0.0 or float(tap_min) == float(tap_max):
            continue
        hv_bus = int(row["hv_bus"])
        lv_bus = int(row["lv_bus"])
        hv_kv = float(row.get("vn_hv_kv", 0.0) or 0.0)
        lv_kv = float(row.get("vn_lv_kv", 0.0) or 0.0)
        controlled_on_lv = lv_bus in controlled_set and lv_kv <= 120.0 and hv_kv >= 220.0
        controlled_on_hv = hv_bus in controlled_set and hv_kv <= 120.0 and lv_kv >= 220.0
        if controlled_on_lv or controlled_on_hv:
            indices.append(int(trafo_idx))
    return indices


def get_controlled_side_for_trafo(trafo_row, controlled_hv_bus_indices):
    controlled_set = set(int(idx) for idx in controlled_hv_bus_indices)
    if int(trafo_row["lv_bus"]) in controlled_set:
        return "lv"
    if int(trafo_row["hv_bus"]) in controlled_set:
        return "hv"
    return None


def get_oltc_step_direction(trafo_row, controlled_side):
    tap_side = str(trafo_row.get("tap_side", "")).strip().lower()
    if controlled_side == "lv":
        return 1 if tap_side == "hv" else -1
    if controlled_side == "hv":
        return -1 if tap_side == "hv" else 1
    return 0


def apply_explicit_oltc_adjustment(net, controlled_hv_bus_indices, controlled_hv_max_vm):
    trafo_indices = get_controlled_oltc_trafo_indices(net, controlled_hv_bus_indices)
    if not trafo_indices:
        return []
    voltage_excess = max(controlled_hv_max_vm - CONTROL_ACTIVATION_THRESHOLD_PU, 0.0)
    requested_steps = max(1, min(MAX_OLTC_STEP_CHANGE_PER_STAGE, int(math.ceil(voltage_excess / 0.01))))
    changed = []
    for trafo_idx in trafo_indices:
        row = net.trafo.loc[trafo_idx]
        controlled_side = get_controlled_side_for_trafo(row, controlled_hv_bus_indices)
        direction = get_oltc_step_direction(row, controlled_side)
        if direction == 0:
            continue
        current_tap = row.get("tap_pos")
        if pd.isna(current_tap):
            current_tap = row.get("tap_neutral", 0.0)
        current_tap = int(round(float(current_tap)))
        tap_min = int(math.floor(float(row.get("tap_min", current_tap))))
        tap_max = int(math.ceil(float(row.get("tap_max", current_tap))))
        target_tap = max(min(current_tap + direction * requested_steps, tap_max), tap_min)
        if target_tap != current_tap:
            net.trafo.at[trafo_idx, "tap_pos"] = target_tap
            changed.append(int(trafo_idx))
    return changed


def control_metric_tap_column(trafo_idx):
    return f"oltc_tap_pos_trafo_{int(trafo_idx)}"


def ensure_control_metrics_table(net):
    if CONTROL_METRICS_TABLE not in net or not isinstance(net[CONTROL_METRICS_TABLE], pd.DataFrame):
        net[CONTROL_METRICS_TABLE] = pd.DataFrame(index=[0])
    metrics_df = net[CONTROL_METRICS_TABLE]
    defaults = {
        "max_controlled_hv_vm_pu": float("nan"),
        "final_max_controlled_hv_vm_pu": float("nan"),
        "total_sgen_q_absorption_mvar": 0.0,
        "voltage_control_active": False,
        "voltage_control_stage_used": CONTROL_STAGE_NONE,
        "curtailed_mw": 0.0,
        "final_voltage_limit_satisfied": False,
    }
    for column, default_value in defaults.items():
        if column not in metrics_df.columns:
            metrics_df[column] = default_value
        metrics_df.at[0, column] = metrics_df.at[0, column] if 0 in metrics_df.index else default_value
    try:
        controlled_hv_bus_indices = get_controlled_hv_bus_indices(net)
    except Exception:
        controlled_hv_bus_indices = []
    for trafo_idx in get_controlled_oltc_trafo_indices(net, controlled_hv_bus_indices):
        column = control_metric_tap_column(trafo_idx)
        default_tap = net.trafo.at[trafo_idx, "tap_pos"]
        if pd.isna(default_tap):
            default_tap = net.trafo.at[trafo_idx, "tap_neutral"] if "tap_neutral" in net.trafo.columns else 0.0
        if column not in metrics_df.columns:
            metrics_df[column] = float(default_tap)
    return metrics_df


def update_control_metrics(net, controlled_hv_bus_indices, initial_max_vm, final_max_vm, voltage_control_active, stage_used, curtailed_mw):
    metrics_df = ensure_control_metrics_table(net)
    metrics_df.at[0, "max_controlled_hv_vm_pu"] = float(initial_max_vm) if not np.isnan(initial_max_vm) else float("nan")
    metrics_df.at[0, "final_max_controlled_hv_vm_pu"] = float(final_max_vm) if not np.isnan(final_max_vm) else float("nan")
    metrics_df.at[0, "total_sgen_q_absorption_mvar"] = get_total_absorbed_sgen_q_mvar(net, controlled_hv_bus_indices)
    metrics_df.at[0, "voltage_control_active"] = bool(voltage_control_active)
    metrics_df.at[0, "voltage_control_stage_used"] = int(stage_used)
    metrics_df.at[0, "curtailed_mw"] = float(curtailed_mw)
    metrics_df.at[0, "final_voltage_limit_satisfied"] = bool(not np.isnan(final_max_vm) and final_max_vm <= FINAL_VOLTAGE_LIMIT_PU)
    for trafo_idx in get_controlled_oltc_trafo_indices(net, controlled_hv_bus_indices):
        column = control_metric_tap_column(trafo_idx)
        metrics_df.at[0, column] = float(net.trafo.at[trafo_idx, "tap_pos"])


def run_voltage_control_stage(net, **kwargs):
    control_kwargs = {
        "algorithm": "nr",
        "init": "results",
        "check_connectivity": True,
        "calculate_voltage_angles": True,
        "max_iteration": 50,
        "tolerance_mva": 1e-6,
        "numba": False,
        "run_control": True,
    }
    control_kwargs.update(kwargs)
    pp.runpp(net, **control_kwargs)


def apply_emergency_wind_curtailment(net, controlled_hv_bus_indices, **kwargs):
    if net.sgen.empty:
        return 0.0
    critical_bus_indices = set(get_emergency_curtailment_bus_indices(net))
    sgen_indices = [
        int(idx)
        for idx in net.sgen.index
        if int(net.sgen.at[idx, "bus"]) in critical_bus_indices and float(net.sgen.at[idx, "p_mw"]) > 0.0
    ]
    if not sgen_indices:
        return 0.0
    base_p = {idx: max(float(net.sgen.at[idx, "p_mw"]), 0.0) for idx in sgen_indices}
    max_iterations = max(1, int(math.ceil(EMERGENCY_CURTAILMENT_MAX_FRACTION / EMERGENCY_CURTAILMENT_STEP_FRACTION)))
    total_curtailed = 0.0
    for _ in range(max_iterations):
        step_curtailed = 0.0
        for sgen_idx in sgen_indices:
            max_curtail = base_p[sgen_idx] * EMERGENCY_CURTAILMENT_MAX_FRACTION
            current_curtail = base_p[sgen_idx] - float(net.sgen.at[sgen_idx, "p_mw"])
            remaining_curtail = max_curtail - current_curtail
            if remaining_curtail <= 1e-9:
                continue
            step_size = min(base_p[sgen_idx] * EMERGENCY_CURTAILMENT_STEP_FRACTION, remaining_curtail)
            current_p = float(net.sgen.at[sgen_idx, "p_mw"])
            new_p = max(current_p - step_size, 0.0)
            actual_step = current_p - new_p
            if actual_step <= 0.0:
                continue
            net.sgen.at[sgen_idx, "p_mw"] = new_p
            step_curtailed += actual_step
        if step_curtailed <= 0.0:
            break
        total_curtailed += step_curtailed
        robust_runpp(net, **kwargs)
        apply_subnet_sgen_voltage_droop(net, controlled_hv_bus_indices)
        robust_runpp(net, **kwargs)
        if subnet_max_voltage_pu(net, controlled_hv_bus_indices) <= FINAL_VOLTAGE_LIMIT_PU:
            break
    return total_curtailed


def voltage_controlled_runpp(net, **kwargs):
    ensure_control_metrics_table(net)
    robust_runpp(net, **kwargs)
    controlled_hv_bus_indices = get_controlled_hv_bus_indices(net)
    initial_max_vm = subnet_max_voltage_pu(net, controlled_hv_bus_indices)
    voltage_control_active = not np.isnan(initial_max_vm) and initial_max_vm > CONTROL_ACTIVATION_THRESHOLD_PU
    if not voltage_control_active:
        update_control_metrics(
            net,
            controlled_hv_bus_indices,
            initial_max_vm,
            initial_max_vm,
            False,
            CONTROL_STAGE_NONE,
            0.0,
        )
        return

    stage_used = CONTROL_STAGE_OLTC_Q
    curtailed_mw = 0.0
    controller_updates = set_voltage_controller_targets(net, 1.00)
    apply_subnet_sgen_voltage_droop(net, controlled_hv_bus_indices)

    if controller_updates > 0:
        run_voltage_control_stage(net, **kwargs)
    else:
        apply_explicit_oltc_adjustment(net, controlled_hv_bus_indices, initial_max_vm)
        robust_runpp(net, **kwargs)

    post_control_max_vm = subnet_max_voltage_pu(net, controlled_hv_bus_indices)
    if not np.isnan(post_control_max_vm) and post_control_max_vm > FINAL_VOLTAGE_LIMIT_PU:
        stage_used = CONTROL_STAGE_EMERGENCY
        curtailed_mw = apply_emergency_wind_curtailment(net, controlled_hv_bus_indices, **kwargs)

    final_max_vm = subnet_max_voltage_pu(net, controlled_hv_bus_indices)
    update_control_metrics(
        net,
        controlled_hv_bus_indices,
        initial_max_vm,
        final_max_vm,
        True,
        stage_used,
        curtailed_mw,
    )


def safe_name(value, fallback):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def available_time_steps(abs_vals):
    return len(abs_vals[("load", "p_mw")].index)


def get_bus_index_by_name(net, bus_name):
    for idx in net.bus.index:
        if safe_name(net.bus.at[idx, "name"], f"bus_{idx}") == bus_name:
            return int(idx)
    raise KeyError(f"Bus not found in SimBench net: {bus_name}")


def add_electric_boiler_profile(net, abs_vals, profile_csv, bus_name, load_name):
    if not profile_csv.exists():
        raise FileNotFoundError(f"Electric boiler profile not found: {profile_csv}")

    profile_df = pd.read_csv(profile_csv)
    if "p_mw" not in profile_df.columns:
        raise ValueError("Electric boiler profile must contain a 'p_mw' column.")
    if "time_step" not in profile_df.columns and "datetime_utc" not in profile_df.columns:
        raise ValueError(
            "Electric boiler profile must contain either 'time_step' or 'datetime_utc' for SimBench alignment."
        )

    profile_df = profile_df.copy()
    if "time_step" in profile_df.columns:
        profile_df["time_step"] = pd.to_numeric(profile_df["time_step"], errors="coerce")
    else:
        profile_datetimes = pd.to_datetime(profile_df["datetime_utc"], errors="coerce", utc=True)
        base_datetime = pd.Timestamp("2023-01-01 00:00:00Z")
        step_delta = pd.to_timedelta(TIME_STEP_HOURS, unit="h")
        derived_steps = (profile_datetimes - base_datetime) / step_delta
        profile_df["time_step"] = pd.to_numeric(derived_steps, errors="coerce")
    profile_df = profile_df.dropna(subset=["time_step"]).copy()
    profile_df["time_step"] = profile_df["time_step"].round().astype(int)
    q_source = profile_df["q_mvar"] if "q_mvar" in profile_df.columns else pd.Series(0.0, index=profile_df.index)
    scaling_source = profile_df["scaling"] if "scaling" in profile_df.columns else pd.Series(1.0, index=profile_df.index)
    profile_df["q_mvar"] = pd.to_numeric(q_source, errors="coerce").fillna(0.0)
    profile_df["scaling"] = pd.to_numeric(scaling_source, errors="coerce").fillna(1.0)
    if "in_service" in profile_df.columns:
        in_service_raw = profile_df["in_service"]
        in_service_numeric = pd.to_numeric(in_service_raw, errors="coerce")
        in_service = pd.Series(True, index=profile_df.index)
        numeric_mask = in_service_numeric.notna()
        in_service.loc[numeric_mask] = in_service_numeric.loc[numeric_mask] != 0.0
        text_mask = ~numeric_mask
        if text_mask.any():
            in_service.loc[text_mask] = in_service_raw.loc[text_mask].astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes", "y", "on"]
            )
    else:
        in_service = pd.Series(True, index=profile_df.index)

    profile_df["p_mw"] = pd.to_numeric(profile_df["p_mw"], errors="coerce").fillna(0.0) * profile_df["scaling"] * in_service.astype(float)
    profile_df["q_mvar"] = profile_df["q_mvar"] * profile_df["scaling"] * in_service.astype(float)

    target_steps = pd.Index(abs_vals[("load", "p_mw")].index.astype(int), name="time_step")
    profile_df = profile_df.sort_values("time_step").drop_duplicates(subset=["time_step"], keep="first")
    profile_by_step = profile_df.set_index("time_step")

    # The improved boiler profiles carry both time_step and datetime_utc. When a
    # valid timestamp column is present, align with the expected SimBench 2023
    # 15-minute calendar first and only fall back to raw time_step alignment if
    # the timestamps are missing or unusable.
    aligned_profile_df = None
    if "datetime_utc" in profile_df.columns:
        profile_datetimes = pd.to_datetime(profile_df["datetime_utc"], errors="coerce", utc=True)
        valid_datetime_mask = profile_datetimes.notna()
        if valid_datetime_mask.any():
            profile_by_datetime = profile_df.loc[valid_datetime_mask].copy()
            profile_by_datetime["datetime_utc"] = profile_datetimes.loc[valid_datetime_mask]
            profile_by_datetime = profile_by_datetime.sort_values("datetime_utc").drop_duplicates(
                subset=["datetime_utc"], keep="first"
            )
            profile_by_datetime = profile_by_datetime.set_index("datetime_utc")
            target_datetimes = pd.date_range(
                start=pd.Timestamp("2023-01-01 00:00:00Z"),
                periods=len(target_steps),
                freq=pd.to_timedelta(TIME_STEP_HOURS, unit="h"),
                tz="UTC",
                name="datetime_utc",
            )
            overlapping_datetimes = profile_by_datetime.index.intersection(target_datetimes)
            if not overlapping_datetimes.empty:
                aligned_profile_df = profile_by_datetime.loc[overlapping_datetimes, ["p_mw", "q_mvar"]].copy()
                aligned_profile_df = aligned_profile_df.reindex(target_datetimes).fillna(0.0)
                aligned_profile_df.index = target_steps

    if aligned_profile_df is None:
        # Fallback for legacy profiles that only carry numeric study indices.
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

    profile_df = aligned_profile_df

    boiler_bus_idx = get_bus_index_by_name(net, bus_name)
    initial_p = float(profile_df["p_mw"].iloc[0]) if not profile_df.empty else 0.0
    initial_q = float(profile_df["q_mvar"].iloc[0]) if not profile_df.empty else 0.0

    boiler_load_idx = pp.create_load(
        net,
        bus=boiler_bus_idx,
        p_mw=initial_p,
        q_mvar=initial_q,
        name=load_name,
    )

    load_p = abs_vals[("load", "p_mw")].copy()
    load_p[boiler_load_idx] = 0.0
    load_p.loc[target_steps, boiler_load_idx] = profile_df["p_mw"].to_numpy(dtype=float)
    abs_vals[("load", "p_mw")] = load_p

    if ("load", "q_mvar") in abs_vals:
        load_q = abs_vals[("load", "q_mvar")].copy()
    else:
        load_q = pd.DataFrame(0.0, index=load_p.index, columns=load_p.columns)
    load_q[boiler_load_idx] = 0.0
    load_q.loc[target_steps, boiler_load_idx] = profile_df["q_mvar"].to_numpy(dtype=float)
    abs_vals[("load", "q_mvar")] = load_q

    return boiler_load_idx


def add_configured_electric_boilers(net, abs_vals):
    boiler_indices = []
    for boiler_config in ELECTRIC_BOILER_CONFIGS:
        boiler_idx = add_electric_boiler_profile(
            net,
            abs_vals,
            boiler_config["profile_csv"],
            boiler_config["bus_name"],
            boiler_config["load_name"],
        )
        boiler_indices.append(boiler_idx)
    return boiler_indices

def prepare_output_writer(net, time_steps, output_path):
    os.makedirs(output_path, exist_ok=True)
    metrics_df = ensure_control_metrics_table(net)
    ow = OutputWriter(net, time_steps=time_steps, output_path=output_path, output_file_type=".csv")
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "i_ka")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    ow.log_variable("sgen", "q_mvar")
    ow.log_variable("res_sgen", "q_mvar")
    ow.log_variable("trafo", "tap_pos")
    for column in metrics_df.columns:
        ow.log_variable(CONTROL_METRICS_TABLE, column)
    return ow


def export_result_with_names(csv_path, meta, name_column, index_label, value_column, output_path):
    df = pd.read_csv(csv_path, sep=";", decimal=".", index_col=0)
    df.index.name = "time_step"
    df.columns = [int(col) for col in df.columns]
    meta_export = meta.copy()
    meta_export[index_label] = meta_export.index
    meta_export[name_column] = meta_export[name_column].fillna("").astype(str)
    long_df = df.reset_index().melt(id_vars=["time_step"], var_name=index_label, value_name=value_column)
    merged = long_df.merge(meta_export[[index_label, name_column]], on=index_label, how="left")
    merged = merged[["time_step", index_label, name_column, value_column]]
    merged.to_csv(output_path, index=False)


def export_named_results(net, output_path):
    export_result_with_names(
        os.path.join(output_path, "res_bus", "vm_pu.csv"), net.bus, "name", "bus_index", "vm_pu",
        os.path.join(output_path, "bus_vm_pu_named.csv"),
    )
    export_result_with_names(
        os.path.join(output_path, "res_line", "i_ka.csv"), net.line, "name", "line_index", "i_ka",
        os.path.join(output_path, "line_i_ka_named.csv"),
    )
    export_result_with_names(
        os.path.join(output_path, "res_line", "loading_percent.csv"), net.line, "name", "line_index", "loading_percent",
        os.path.join(output_path, "line_loading_percent_named.csv"),
    )
    export_result_with_names(
        os.path.join(output_path, "res_trafo", "loading_percent.csv"), net.trafo, "name", "trafo_index", "loading_percent",
        os.path.join(output_path, "trafo_loading_percent_named.csv"),
    )


def filter_named_results(csv_path, index_label, allowed_indices):
    if not allowed_indices:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df[df[index_label].isin(allowed_indices)].copy()


def first_valid_value(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return None


def get_bus_coordinate_map(net, bus_indices):
    if hasattr(net, "bus_geodata") and net.bus_geodata is not None and len(net.bus_geodata) > 0:
        geodata = net.bus_geodata
        geodata_cols = {col.lower(): col for col in geodata.columns}
        if "x" in geodata_cols and "y" in geodata_cols:
            pos = {}
            for bus_idx in bus_indices:
                if bus_idx in geodata.index:
                    pos[int(bus_idx)] = (
                        float(geodata.at[bus_idx, geodata_cols["x"]]),
                        float(geodata.at[bus_idx, geodata_cols["y"]]),
                    )
            if pos:
                return pos
    if "geo" in net.bus.columns:
        pos = {}
        for bus_idx in bus_indices:
            geo = net.bus.at[bus_idx, "geo"]
            if not isinstance(geo, str) or not geo.strip():
                continue
            try:
                geo_json = json.loads(geo)
            except json.JSONDecodeError:
                continue
            coords = geo_json.get("coordinates", [])
            if geo_json.get("type") == "Point" and len(coords) >= 2:
                pos[int(bus_idx)] = (float(coords[0]), float(coords[1]))
        return pos
    return {}


def get_subnet_positions(net, subnet_bus_indices, graph):
    pos = get_bus_coordinate_map(net, subnet_bus_indices)
    if len(pos) == len(subnet_bus_indices):
        return pos
    return nx.spring_layout(graph, seed=42)


def get_subnet_bus_map(net, bus_names):
    available = {safe_name(net.bus.at[idx, "name"], f"bus_{idx}"): int(idx) for idx in net.bus.index}
    missing = [name for name in bus_names if name not in available]
    if missing:
        raise KeyError(f"Subnet bus names not found in SimBench net: {missing}")
    return {name: available[name] for name in bus_names}


def aggregate_count_by_bus(element_df, bus_column, subnet_bus_indices):
    if element_df.empty:
        return [0] * len(subnet_bus_indices)
    counts = element_df.groupby(bus_column).size()
    return [int(counts.get(idx, 0)) for idx in subnet_bus_indices]


def aggregate_sum_by_bus(element_df, bus_column, value_column, subnet_bus_indices):
    if element_df.empty or value_column not in element_df.columns:
        return [0.0] * len(subnet_bus_indices)
    sums = element_df.groupby(bus_column)[value_column].sum()
    return [float(sums.get(idx, 0.0)) for idx in subnet_bus_indices]


def build_subnet_bus_summary(net, subnet_bus_indices):
    bus_df = net.bus.loc[subnet_bus_indices].copy()
    bus_df["bus_index"] = bus_df.index.astype(int)
    bus_df["bus_name"] = [safe_name(bus_df.at[idx, "name"], f"bus_{idx}") for idx in bus_df.index]
    bus_df["load_count"] = aggregate_count_by_bus(net.load, "bus", subnet_bus_indices)
    bus_df["load_p_mw"] = aggregate_sum_by_bus(net.load, "bus", "p_mw", subnet_bus_indices)
    bus_df["load_q_mvar"] = aggregate_sum_by_bus(net.load, "bus", "q_mvar", subnet_bus_indices)
    bus_df["sgen_count"] = aggregate_count_by_bus(net.sgen, "bus", subnet_bus_indices)
    bus_df["sgen_p_mw"] = aggregate_sum_by_bus(net.sgen, "bus", "p_mw", subnet_bus_indices)
    bus_df["sgen_q_mvar"] = aggregate_sum_by_bus(net.sgen, "bus", "q_mvar", subnet_bus_indices)
    bus_df["gen_count"] = aggregate_count_by_bus(net.gen, "bus", subnet_bus_indices)
    bus_df["gen_p_mw"] = aggregate_sum_by_bus(net.gen, "bus", "p_mw", subnet_bus_indices)
    bus_df["is_ext_grid_bus"] = bus_df["bus_index"].isin(net.ext_grid["bus"].tolist())
    return bus_df.reset_index(drop=True)

def compute_line_azimuth_deg(net, from_bus, to_bus):
    pos = get_bus_coordinate_map(net, [from_bus, to_bus])
    if len(pos) < 2:
        return 90.0
    dx = pos[to_bus][0] - pos[from_bus][0]
    dy = pos[to_bus][1] - pos[from_bus][1]
    return math.degrees(math.atan2(dy, dx)) % 180.0


def build_conductor_properties(net, line_row):
    std_type_name = line_row.get("std_type")
    std_line = net.std_types.get("line", {}).get(std_type_name, {}) if std_type_name else {}
    q_mm2 = first_valid_value(std_line.get("q_mm2"), line_row.get("q_mm2"))
    diameter_value = first_valid_value(
        std_line.get("diameter_mm"), std_line.get("diameter"), line_row.get("diameter_mm"), line_row.get("diameter_m")
    )
    if diameter_value is not None:
        diameter_m = float(diameter_value)
        if diameter_m > 1.0:
            diameter_m /= 1000.0
    elif q_mm2:
        diameter_m = 2.0 * math.sqrt(float(q_mm2) * 1e-6 / math.pi)
    else:
        diameter_m = 0.03
    return {
        "q_mm2": float(q_mm2) if q_mm2 is not None else np.nan,
        "diameter_m_est": float(diameter_m),
        "resistance_20c_ohm_per_km": float(line_row.get("r_ohm_per_km", std_line.get("r_ohm_per_km", 0.0))),
        "alpha_per_c": float(first_valid_value(std_line.get("alpha"), line_row.get("alpha"), DEFAULT_ALPHA)),
        "emissivity": float(first_valid_value(std_line.get("emissivity"), line_row.get("emissivity"), DEFAULT_EMISSIVITY)),
        "absorptivity": float(first_valid_value(std_line.get("absorptivity"), line_row.get("absorptivity"), DEFAULT_ABSORPTIVITY)),
        "parallel_count": int(line_row.get("parallel", 1) or 1),
        "line_azimuth_deg": compute_line_azimuth_deg(net, int(line_row["from_bus"]), int(line_row["to_bus"])),
        "dlr_conductor_temp_c": CONDUCTOR_TEMP_C,
    }


def build_line_conductor_id(line_df):
    parts = []
    std_types = line_df["std_type"] if "std_type" in line_df.columns else pd.Series("", index=line_df.index)
    for idx in line_df.index:
        std_type = safe_name(std_types.get(idx, ""), "")
        if std_type:
            parts.append(std_type)
            continue
        parts.append(
            "r={:.4f}_x={:.4f}_imax={:.3f}".format(
                float(line_df.at[idx, "r_ohm_per_km"] if "r_ohm_per_km" in line_df.columns else 0.0),
                float(line_df.at[idx, "x_ohm_per_km"] if "x_ohm_per_km" in line_df.columns else 0.0),
                float(line_df.at[idx, "max_i_ka"] if "max_i_ka" in line_df.columns else 0.0),
            )
        )
    return parts


def build_subnet_line_summary(net, subnet_lines):
    if subnet_lines.empty:
        return pd.DataFrame()
    line_df = subnet_lines.copy()
    line_df["line_index"] = line_df.index.astype(int)
    line_df["line_name"] = [safe_name(line_df.at[idx, "name"], f"line_{idx}") for idx in line_df.index]
    line_df["from_bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in line_df["from_bus"]]
    line_df["to_bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in line_df["to_bus"]]
    if "parallel" in line_df.columns:
        line_df["parallel"] = line_df["parallel"].fillna(1).astype(int)
    conductor_df = pd.DataFrame([build_conductor_properties(net, row) for _, row in line_df.iterrows()], index=line_df.index)
    line_df = pd.concat([line_df, conductor_df], axis=1)
    line_df["conductor_id"] = build_line_conductor_id(line_df)
    return line_df.reset_index(drop=True)


def build_subnet_trafo_summary(net, subnet_trafos):
    if subnet_trafos.empty:
        return pd.DataFrame()
    trafo_df = subnet_trafos.copy()
    trafo_df["trafo_index"] = trafo_df.index.astype(int)
    trafo_df["trafo_name"] = [safe_name(trafo_df.at[idx, "name"], f"trafo_{idx}") for idx in trafo_df.index]
    trafo_df["hv_bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in trafo_df["hv_bus"]]
    trafo_df["lv_bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in trafo_df["lv_bus"]]
    return trafo_df.reset_index(drop=True)


def build_subnet_component_summary(net, element_type, subnet_bus_indices, label):
    element_df = getattr(net, element_type)
    if element_df.empty:
        return pd.DataFrame()
    filtered = element_df[element_df["bus"].isin(subnet_bus_indices)].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered[f"{label}_index"] = filtered.index.astype(int)
    filtered["bus_index"] = filtered["bus"].astype(int)
    filtered["bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in filtered["bus_index"]]
    if "name" in filtered.columns:
        filtered[f"{label}_name"] = [safe_name(filtered.at[idx, "name"], f"{label}_{idx}") for idx in filtered.index]
    return filtered.reset_index(drop=True)


def infer_sgen_generation_type(row):
    candidate_columns = []
    for column in row.index:
        column_name = str(column).lower()
        if any(token in column_name for token in ["type", "profile", "source", "tech", "energy", "fuel", "name"]):
            candidate_columns.append(column)

    text_parts = []
    for column in candidate_columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        text_parts.append(str(value).strip().lower())

    combined_text = " ".join(text_parts)
    if any(token in combined_text for token in ["pv", "photovolta", "solar"]):
        return "pv"
    if "wind" in combined_text:
        return "wind"
    return "other"


def classify_subnet_bus(load_mw, total_generation_mw, pv_mw, wind_mw):
    load_mw = float(load_mw or 0.0)
    total_generation_mw = float(total_generation_mw or 0.0)
    renewable_generation_mw = float(pv_mw or 0.0) + float(wind_mw or 0.0)
    tolerance = 1e-6

    if total_generation_mw <= tolerance and load_mw <= tolerance:
        return "Mixed"
    if total_generation_mw > max(load_mw * 1.2, tolerance):
        if renewable_generation_mw >= 0.5 * total_generation_mw:
            return "Generation-dominated (PV/Wind)"
        return "Generation-dominated (Other)"
    if load_mw > max(total_generation_mw * 1.2, tolerance):
        return "Load-dominated"
    return "Mixed"


def build_bus_time_series_from_elements(net, abs_vals, time_steps, element_type, subnet_bus_indices):
    key_p = (element_type, "p_mw")
    if key_p not in abs_vals:
        return pd.DataFrame()

    element_df = getattr(net, element_type)
    if element_df.empty:
        return pd.DataFrame()

    filtered = element_df[element_df["bus"].isin(subnet_bus_indices)].copy()
    if filtered.empty:
        return pd.DataFrame()

    p_df = abs_vals[key_p].loc[time_steps, filtered.index].copy()
    p_df.columns = [int(col) for col in p_df.columns]
    long_df = p_df.reset_index(names="time_step").melt(
        id_vars=["time_step"],
        var_name="element_index",
        value_name="p_mw",
    )

    meta = filtered[["bus"]].copy()
    meta["element_index"] = meta.index.astype(int)
    meta["bus_index"] = meta["bus"].astype(int)
    meta["bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in meta["bus_index"]]

    if element_type == "sgen":
        meta["generation_type"] = filtered.apply(infer_sgen_generation_type, axis=1).values
    elif element_type == "gen":
        meta["generation_type"] = "other"
    else:
        meta["generation_type"] = "load"

    merged = long_df.merge(meta[["element_index", "bus_index", "bus_name", "generation_type"]], on="element_index", how="left")
    merged["component_type"] = element_type
    return merged[["time_step", "bus_index", "bus_name", "component_type", "generation_type", "p_mw"]]



def build_subnet_bus_generation_mix(net, abs_vals, time_steps, subnet_bus_indices):
    columns = [
        "bus_index",
        "bus_name",
        "mean_load_mw",
        "peak_load_mw",
        "mean_pv_mw",
        "peak_pv_mw",
        "mean_wind_mw",
        "peak_wind_mw",
        "mean_other_gen_mw",
        "peak_other_gen_mw",
        "mean_total_generation_mw",
        "peak_total_generation_mw",
        "mean_net_injection_mw",
        "peak_net_injection_mw",
        "classification",
    ]
    if not subnet_bus_indices:
        return pd.DataFrame(columns=columns)

    frames = []
    for element_type in ("load", "sgen", "gen"):
        frame = build_bus_time_series_from_elements(net, abs_vals, time_steps, element_type, subnet_bus_indices)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns)

    detailed_df = pd.concat(frames, ignore_index=True)

    load_summary = (
        detailed_df[detailed_df["component_type"] == "load"]
        .groupby(["time_step", "bus_index", "bus_name"], as_index=False)["p_mw"]
        .sum()
        .rename(columns={"p_mw": "load_mw"})
    )
    pv_summary = (
        detailed_df[(detailed_df["component_type"] == "sgen") & (detailed_df["generation_type"] == "pv")]
        .groupby(["time_step", "bus_index", "bus_name"], as_index=False)["p_mw"]
        .sum()
        .rename(columns={"p_mw": "pv_mw"})
    )
    wind_summary = (
        detailed_df[(detailed_df["component_type"] == "sgen") & (detailed_df["generation_type"] == "wind")]
        .groupby(["time_step", "bus_index", "bus_name"], as_index=False)["p_mw"]
        .sum()
        .rename(columns={"p_mw": "wind_mw"})
    )
    other_gen_summary = (
        detailed_df[
            ((detailed_df["component_type"] == "sgen") & (detailed_df["generation_type"] == "other"))
            | (detailed_df["component_type"] == "gen")
        ]
        .groupby(["time_step", "bus_index", "bus_name"], as_index=False)["p_mw"]
        .sum()
        .rename(columns={"p_mw": "other_gen_mw"})
    )

    bus_time_steps = pd.DataFrame({
        "time_step": pd.Index(time_steps, dtype=int).repeat(len(subnet_bus_indices)),
        "bus_index": np.tile(np.array(subnet_bus_indices, dtype=int), len(time_steps)),
    })
    bus_time_steps["bus_name"] = [safe_name(net.bus.at[idx, "name"], f"bus_{idx}") for idx in bus_time_steps["bus_index"]]

    time_series_df = bus_time_steps.merge(load_summary, on=["time_step", "bus_index", "bus_name"], how="left")
    time_series_df = time_series_df.merge(pv_summary, on=["time_step", "bus_index", "bus_name"], how="left")
    time_series_df = time_series_df.merge(wind_summary, on=["time_step", "bus_index", "bus_name"], how="left")
    time_series_df = time_series_df.merge(other_gen_summary, on=["time_step", "bus_index", "bus_name"], how="left")
    for column in ["load_mw", "pv_mw", "wind_mw", "other_gen_mw"]:
        time_series_df[column] = time_series_df[column].fillna(0.0)

    time_series_df["total_generation_mw"] = (
        time_series_df["pv_mw"] + time_series_df["wind_mw"] + time_series_df["other_gen_mw"]
    )
    time_series_df["net_injection_mw"] = time_series_df["total_generation_mw"] - time_series_df["load_mw"]

    summary_df = (
        time_series_df.groupby(["bus_index", "bus_name"], as_index=False).agg(
            mean_load_mw=("load_mw", "mean"),
            peak_load_mw=("load_mw", "max"),
            mean_pv_mw=("pv_mw", "mean"),
            peak_pv_mw=("pv_mw", "max"),
            mean_wind_mw=("wind_mw", "mean"),
            peak_wind_mw=("wind_mw", "max"),
            mean_other_gen_mw=("other_gen_mw", "mean"),
            peak_other_gen_mw=("other_gen_mw", "max"),
            mean_total_generation_mw=("total_generation_mw", "mean"),
            peak_total_generation_mw=("total_generation_mw", "max"),
            mean_net_injection_mw=("net_injection_mw", "mean"),
            peak_net_injection_mw=("net_injection_mw", "max"),
        )
    )
    summary_df["classification"] = summary_df.apply(
        lambda row: classify_subnet_bus(
            row["mean_load_mw"],
            row["mean_total_generation_mw"],
            row["mean_pv_mw"],
            row["mean_wind_mw"],
        ),
        axis=1,
    )
    return summary_df[columns].sort_values(["bus_index", "bus_name"]).reset_index(drop=True)


def load_aitolahti_weather(time_steps):
    if not AITOLAHTI_WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"Aitolahti weather file not found: {AITOLAHTI_WEATHER_CSV}. Run dlr_weather_tampere_full.py first."
        )
    weather_df = pd.read_csv(AITOLAHTI_WEATHER_CSV).reset_index(drop=True)
    weather_df["time_step"] = weather_df.index.astype(int)
    weather_df = weather_df[weather_df["time_step"].isin(time_steps)].copy()
    required = ["ambient_temp_c", "wind_speed_mps", "wind_angle_deg", "solar_wm2"]
    missing = [column for column in required if column not in weather_df.columns]
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


def build_subnet_dlr_timeseries(line_summary, line_i_ka, weather_df):
    if line_summary.empty or line_i_ka.empty or weather_df.empty:
        return pd.DataFrame()
    weather_lookup = weather_df.set_index("time_step")
    records = []
    for _, line_row in line_summary.iterrows():
        current_rows = line_i_ka[line_i_ka["line_index"] == int(line_row["line_index"])]
        for _, current_row in current_rows.iterrows():
            time_step = int(current_row["time_step"])
            if time_step not in weather_lookup.index:
                continue
            weather_row = weather_lookup.loc[time_step]
            dlr_ka = ieee738_ampacity_ka(line_row, weather_row)
            utilization = np.nan if dlr_ka <= 0.0 else 100.0 * float(current_row["i_ka"]) / dlr_ka
            records.append(
                {
                    "time_step": time_step,
                    "hour_of_day": float(weather_row["hour_of_day"]),
                    "line_index": int(line_row["line_index"]),
                    "name": safe_name(line_row["line_name"], f"line_{int(line_row['line_index'])}"),
                    "ambient_temp_c": float(weather_row["ambient_temp_c"]),
                    "wind_speed_mps": float(weather_row["wind_speed_mps"]),
                    "wind_angle_deg": float(weather_row["wind_angle_deg"]),
                    "solar_wm2": float(weather_row["solar_wm2"]),
                    "actual_i_ka": float(current_row["i_ka"]),
                    "dlr_ka": float(dlr_ka),
                    "dlr_utilization_percent": float(utilization) if not np.isnan(utilization) else np.nan,
                }
            )
    return pd.DataFrame(records)


def build_loading_comparison_timeseries(line_loading, dlr_df):
    if line_loading.empty or dlr_df.empty:
        return pd.DataFrame()

    no_dlr = line_loading.copy()
    no_dlr["hour_of_day"] = no_dlr["time_step"] * TIME_STEP_HOURS
    no_dlr = no_dlr.rename(columns={"loading_percent": "loading_percent_without_dlr"})

    with_dlr = dlr_df[["time_step", "line_index", "name", "hour_of_day", "dlr_utilization_percent"]].copy()
    with_dlr = with_dlr.rename(columns={"dlr_utilization_percent": "loading_percent_with_dlr"})

    merged = no_dlr.merge(
        with_dlr,
        on=["time_step", "line_index", "name", "hour_of_day"],
        how="left",
    )
    merged["dlr_benefit_percent_points"] = (
        merged["loading_percent_without_dlr"] - merged["loading_percent_with_dlr"]
    )
    return merged


def build_conductor_validation_summary(line_summary):
    if line_summary.empty:
        return pd.DataFrame()

    summary = line_summary.copy()
    summary["resistance_match"] = np.isclose(
        summary["resistance_20c_ohm_per_km"],
        summary["r_ohm_per_km"],
        equal_nan=True,
    )
    summary["parallel_match"] = summary["parallel_count"].astype(int) == summary["parallel"].fillna(1).astype(int)
    summary["std_type_used_for_dlr"] = summary["std_type"].fillna("").astype(str)

    ordered = [
        "line_index", "line_name", "std_type", "conductor_id",
        "from_bus_name", "to_bus_name",
        "parallel", "parallel_count", "parallel_match",
        "r_ohm_per_km", "resistance_20c_ohm_per_km", "resistance_match",
        "x_ohm_per_km", "max_i_ka", "q_mm2", "diameter_m_est",
        "alpha_per_c", "emissivity", "absorptivity",
        "line_azimuth_deg", "dlr_conductor_temp_c"
    ]
    return summary[[col for col in ordered if col in summary.columns]].copy()

def build_subnet_injection_timeseries(net, abs_vals, time_steps, subnet_bus_indices):
    frames = []
    for element_type in ("load", "sgen", "gen"):
        frame = aggregate_abs_values_by_bus(net, abs_vals, time_steps, element_type, "bus", subnet_bus_indices, element_type)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["time_step", "bus_index", "bus_name", "source_type", "p_mw", "q_mvar"])
    return pd.concat(frames, ignore_index=True)


def aggregate_abs_values_by_bus(net, abs_vals, time_steps, element_type, bus_column, subnet_bus_indices, source_label):
    key_p = (element_type, "p_mw")
    if key_p not in abs_vals:
        return pd.DataFrame()
    meta = getattr(net, element_type)
    if meta.empty:
        return pd.DataFrame()
    p_df = abs_vals[key_p].loc[time_steps].copy()
    p_df.columns = [int(col) for col in p_df.columns]
    p_long = p_df.reset_index(names="time_step").melt(id_vars=["time_step"], var_name="element_index", value_name="p_mw")
    key_q = (element_type, "q_mvar")
    if key_q in abs_vals:
        q_df = abs_vals[key_q].loc[time_steps].copy()
        q_df.columns = [int(col) for col in q_df.columns]
        q_long = q_df.reset_index(names="time_step").melt(id_vars=["time_step"], var_name="element_index", value_name="q_mvar")
        merged = p_long.merge(q_long, on=["time_step", "element_index"], how="left")
    else:
        merged = p_long.copy()
        merged["q_mvar"] = 0.0
    meta_export = meta[[bus_column]].copy()
    meta_export["element_index"] = meta_export.index.astype(int)
    merged = merged.merge(meta_export, on="element_index", how="left")
    merged = merged[merged[bus_column].isin(subnet_bus_indices)].copy()
    if merged.empty:
        return pd.DataFrame()
    grouped = merged.groupby(["time_step", bus_column], as_index=False)[["p_mw", "q_mvar"]].sum().rename(columns={bus_column: "bus_index"})
    grouped["bus_name"] = [safe_name(net.bus.at[idx, "name"], f"bus_{idx}") for idx in grouped["bus_index"]]
    grouped["source_type"] = source_label
    return grouped[["time_step", "bus_index", "bus_name", "source_type", "p_mw", "q_mvar"]]

def summarize_metric(df, index_column, value_column):
    if df.empty:
        return pd.DataFrame()
    return df.groupby(index_column)[value_column].agg(["min", "max", "mean"])


def summarize_bus_vm(bus_vm):
    if bus_vm.empty:
        return pd.DataFrame()
    return bus_vm.groupby("bus_index")["vm_pu"].agg(["min", "max", "mean"])



def build_subnet_generation_analysis(injection_df, bus_vm):
    columns = [
        "time_step",
        "total_generation_mw",
        "total_load_mw",
        "net_injection_mw",
        "max_vm_pu",
    ]
    if injection_df.empty or bus_vm.empty:
        return pd.DataFrame(columns=columns)

    generation = (
        injection_df[injection_df["source_type"].isin(["sgen", "gen"])]
        .groupby("time_step", as_index=False)["p_mw"]
        .sum()
        .rename(columns={"p_mw": "total_generation_mw"})
    )
    load = (
        injection_df[injection_df["source_type"] == "load"]
        .groupby("time_step", as_index=False)["p_mw"]
        .sum()
        .rename(columns={"p_mw": "total_load_mw"})
    )
    max_vm = (
        bus_vm.groupby("time_step", as_index=False)["vm_pu"]
        .max()
        .rename(columns={"vm_pu": "max_vm_pu"})
    )

    analysis_df = generation.merge(load, on="time_step", how="outer").merge(max_vm, on="time_step", how="outer")
    analysis_df["total_generation_mw"] = analysis_df["total_generation_mw"].fillna(0.0)
    analysis_df["total_load_mw"] = analysis_df["total_load_mw"].fillna(0.0)
    analysis_df["net_injection_mw"] = analysis_df["total_generation_mw"] - analysis_df["total_load_mw"]
    return analysis_df.sort_values("time_step")[columns].reset_index(drop=True)


def summarize_subnet_overvoltage_correlation(analysis_df, season_name):
    if analysis_df.empty:
        print(f"Subnet export/voltage analysis for {season_name}: no data available.")
        return

    positive_export_share = (analysis_df["net_injection_mw"] > 0.0).mean()
    peak_row = analysis_df.loc[analysis_df["max_vm_pu"].idxmax()]
    summary_parts = [
        f"Subnet export/voltage analysis for {season_name}:",
        f"peak max voltage = {peak_row['max_vm_pu']:.4f} pu at time step {int(peak_row['time_step'])}",
        f"net injection at voltage peak = {peak_row['net_injection_mw']:.2f} MW",
        f"share of steps with net export (>0 MW) = {positive_export_share:.1%}",
    ]

    for threshold in (1.05, 1.10):
        exceedances = analysis_df[analysis_df["max_vm_pu"] > threshold].copy()
        if exceedances.empty:
            summary_parts.append(f"no steps above {threshold:.2f} pu")
            continue
        mean_export = exceedances["net_injection_mw"].mean()
        max_export = exceedances["net_injection_mw"].max()
        summary_parts.append(
            f"{len(exceedances)} steps above {threshold:.2f} pu with mean net injection {mean_export:.2f} MW "
            f"and peak net injection {max_export:.2f} MW"
        )

    print(" | ".join(summary_parts))


def build_subnet_voltage_summary(bus_vm):
    if bus_vm.empty:
        return pd.DataFrame(columns=["time_step", "max_vm_pu", "mean_vm_pu"])
    return (
        bus_vm.groupby("time_step", as_index=False)["vm_pu"]
        .agg(max_vm_pu="max", mean_vm_pu="mean")
        .sort_values("time_step")
        .reset_index(drop=True)
    )


def build_subnet_loading_summary(loading_comparison_df):
    columns = [
        "time_step",
        "max_loading_without_dlr_percent",
        "mean_loading_without_dlr_percent",
        "max_loading_with_dlr_percent",
        "mean_loading_with_dlr_percent",
        "max_dlr_benefit_percent_points",
        "mean_dlr_benefit_percent_points",
    ]
    if loading_comparison_df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        loading_comparison_df.groupby("time_step", as_index=False).agg(
            max_loading_without_dlr_percent=("loading_percent_without_dlr", "max"),
            mean_loading_without_dlr_percent=("loading_percent_without_dlr", "mean"),
            max_loading_with_dlr_percent=("loading_percent_with_dlr", "max"),
            mean_loading_with_dlr_percent=("loading_percent_with_dlr", "mean"),
            max_dlr_benefit_percent_points=("dlr_benefit_percent_points", "max"),
            mean_dlr_benefit_percent_points=("dlr_benefit_percent_points", "mean"),
        )
    )
    return summary.sort_values("time_step").reset_index(drop=True)


def attach_rank_labels(top_df, rank_prefix):
    ranked = top_df.copy().reset_index(drop=True)
    ranked["rank_label"] = [f"{rank_prefix}{idx + 1}" for idx in range(len(ranked))]
    return ranked


def top_n_time_points(df, value_column, n=10):
    if df.empty or value_column not in df.columns:
        return pd.DataFrame()
    top_df = df.nlargest(min(n, len(df)), value_column).copy()
    return top_df.sort_values(value_column, ascending=False).reset_index(drop=True)


def annotate_time_points(ax, points_df, x_column, y_column, color):
    if points_df.empty:
        return
    ax.scatter(points_df[x_column], points_df[y_column], color=color, s=28, zorder=5)
    for _, row in points_df.iterrows():
        ax.annotate(
            row["rank_label"],
            (row[x_column], row[y_column]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=color,
        )


def build_seasonal_dashboard_summary(voltage_summary, loading_summary, generation_analysis_df):
    merged = voltage_summary.merge(loading_summary, on="time_step", how="outer")
    if not generation_analysis_df.empty:
        merged = merged.merge(
            generation_analysis_df[["time_step", "net_injection_mw", "total_generation_mw", "total_load_mw"]],
            on="time_step",
            how="left",
        )
    return merged.sort_values("time_step").reset_index(drop=True)


def plot_subnet_bus_generation_mix(bus_mix_df, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    if bus_mix_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    plot_df = bus_mix_df.copy().sort_values("mean_total_generation_mw", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(plot_df))

    mean_ax = axes[0]
    mean_ax.barh(y_positions, plot_df["mean_load_mw"], color="#7f7f7f", alpha=0.35, label="Mean load")
    mean_ax.barh(y_positions, plot_df["mean_pv_mw"], color="#ffbb78", label="Mean PV")
    mean_ax.barh(y_positions, plot_df["mean_wind_mw"], left=plot_df["mean_pv_mw"], color="#17becf", label="Mean wind")
    mean_ax.barh(
        y_positions,
        plot_df["mean_other_gen_mw"],
        left=plot_df["mean_pv_mw"] + plot_df["mean_wind_mw"],
        color="#2ca02c",
        label="Mean other generation",
    )
    mean_ax.scatter(plot_df["mean_net_injection_mw"], y_positions, color="#1f77b4", marker="D", s=36, label="Mean net injection")
    mean_ax.set_title("Subnet bus generation mix from seasonal time-series means")
    mean_ax.set_xlabel("Mean power [MW]")
    mean_ax.set_yticks(y_positions)
    mean_ax.set_yticklabels(plot_df["bus_name"])
    mean_ax.grid(True, axis="x", alpha=0.3)
    mean_ax.legend(fontsize=8, ncol=2)

    peak_ax = axes[1]
    peak_ax.barh(y_positions, plot_df["peak_load_mw"], color="#7f7f7f", alpha=0.35, label="Peak load")
    peak_ax.barh(y_positions, plot_df["peak_pv_mw"], color="#ffbb78", label="Peak PV")
    peak_ax.barh(y_positions, plot_df["peak_wind_mw"], left=plot_df["peak_pv_mw"], color="#17becf", label="Peak wind")
    peak_ax.barh(
        y_positions,
        plot_df["peak_other_gen_mw"],
        left=plot_df["peak_pv_mw"] + plot_df["peak_wind_mw"],
        color="#2ca02c",
        label="Peak other generation",
    )
    peak_ax.scatter(plot_df["peak_net_injection_mw"], y_positions, color="#d62728", marker="o", s=36, label="Peak net injection")
    for idx, row in plot_df.iterrows():
        peak_ax.text(
            max(row["peak_total_generation_mw"], row["peak_load_mw"]) + max(plot_df["peak_total_generation_mw"].max(), plot_df["peak_load_mw"].max(), 1.0) * 0.01,
            y_positions[idx],
            row["classification"],
            va="center",
            fontsize=8,
        )
    peak_ax.set_title("Subnet bus generation mix from seasonal peak values")
    peak_ax.set_xlabel("Peak power [MW]")
    peak_ax.set_yticks(y_positions)
    peak_ax.set_yticklabels(plot_df["bus_name"])
    peak_ax.grid(True, axis="x", alpha=0.3)
    peak_ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grouped_series(ax, df, x_column, label_column, value_column, ylabel, title):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    for label, group in df.groupby(label_column):
        group = group.sort_values(x_column)
        ax.plot(group[x_column], group[value_column], label=label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)


def plot_current_vs_dlr(ax, line_i_ka, dlr_df):
    ax.set_title("Subnet line current vs dynamic rating")
    ax.set_ylabel("Current / DLR [kA]")
    if line_i_ka.empty or dlr_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    current_plot = line_i_ka.assign(hour_of_day=line_i_ka["time_step"] * TIME_STEP_HOURS)
    merged = current_plot.merge(
        dlr_df[["time_step", "line_index", "dlr_ka"]],
        on=["time_step", "line_index"],
        how="left",
    )

    for label, group in merged.groupby("name"):
        group = group.sort_values("hour_of_day")
        ax.plot(group["hour_of_day"], group["i_ka"], label=f"{label} I")
        ax.plot(group["hour_of_day"], group["dlr_ka"], linestyle="--", label=f"{label} DLR")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def plot_loading_comparison(ax, loading_comparison_df):
    ax.set_title("Subnet loading comparison: without DLR vs with DLR")
    ax.set_ylabel("Loading [%]")
    if loading_comparison_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    for label, group in loading_comparison_df.groupby("name"):
        group = group.sort_values("hour_of_day")
        ax.plot(group["hour_of_day"], group["loading_percent_without_dlr"], label=f"{label} no DLR")
        ax.plot(group["hour_of_day"], group["loading_percent_with_dlr"], linestyle="--", label=f"{label} with DLR")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def plot_subnet_timeseries(bus_vm, line_i_ka, line_loading, dlr_df, output_path):
    loading_comparison_df = build_loading_comparison_timeseries(line_loading, dlr_df)
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    plot_grouped_series(axes[0], bus_vm.assign(hour_of_day=bus_vm["time_step"] * TIME_STEP_HOURS), "hour_of_day", "name", "vm_pu", "Voltage [p.u.]", "Subnet bus voltages")
    plot_current_vs_dlr(axes[1], line_i_ka, dlr_df)
    plot_loading_comparison(axes[2], loading_comparison_df)
    axes[2].set_xlabel("Hour of day")
    axes[2].set_xlim(0.0, 24.0)
    axes[2].set_xticks(np.arange(0.0, 25.0, 2.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_subnet_topology(net, subnet_bus_indices, line_summary, trafo_summary, bus_vm, line_i_ka, line_loading, trafo_loading, dlr_df, output_path):
    graph = nx.Graph()
    for bus_idx in subnet_bus_indices:
        graph.add_node(int(bus_idx))
    for _, row in line_summary.iterrows():
        graph.add_edge(int(row["from_bus"]), int(row["to_bus"]), edge_type="line")
    for _, row in trafo_summary.iterrows():
        graph.add_edge(int(row["hv_bus"]), int(row["lv_bus"]), edge_type="trafo")
    pos = get_subnet_positions(net, subnet_bus_indices, graph)
    fig, ax = plt.subplots(figsize=(14, 10))
    line_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["edge_type"] == "line"]
    trafo_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["edge_type"] == "trafo"]
    if line_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=line_edges, width=2.0, edge_color="#1f77b4", ax=ax)
    if trafo_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=trafo_edges, width=2.0, edge_color="#d35400", style="dashed", ax=ax)
    ext_grid_buses = set(net.ext_grid["bus"].tolist())
    node_colors = ["#f5b041" if bus_idx in ext_grid_buses else "#85c1e9" for bus_idx in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color=node_colors, edgecolors="black", ax=ax)

    bus_stats = summarize_bus_vm(bus_vm)
    node_labels = {}
    for bus_idx in graph.nodes:
        bus_name = safe_name(net.bus.at[bus_idx, "name"], f"bus_{bus_idx}")
        if bus_idx in bus_stats.index:
            stats = bus_stats.loc[bus_idx]
            node_labels[bus_idx] = f"{bus_name}\nVmin={stats['min']:.3f} pu\nVmax={stats['max']:.3f} pu"
        else:
            node_labels[bus_idx] = bus_name
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, ax=ax)

    line_i_stats = summarize_metric(line_i_ka, "line_index", "i_ka")
    line_loading_stats = summarize_metric(line_loading, "line_index", "loading_percent")
    dlr_stats = summarize_metric(dlr_df, "line_index", "dlr_ka")
    edge_labels = {}
    for _, row in line_summary.iterrows():
        edge = (int(row["from_bus"]), int(row["to_bus"]))
        line_idx = int(row["line_index"])
        parts = [safe_name(row["line_name"], f"line_{line_idx}")]
        if line_idx in line_i_stats.index:
            parts.append(f"Imax={line_i_stats.at[line_idx, 'max']:.3f} kA")
        if line_idx in dlr_stats.index:
            parts.append(f"DLRavg={dlr_stats.at[line_idx, 'mean']:.3f} kA")
        if line_idx in line_loading_stats.index:
            parts.append(f"Lmax={line_loading_stats.at[line_idx, 'max']:.1f}%")
        edge_labels[edge] = "\n".join(parts)
    if not trafo_summary.empty and not trafo_loading.empty:
        trafo_stats = summarize_metric(trafo_loading, "trafo_index", "loading_percent")
        for _, row in trafo_summary.iterrows():
            edge = (int(row["hv_bus"]), int(row["lv_bus"]))
            trafo_idx = int(row["trafo_index"])
            label = safe_name(row["trafo_name"], f"trafo_{trafo_idx}")
            if trafo_idx in trafo_stats.index:
                label = f"{label}\nLmax={trafo_stats.at[trafo_idx, 'max']:.1f}%"
            edge_labels[edge] = label
    for (u, v), label in edge_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2.0
        y = (pos[u][1] + pos[v][1]) / 2.0
        ax.text(x, y, label, fontsize=7, ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))
    ax.set_title("Subnet power flow and DLR summary")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_subnet_results(net, abs_vals, time_steps, output_path):
    subnet_dir = os.path.join(output_path, SUBNET_DIR_NAME)
    os.makedirs(subnet_dir, exist_ok=True)
    bus_map = get_subnet_bus_map(net, SUBNET_BUS_NAMES)
    subnet_bus_indices = list(bus_map.values())
    subnet_bus_set = set(subnet_bus_indices)
    subnet_lines = net.line[net.line["from_bus"].isin(subnet_bus_set) & net.line["to_bus"].isin(subnet_bus_set)].copy()
    subnet_trafos = net.trafo[net.trafo["hv_bus"].isin(subnet_bus_set) & net.trafo["lv_bus"].isin(subnet_bus_set)].copy()
    weather_df = load_aitolahti_weather(time_steps)

    bus_summary = build_subnet_bus_summary(net, subnet_bus_indices)
    line_summary = build_subnet_line_summary(net, subnet_lines)
    trafo_summary = build_subnet_trafo_summary(net, subnet_trafos)
    load_summary = build_subnet_component_summary(net, "load", subnet_bus_indices, "load")
    sgen_summary = build_subnet_component_summary(net, "sgen", subnet_bus_indices, "sgen")
    gen_summary = build_subnet_component_summary(net, "gen", subnet_bus_indices, "gen")
    bus_summary.to_csv(os.path.join(subnet_dir, "subnet_bus_summary.csv"), index=False)
    line_summary.to_csv(os.path.join(subnet_dir, "subnet_line_summary.csv"), index=False)
    trafo_summary.to_csv(os.path.join(subnet_dir, "subnet_trafo_summary.csv"), index=False)
    load_summary.to_csv(os.path.join(subnet_dir, "subnet_load_summary.csv"), index=False)
    sgen_summary.to_csv(os.path.join(subnet_dir, "subnet_sgen_summary.csv"), index=False)
    gen_summary.to_csv(os.path.join(subnet_dir, "subnet_gen_summary.csv"), index=False)
    weather_df.to_csv(os.path.join(subnet_dir, "aitolahti_weather_timeseries.csv"), index=False)

    bus_vm = filter_named_results(os.path.join(output_path, "bus_vm_pu_named.csv"), "bus_index", subnet_bus_indices)
    bus_vm.to_csv(os.path.join(subnet_dir, "subnet_bus_vm_pu.csv"), index=False)
    line_indices = subnet_lines.index.astype(int).tolist()
    line_i_ka = filter_named_results(os.path.join(output_path, "line_i_ka_named.csv"), "line_index", line_indices)
    line_loading = filter_named_results(os.path.join(output_path, "line_loading_percent_named.csv"), "line_index", line_indices)
    line_i_ka.to_csv(os.path.join(subnet_dir, "subnet_line_i_ka.csv"), index=False)
    line_loading.to_csv(os.path.join(subnet_dir, "subnet_line_loading_percent.csv"), index=False)
    trafo_loading = pd.DataFrame()
    if not subnet_trafos.empty:
        trafo_loading = filter_named_results(os.path.join(output_path, "trafo_loading_percent_named.csv"), "trafo_index", subnet_trafos.index.astype(int).tolist())
        trafo_loading.to_csv(os.path.join(subnet_dir, "subnet_trafo_loading_percent.csv"), index=False)
    injection_df = build_subnet_injection_timeseries(net, abs_vals, time_steps, subnet_bus_indices)
    injection_df.to_csv(os.path.join(subnet_dir, "subnet_bus_injections.csv"), index=False)
    dlr_df = build_subnet_dlr_timeseries(line_summary, line_i_ka, weather_df)
    loading_comparison_df = build_loading_comparison_timeseries(line_loading, dlr_df)
    conductor_validation_df = build_conductor_validation_summary(line_summary)

    dlr_df.to_csv(os.path.join(subnet_dir, "subnet_line_dlr_ka.csv"), index=False)
    loading_comparison_df.to_csv(os.path.join(subnet_dir, "subnet_line_loading_comparison.csv"), index=False)
    conductor_validation_df.to_csv(os.path.join(subnet_dir, "subnet_dlr_conductor_check.csv"), index=False)

    plot_subnet_timeseries(bus_vm, line_i_ka, line_loading, dlr_df, os.path.join(subnet_dir, "subnet_timeseries.png"))
    plot_subnet_topology(net, subnet_bus_indices, line_summary, trafo_summary, bus_vm, line_i_ka, line_loading, trafo_loading, dlr_df, os.path.join(subnet_dir, "subnet_topology_results.png"))


from collections import OrderedDict

import matplotlib.dates as mdates



# Each season is defined explicitly from calendar months so the seasonal
# studies are reproducible and easy to audit.
SEASON_MONTHS = OrderedDict(
    [
        ("winter", [1, 2, 12]),
        ("spring", [3, 4, 5]),
        ("summer", [6, 7, 8]),
        ("autumn", [9, 10, 11]),
    ]
)

# A dedicated root folder keeps the seasonal workflow separated from the
# existing 1-day and 1-year outputs.
SEASONAL_OUTPUT_ROOT = "results_1year_seasons"

# The full-network overview plots can become unreadable if every line is shown,
# so the seasonal script highlights the most stressed lines only.
TOP_LINE_COUNT = 20


# Loading the weather file once gives us the calendar timestamps that are used
# to split the annual study into four independent seasonal studies.
def load_weather_calendar(max_steps):
    if not AITOLAHTI_WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"Aitolahti weather file not found: {AITOLAHTI_WEATHER_CSV}. "
            "Run dlr_weather_tampere_full.py first."
        )

    weather_df = pd.read_csv(AITOLAHTI_WEATHER_CSV).reset_index(drop=True)
    weather_df["time_step"] = weather_df.index.astype(int)

    if "time_utc" in weather_df.columns:
        weather_df["time_utc"] = pd.to_datetime(weather_df["time_utc"], errors="coerce", utc=True)
    else:
        weather_df["time_utc"] = pd.date_range(
            start="2023-01-01 00:00:00",
            periods=len(weather_df),
            freq="15min",
            tz="UTC",
        )

    weather_df = weather_df[weather_df["time_step"] < max_steps].copy()
    if weather_df["time_utc"].isna().any():
        raise ValueError("The weather file contains invalid 'time_utc' values, so seasons cannot be assigned reliably.")

    weather_df["month"] = weather_df["time_utc"].dt.month.astype(int)
    return weather_df


# The seasonal studies are run independently, but this helper ensures every
# season receives exactly the time steps that belong to its calendar months.
def build_season_time_steps(weather_calendar):
    season_steps = OrderedDict()
    for season_name, months in SEASON_MONTHS.items():
        steps = weather_calendar.loc[weather_calendar["month"].isin(months), "time_step"].astype(int).tolist()
        season_steps[season_name] = steps
    return season_steps


# A simple lookup is reused many times when the exported CSVs and plots need a
# real timestamp instead of a raw annual time-step index.
def build_time_lookup(weather_calendar):
    return weather_calendar.set_index("time_step")["time_utc"]


# The exported CSVs are easier to interpret when every row carries the real
# timestamp of the simulated operating point.
def attach_time_utc(df, time_lookup):
    if df.empty or "time_step" not in df.columns:
        return df.copy()
    enriched = df.copy()
    enriched["time_utc"] = pd.to_datetime(enriched["time_step"].map(time_lookup), utc=True)
    return enriched


# Seasonal plots should only show the participating months, not the empty gap
# between February and December in the winter selection. A synthetic seasonal
# axis keeps the chosen months contiguous while preserving their month labels.
def build_season_axis_lookup(time_steps, time_lookup):
    axis_df = pd.DataFrame({"time_step": pd.Index(time_steps, dtype=int).unique()})
    axis_df["time_utc"] = pd.to_datetime(axis_df["time_step"].map(time_lookup), utc=True)
    axis_df = axis_df.dropna().sort_values("time_utc").reset_index(drop=True)
    axis_df["season_position"] = np.arange(len(axis_df))
    axis_df["month_label"] = axis_df["time_utc"].dt.strftime("%b")
    month_ticks = axis_df.groupby("month_label", sort=False)["season_position"].min()
    return axis_df[["time_step", "season_position"]], month_ticks.index.tolist(), month_ticks.tolist()


# Every plotted dataframe gets the same synthetic seasonal x-axis so the three
# subnet panels stay aligned and only the selected season months are displayed.
def attach_season_position(df, season_axis_lookup):
    if df.empty:
        return df.copy()
    return df.merge(season_axis_lookup, on="time_step", how="left")


# This x-axis formatter labels the compressed seasonal axis with the months that
# are actually part of the selected season.
def format_season_axis(ax, month_labels, month_positions):
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels)
    ax.grid(True, alpha=0.3)


# This plotting helper is used for both bus voltages and other seasonal series.
def plot_datetime_grouped_series(ax, df, label_column, value_column, ylabel, title, month_labels, month_positions):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    for label, group in df.groupby(label_column):
        ordered = group.sort_values("season_position")
        ax.plot(ordered["season_position"], ordered[value_column], label=label)

    format_season_axis(ax, month_labels, month_positions)
    ax.legend(fontsize=8, ncol=2)


# The current-vs-DLR figure is retained from the yearly workflow, but here the
# x-axis spans the whole season instead of a single day.
def plot_seasonal_current_vs_dlr(ax, line_i_ka, dlr_df, month_labels, month_positions):
    ax.set_title("Subnet line current vs dynamic rating")
    ax.set_ylabel("Current / DLR [kA]")
    if line_i_ka.empty or dlr_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    merged = line_i_ka.merge(
        dlr_df[["time_step", "line_index", "dlr_ka"]],
        on=["time_step", "line_index"],
        how="left",
    )

    for label, group in merged.groupby("name"):
        ordered = group.sort_values("season_position")
        ax.plot(ordered["season_position"], ordered["i_ka"], label=f"{label} I")
        ax.plot(ordered["season_position"], ordered["dlr_ka"], linestyle="--", label=f"{label} DLR")

    format_season_axis(ax, month_labels, month_positions)
    ax.legend(fontsize=7, ncol=2)


# This plot shows the difference between the normal loading and the DLR-based
# loading utilization over the whole season.
def plot_seasonal_loading_comparison(ax, loading_summary_df, month_labels, month_positions, highlighted_points):
    ax.set_title("Subnet line loading summary")
    ax.set_ylabel("Loading [%]")
    if loading_summary_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(
        loading_summary_df["season_position"],
        loading_summary_df["max_loading_without_dlr_percent"],
        color="#d62728",
        label="Max without DLR",
    )
    ax.plot(
        loading_summary_df["season_position"],
        loading_summary_df["max_loading_with_dlr_percent"],
        color="#2ca02c",
        linestyle="--",
        label="Max with DLR",
    )
    ax.plot(
        loading_summary_df["season_position"],
        loading_summary_df["mean_loading_without_dlr_percent"],
        color="#ff9896",
        alpha=0.9,
        label="Mean without DLR",
    )
    ax.plot(
        loading_summary_df["season_position"],
        loading_summary_df["mean_loading_with_dlr_percent"],
        color="#98df8a",
        linestyle=":",
        alpha=0.9,
        label="Mean with DLR",
    )
    annotate_time_points(
        ax,
        highlighted_points,
        "season_position",
        "max_loading_without_dlr_percent",
        "#d62728",
    )
    format_season_axis(ax, month_labels, month_positions)
    ax.legend(fontsize=8, ncol=2)


def plot_seasonal_dlr_benefit(ax, loading_summary_df, month_labels, month_positions, highlighted_points):
    ax.set_title("Direct DLR benefit on subnet loading")
    ax.set_ylabel("Benefit [percentage points]")
    if loading_summary_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.fill_between(
        loading_summary_df["season_position"],
        0.0,
        loading_summary_df["max_dlr_benefit_percent_points"],
        color="#9edae5",
        alpha=0.55,
        label="Max DLR benefit",
    )
    ax.plot(
        loading_summary_df["season_position"],
        loading_summary_df["mean_dlr_benefit_percent_points"],
        color="#1f77b4",
        linewidth=1.6,
        label="Mean DLR benefit",
    )
    annotate_time_points(
        ax,
        highlighted_points,
        "season_position",
        "max_dlr_benefit_percent_points",
        "#1f77b4",
    )
    format_season_axis(ax, month_labels, month_positions)
    ax.legend(fontsize=8)


def plot_subnet_generation_voltage_analysis(analysis_df, time_steps, time_lookup, season_name, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    if analysis_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    season_axis_lookup, month_labels, month_positions = build_season_axis_lookup(time_steps, time_lookup)
    plot_df = attach_season_position(analysis_df, season_axis_lookup).sort_values("season_position")
    top_voltage_points = attach_rank_labels(top_n_time_points(plot_df, "max_vm_pu", n=10), "V")

    voltage_ax = axes[0]
    injection_ax = voltage_ax.twinx()
    voltage_ax.plot(plot_df["season_position"], plot_df["max_vm_pu"], color="#d62728", label="Max bus voltage")
    injection_ax.plot(plot_df["season_position"], plot_df["net_injection_mw"], color="#1f77b4", alpha=0.8, label="Net injection")
    for threshold, color in ((1.05, "#ff7f0e"), (1.10, "#9467bd")):
        voltage_ax.axhline(threshold, color=color, linewidth=1.0, linestyle="--", label=f"{threshold:.2f} pu")
    annotate_time_points(voltage_ax, top_voltage_points, "season_position", "max_vm_pu", "#d62728")
    voltage_ax.set_title(f"Subnet net injection and voltage - {season_name.title()}")
    voltage_ax.set_ylabel("Max voltage [p.u.]")
    injection_ax.set_ylabel("Net injection [MW]")
    format_season_axis(voltage_ax, month_labels, month_positions)
    handles1, labels1 = voltage_ax.get_legend_handles_labels()
    handles2, labels2 = injection_ax.get_legend_handles_labels()
    voltage_ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, ncol=2, loc="upper left")

    scatter = axes[1].scatter(
        plot_df["net_injection_mw"],
        plot_df["max_vm_pu"],
        c=plot_df["season_position"],
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    for _, row in top_voltage_points.iterrows():
        axes[1].annotate(
            row["rank_label"],
            (row["net_injection_mw"], row["max_vm_pu"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#d62728",
        )
    for threshold, color in ((1.05, "#ff7f0e"), (1.10, "#9467bd")):
        axes[1].axhline(threshold, color=color, linewidth=1.0, linestyle="--")
    axes[1].axvline(0.0, color="black", linewidth=1.0, linestyle=":")
    axes[1].set_title("Net injection vs maximum bus voltage")
    axes[1].set_xlabel("Net injection [MW]")
    axes[1].set_ylabel("Max voltage [p.u.]")
    axes[1].grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=axes[1], pad=0.02)
    cbar.set_label("Season position")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# The seasonal subnet figure now focuses on system-level voltage, loading, and
# DLR benefit so the seasonal behavior is easier to interpret than many line overlays.
def plot_subnet_timeseries_for_season(
    bus_vm,
    line_i_ka,
    line_loading,
    dlr_df,
    generation_analysis_df,
    time_lookup,
    time_steps,
    season_name,
    output_path,
):
    bus_vm = attach_time_utc(bus_vm, time_lookup)
    line_loading = attach_time_utc(line_loading, time_lookup)
    loading_comparison_df = build_loading_comparison_timeseries(line_loading, dlr_df)
    loading_comparison_df = attach_time_utc(loading_comparison_df, time_lookup)
    generation_analysis_df = attach_time_utc(generation_analysis_df, time_lookup)

    voltage_summary_df = attach_time_utc(build_subnet_voltage_summary(bus_vm), time_lookup)
    loading_summary_df = attach_time_utc(build_subnet_loading_summary(loading_comparison_df), time_lookup)
    dashboard_df = build_seasonal_dashboard_summary(voltage_summary_df, loading_summary_df, generation_analysis_df)

    season_axis_lookup, month_labels, month_positions = build_season_axis_lookup(time_steps, time_lookup)
    dashboard_df = attach_season_position(dashboard_df, season_axis_lookup)

    top_voltage_points = attach_rank_labels(top_n_time_points(dashboard_df, "max_vm_pu", n=10), "V")
    top_loading_points = attach_rank_labels(top_n_time_points(dashboard_df, "max_loading_without_dlr_percent", n=10), "L")
    top_benefit_points = attach_rank_labels(top_n_time_points(dashboard_df, "max_dlr_benefit_percent_points", n=10), "B")

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    axes[0].plot(dashboard_df["season_position"], dashboard_df["max_vm_pu"], color="#d62728", label="Max voltage")
    axes[0].plot(dashboard_df["season_position"], dashboard_df["mean_vm_pu"], color="#ff9896", label="Mean voltage")
    for threshold, color in ((1.05, "#ff7f0e"), (1.10, "#9467bd")):
        axes[0].axhline(threshold, color=color, linewidth=1.0, linestyle="--", label=f"{threshold:.2f} pu")
    annotate_time_points(axes[0], top_voltage_points, "season_position", "max_vm_pu", "#d62728")
    axes[0].set_title(f"Subnet seasonal dashboard - {season_name.title()}")
    axes[0].set_ylabel("Voltage [p.u.]")
    format_season_axis(axes[0], month_labels, month_positions)
    axes[0].legend(fontsize=8, ncol=2)

    plot_seasonal_loading_comparison(axes[1], dashboard_df, month_labels, month_positions, top_loading_points)
    plot_seasonal_dlr_benefit(axes[2], dashboard_df, month_labels, month_positions, top_benefit_points)

    axes[3].plot(dashboard_df["season_position"], dashboard_df["net_injection_mw"], color="#1f77b4", label="Net injection")
    axes[3].axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    axes[3].set_ylabel("Net injection [MW]")
    axes[3].set_xlabel("Season month")
    format_season_axis(axes[3], month_labels, month_positions)
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# The HV1-wide summary merges the line current and loading peaks into one table
# so each season gets a compact, directly comparable network overview.
def build_line_peak_summary(line_i_ka, line_loading, loading_comparison_df, time_lookup):
    if line_i_ka.empty or line_loading.empty:
        return pd.DataFrame()

    current_peak = line_i_ka.loc[line_i_ka.groupby("line_index")["i_ka"].idxmax()].copy()
    current_peak = current_peak[["line_index", "name", "time_step", "i_ka"]].rename(
        columns={"time_step": "time_step_max_i_ka", "i_ka": "max_i_ka"}
    )

    loading_peak = line_loading.loc[line_loading.groupby("line_index")["loading_percent"].idxmax()].copy()
    loading_peak = loading_peak[["line_index", "name", "time_step", "loading_percent"]].rename(
        columns={
            "time_step": "time_step_max_loading_without_dlr",
            "loading_percent": "max_loading_without_dlr_percent",
        }
    )

    summary = current_peak.merge(loading_peak, on=["line_index", "name"], how="outer")

    if not loading_comparison_df.empty:
        with_dlr_peak = loading_comparison_df.loc[
            loading_comparison_df.groupby("line_index")["loading_percent_with_dlr"].idxmax()
        ].copy()
        with_dlr_peak = with_dlr_peak[
            ["line_index", "name", "time_step", "loading_percent_with_dlr"]
        ].rename(
            columns={
                "time_step": "time_step_max_loading_with_dlr",
                "loading_percent_with_dlr": "max_loading_with_dlr_percent",
            }
        )
        summary = summary.merge(with_dlr_peak, on=["line_index", "name"], how="left")

    for step_column in [
        "time_step_max_i_ka",
        "time_step_max_loading_without_dlr",
        "time_step_max_loading_with_dlr",
    ]:
        if step_column in summary.columns:
            summary[f"{step_column}_utc"] = pd.to_datetime(summary[step_column].map(time_lookup), utc=True)

    return summary.sort_values("max_loading_without_dlr_percent", ascending=False).reset_index(drop=True)


# The full-network plots focus on the most stressed lines because plotting every
# HV1 line in one bar chart would not be readable.
def plot_top_lines(ax, summary_df, value_column, title, xlabel):
    ax.set_title(title)
    if summary_df.empty or value_column not in summary_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    plot_df = summary_df.nlargest(TOP_LINE_COUNT, value_column).sort_values(value_column)
    ax.barh(plot_df["name"], plot_df[value_column], color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)


# Each seasonal study gets a compact three-panel figure that highlights the line
# peaks across the entire HV1 network.
def plot_hv1_line_peak_figure(summary_df, season_name, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    plot_top_lines(axes[0], summary_df, "max_i_ka", f"HV1 max line currents - {season_name.title()}", "Current [kA]")
    plot_top_lines(
        axes[1],
        summary_df,
        "max_loading_without_dlr_percent",
        f"HV1 max line loading without DLR - {season_name.title()}",
        "Loading [%]",
    )
    plot_top_lines(
        axes[2],
        summary_df,
        "max_loading_with_dlr_percent",
        f"HV1 max line loading with DLR - {season_name.title()}",
        "Loading [%]",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# The seasonal subnet export is intentionally explicit. The goal is that every
# CSV and figure in the season folder reflects the entire season, not a single day.
def export_subnet_results_for_season(net, abs_vals, time_steps, output_path, time_lookup, season_name):
    subnet_dir = os.path.join(output_path, SUBNET_DIR_NAME)
    os.makedirs(subnet_dir, exist_ok=True)

    bus_map = get_subnet_bus_map(net, SUBNET_BUS_NAMES)
    subnet_bus_indices = list(bus_map.values())
    subnet_bus_set = set(subnet_bus_indices)
    subnet_lines = net.line[net.line["from_bus"].isin(subnet_bus_set) & net.line["to_bus"].isin(subnet_bus_set)].copy()
    subnet_trafos = net.trafo[net.trafo["hv_bus"].isin(subnet_bus_set) & net.trafo["lv_bus"].isin(subnet_bus_set)].copy()
    weather_df = load_aitolahti_weather(time_steps)

    bus_summary = build_subnet_bus_summary(net, subnet_bus_indices)
    line_summary = build_subnet_line_summary(net, subnet_lines)
    trafo_summary = build_subnet_trafo_summary(net, subnet_trafos)
    load_summary = build_subnet_component_summary(net, "load", subnet_bus_indices, "load")
    sgen_summary = build_subnet_component_summary(net, "sgen", subnet_bus_indices, "sgen")
    gen_summary = build_subnet_component_summary(net, "gen", subnet_bus_indices, "gen")
    bus_generation_mix_df = build_subnet_bus_generation_mix(net, abs_vals, time_steps, subnet_bus_indices)

    bus_summary.to_csv(os.path.join(subnet_dir, "subnet_bus_summary.csv"), index=False)
    line_summary.to_csv(os.path.join(subnet_dir, "subnet_line_summary.csv"), index=False)
    trafo_summary.to_csv(os.path.join(subnet_dir, "subnet_trafo_summary.csv"), index=False)
    load_summary.to_csv(os.path.join(subnet_dir, "subnet_load_summary.csv"), index=False)
    sgen_summary.to_csv(os.path.join(subnet_dir, "subnet_sgen_summary.csv"), index=False)
    gen_summary.to_csv(os.path.join(subnet_dir, "subnet_gen_summary.csv"), index=False)
    weather_df.to_csv(os.path.join(subnet_dir, "aitolahti_weather_timeseries.csv"), index=False)

    bus_vm = filter_named_results(os.path.join(output_path, "bus_vm_pu_named.csv"), "bus_index", subnet_bus_indices)
    line_indices = subnet_lines.index.astype(int).tolist()
    line_i_ka = filter_named_results(os.path.join(output_path, "line_i_ka_named.csv"), "line_index", line_indices)
    line_loading = filter_named_results(os.path.join(output_path, "line_loading_percent_named.csv"), "line_index", line_indices)
    trafo_loading = pd.DataFrame()
    if not subnet_trafos.empty:
        trafo_loading = filter_named_results(
            os.path.join(output_path, "trafo_loading_percent_named.csv"),
            "trafo_index",
            subnet_trafos.index.astype(int).tolist(),
        )

    injection_df = build_subnet_injection_timeseries(net, abs_vals, time_steps, subnet_bus_indices)
    dlr_df = build_subnet_dlr_timeseries(line_summary, line_i_ka, weather_df)
    loading_comparison_df = build_loading_comparison_timeseries(line_loading, dlr_df)
    conductor_validation_df = build_conductor_validation_summary(line_summary)
    generation_analysis_df = build_subnet_generation_analysis(injection_df, bus_vm)

    bus_vm = attach_time_utc(bus_vm, time_lookup)
    line_i_ka = attach_time_utc(line_i_ka, time_lookup)
    line_loading = attach_time_utc(line_loading, time_lookup)
    trafo_loading = attach_time_utc(trafo_loading, time_lookup)
    injection_df = attach_time_utc(injection_df, time_lookup)
    dlr_df = attach_time_utc(dlr_df, time_lookup)
    loading_comparison_df = attach_time_utc(loading_comparison_df, time_lookup)
    generation_analysis_df = attach_time_utc(generation_analysis_df, time_lookup)

    bus_vm.to_csv(os.path.join(subnet_dir, "subnet_bus_vm_pu.csv"), index=False)
    line_i_ka.to_csv(os.path.join(subnet_dir, "subnet_line_i_ka.csv"), index=False)
    line_loading.to_csv(os.path.join(subnet_dir, "subnet_line_loading_percent.csv"), index=False)
    if not trafo_loading.empty:
        trafo_loading.to_csv(os.path.join(subnet_dir, "subnet_trafo_loading_percent.csv"), index=False)
    injection_df.to_csv(os.path.join(subnet_dir, "subnet_bus_injections.csv"), index=False)
    dlr_df.to_csv(os.path.join(subnet_dir, "subnet_line_dlr_ka.csv"), index=False)
    loading_comparison_df.to_csv(os.path.join(subnet_dir, "subnet_line_loading_comparison.csv"), index=False)
    conductor_validation_df.to_csv(os.path.join(subnet_dir, "subnet_dlr_conductor_check.csv"), index=False)
    generation_analysis_df.to_csv(os.path.join(subnet_dir, "subnet_generation_analysis.csv"), index=False)
    bus_generation_mix_df.to_csv(os.path.join(subnet_dir, "subnet_bus_generation_mix.csv"), index=False)
    build_subnet_loading_summary(loading_comparison_df).to_csv(
        os.path.join(subnet_dir, "subnet_system_loading_summary.csv"),
        index=False,
    )
    summarize_subnet_overvoltage_correlation(generation_analysis_df, season_name)

    plot_subnet_timeseries_for_season(
        bus_vm,
        line_i_ka,
        line_loading,
        dlr_df,
        generation_analysis_df,
        time_lookup,
        time_steps,
        season_name,
        os.path.join(subnet_dir, "subnet_timeseries.png"),
    )
    plot_subnet_generation_voltage_analysis(
        generation_analysis_df,
        time_steps,
        time_lookup,
        season_name,
        os.path.join(subnet_dir, "subnet_generation_voltage_analysis.png"),
    )
    plot_subnet_bus_generation_mix(
        bus_generation_mix_df,
        os.path.join(subnet_dir, "subnet_bus_generation_mix.png"),
    )
    plot_subnet_topology(
        net,
        subnet_bus_indices,
        line_summary,
        trafo_summary,
        bus_vm,
        line_i_ka,
        line_loading,
        trafo_loading,
        dlr_df,
        os.path.join(subnet_dir, "subnet_topology_results.png"),
    )


# This export adds the requested HV1-wide seasonal metrics and figures on top of
# the already existing full-network power-flow CSV exports.
def export_hv1_results_for_season(net, time_steps, output_path, time_lookup, season_name):
    hv1_dir = os.path.join(output_path, "hv1_network")
    os.makedirs(hv1_dir, exist_ok=True)

    weather_df = load_aitolahti_weather(time_steps)
    line_summary = build_subnet_line_summary(net, net.line.copy())
    line_i_ka = pd.read_csv(os.path.join(output_path, "line_i_ka_named.csv"))
    line_loading = pd.read_csv(os.path.join(output_path, "line_loading_percent_named.csv"))

    hv1_dlr_df = build_subnet_dlr_timeseries(line_summary, line_i_ka, weather_df)
    hv1_loading_comparison_df = build_loading_comparison_timeseries(line_loading, hv1_dlr_df)
    hv1_peak_summary = build_line_peak_summary(line_i_ka, line_loading, hv1_loading_comparison_df, time_lookup)

    hv1_dlr_df = attach_time_utc(hv1_dlr_df, time_lookup)
    hv1_loading_comparison_df = attach_time_utc(hv1_loading_comparison_df, time_lookup)

    line_summary.to_csv(os.path.join(hv1_dir, "hv1_line_summary.csv"), index=False)
    hv1_dlr_df.to_csv(os.path.join(hv1_dir, "hv1_line_dlr_ka.csv"), index=False)
    hv1_loading_comparison_df.to_csv(os.path.join(hv1_dir, "hv1_line_loading_comparison.csv"), index=False)
    hv1_peak_summary.to_csv(os.path.join(hv1_dir, "hv1_line_peak_summary.csv"), index=False)
    plot_hv1_line_peak_figure(hv1_peak_summary, season_name, os.path.join(hv1_dir, "hv1_line_peak_overview.png"))

    if not hv1_peak_summary.empty:
        max_current_row = hv1_peak_summary.loc[hv1_peak_summary["max_i_ka"].idxmax()]
        max_loading_row = hv1_peak_summary.loc[hv1_peak_summary["max_loading_without_dlr_percent"].idxmax()]
        print(
            f"  HV1 peak current: {max_current_row['name']} = {max_current_row['max_i_ka']:.3f} kA at "
            f"{max_current_row['time_step_max_i_ka_utc']}"
        )
        print(
            f"  HV1 peak loading without DLR: {max_loading_row['name']} = "
            f"{max_loading_row['max_loading_without_dlr_percent']:.1f}% at "
            f"{max_loading_row['time_step_max_loading_without_dlr_utc']}"
        )


# Each season is solved on a fresh network so the workflow is operationally
# independent, as requested, and each season writes into its own folder.
def run_season_study(season_name, time_steps, output_root, time_lookup):
    output_path = os.path.join(output_root, season_name)
    net = sb.get_simbench_net(GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    add_configured_electric_boilers(net, abs_vals)
    sb.apply_const_controllers(net, abs_vals)
    prepare_output_writer(net, time_steps, output_path)

    try:
        run_timeseries(
            net,
            time_steps=time_steps,
            run=voltage_controlled_runpp,
            continue_on_divergence=True,
            verbose=True,
        )
    except LoadflowNotConverged as err:
        print(f"Season {season_name} stopped on a non-converged step: {err}")

    export_named_results(net, output_path)
    export_subnet_results_for_season(net, abs_vals, time_steps, output_path, time_lookup, season_name)
    export_hv1_results_for_season(net, time_steps, output_path, time_lookup, season_name)


# The root summary tables make the seasonal results comparable without opening
# each season folder one by one. Every line gets one row and one set of columns
# for winter, spring, summer, and autumn.
def build_cross_season_peak_table(season_frames):
    if not season_frames:
        return pd.DataFrame()

    combined = pd.concat(season_frames, ignore_index=True)
    value_columns = [
        "max_i_ka",
        "max_loading_without_dlr_percent",
        "max_loading_with_dlr_percent",
    ]
    available_values = [column for column in value_columns if column in combined.columns]
    if not available_values:
        return combined

    wide = combined.pivot_table(
        index=["line_index", "name"],
        columns="season",
        values=available_values,
        aggfunc="first",
    )
    wide.columns = [f"{season}_{metric}" for metric, season in wide.columns]
    return wide.reset_index().sort_values("name").reset_index(drop=True)


# The subnet summary is rebuilt from the seasonal subnet exports so the values
# reflect the complete season with and without DLR.
def collect_subnet_cross_season_summary(output_root):
    frames = []
    for season_name in SEASON_MONTHS:
        subnet_dir = os.path.join(output_root, season_name, SUBNET_DIR_NAME)
        line_i_path = os.path.join(subnet_dir, "subnet_line_i_ka.csv")
        loading_cmp_path = os.path.join(subnet_dir, "subnet_line_loading_comparison.csv")
        if not (os.path.exists(line_i_path) and os.path.exists(loading_cmp_path)):
            continue

        line_i_ka = pd.read_csv(line_i_path)
        loading_comparison_df = pd.read_csv(loading_cmp_path)
        if line_i_ka.empty or loading_comparison_df.empty:
            continue

        current_peak = line_i_ka.loc[line_i_ka.groupby("line_index")["i_ka"].idxmax(), ["line_index", "name", "i_ka"]]
        current_peak = current_peak.rename(columns={"i_ka": "max_i_ka"})

        no_dlr_peak = loading_comparison_df.loc[
            loading_comparison_df.groupby("line_index")["loading_percent_without_dlr"].idxmax(),
            ["line_index", "name", "loading_percent_without_dlr"],
        ].rename(columns={"loading_percent_without_dlr": "max_loading_without_dlr_percent"})

        subnet_summary = current_peak.merge(no_dlr_peak, on=["line_index", "name"], how="outer")

        if "loading_percent_with_dlr" in loading_comparison_df.columns:
            with_dlr_source = loading_comparison_df.dropna(subset=["loading_percent_with_dlr"]).copy()
            if not with_dlr_source.empty:
                with_dlr_peak = with_dlr_source.loc[
                    with_dlr_source.groupby("line_index")["loading_percent_with_dlr"].idxmax(),
                    ["line_index", "name", "loading_percent_with_dlr"],
                ].rename(columns={"loading_percent_with_dlr": "max_loading_with_dlr_percent"})
                subnet_summary = subnet_summary.merge(with_dlr_peak, on=["line_index", "name"], how="left")

        subnet_summary["season"] = season_name
        frames.append(subnet_summary)

    return build_cross_season_peak_table(frames)


# The HV1 summary reuses the per-season HV1 peak CSV that is already written by
# the seasonal workflow, then pivots it into a side-by-side comparison table.
def collect_seasonal_dlr_benefit_summary(output_root):
    records = []
    for season_name in SEASON_MONTHS:
        subnet_dir = os.path.join(output_root, season_name, SUBNET_DIR_NAME)
        summary_path = os.path.join(subnet_dir, "subnet_system_loading_summary.csv")
        generation_path = os.path.join(subnet_dir, "subnet_generation_analysis.csv")
        if not os.path.exists(summary_path):
            continue

        loading_summary_df = pd.read_csv(summary_path)
        generation_df = pd.read_csv(generation_path) if os.path.exists(generation_path) else pd.DataFrame()
        if loading_summary_df.empty:
            continue

        record = {
            "season": season_name,
            "peak_loading_without_dlr_percent": pd.to_numeric(
                loading_summary_df.get("max_loading_without_dlr_percent"), errors="coerce"
            ).max(),
            "peak_loading_with_dlr_percent": pd.to_numeric(
                loading_summary_df.get("max_loading_with_dlr_percent"), errors="coerce"
            ).max(),
            "peak_dlr_benefit_percent_points": pd.to_numeric(
                loading_summary_df.get("max_dlr_benefit_percent_points"), errors="coerce"
            ).max(),
            "mean_dlr_benefit_percent_points": pd.to_numeric(
                loading_summary_df.get("mean_dlr_benefit_percent_points"), errors="coerce"
            ).mean(),
        }
        if not generation_df.empty and "max_vm_pu" in generation_df.columns:
            record["peak_max_vm_pu"] = pd.to_numeric(generation_df["max_vm_pu"], errors="coerce").max()
        records.append(record)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def collect_hv1_cross_season_summary(output_root):
    frames = []
    required_columns = [
        "line_index",
        "name",
        "max_i_ka",
        "max_loading_without_dlr_percent",
        "max_loading_with_dlr_percent",
    ]
    for season_name in SEASON_MONTHS:
        hv1_path = os.path.join(output_root, season_name, "hv1_network", "hv1_line_peak_summary.csv")
        if not os.path.exists(hv1_path):
            continue

        hv1_summary = pd.read_csv(hv1_path)
        if hv1_summary.empty:
            continue

        available_columns = [column for column in required_columns if column in hv1_summary.columns]
        hv1_summary = hv1_summary[available_columns].copy()
        hv1_summary["season"] = season_name
        frames.append(hv1_summary)

    return build_cross_season_peak_table(frames)


# After the four independent seasonal runs finish, these exports provide the
# requested side-by-side comparison at the root seasonal-results folder.
def export_cross_season_summary_tables(output_root):
    subnet_summary = collect_subnet_cross_season_summary(output_root)
    hv1_summary = collect_hv1_cross_season_summary(output_root)
    dlr_summary = collect_seasonal_dlr_benefit_summary(output_root)

    if not subnet_summary.empty:
        subnet_summary.to_csv(os.path.join(output_root, "seasonal_subnet_line_peak_comparison.csv"), index=False)
        plot_cross_season_overview(
            subnet_summary,
            "Subnet",
            os.path.join(output_root, "seasonal_subnet_line_peak_comparison.png"),
        )
    if not hv1_summary.empty:
        hv1_summary.to_csv(os.path.join(output_root, "seasonal_hv1_line_peak_comparison.csv"), index=False)
        plot_cross_season_overview(
            hv1_summary,
            "HV1 network",
            os.path.join(output_root, "seasonal_hv1_line_peak_comparison.png"),
        )
    if not dlr_summary.empty:
        dlr_summary.to_csv(os.path.join(output_root, "seasonal_subnet_dlr_summary.csv"), index=False)
        plot_seasonal_dlr_benefit_overview(
            dlr_summary,
            os.path.join(output_root, "seasonal_subnet_dlr_summary.png"),
        )


# The comparison figure condenses the side-by-side seasonal tables into one
# visual summary by taking the maximum seasonal value across all lines.
def plot_cross_season_overview(summary_df, scope_label, output_path):
    metrics = [
        ("max_i_ka", "Peak current [kA]"),
        ("max_loading_without_dlr_percent", "Peak loading without DLR [%]"),
        ("max_loading_with_dlr_percent", "Peak loading with DLR [%]"),
    ]
    seasons = list(SEASON_MONTHS.keys())

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, (metric_suffix, ylabel) in zip(axes, metrics):
        values = []
        labels = []
        for season_name in seasons:
            column = f"{season_name}_{metric_suffix}"
            if column in summary_df.columns:
                values.append(pd.to_numeric(summary_df[column], errors="coerce").max())
            else:
                values.append(float("nan"))
            labels.append(season_name.title())
        ax.bar(labels, values, color="#1f77b4")
        ax.set_title(f"{scope_label} seasonal comparison: {ylabel}")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_seasonal_dlr_benefit_overview(summary_df, output_path):
    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df["season_label"] = plot_df["season"].str.title()
    x = np.arange(len(plot_df))
    width = 0.34

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        x - width / 2,
        plot_df["peak_loading_without_dlr_percent"],
        width=width,
        color="#d62728",
        label="Peak loading without DLR",
    )
    ax1.bar(
        x + width / 2,
        plot_df["peak_loading_with_dlr_percent"],
        width=width,
        color="#2ca02c",
        label="Peak loading with DLR",
    )
    ax1.set_ylabel("Peak loading [%]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["season_label"])
    ax1.set_title("Seasonal subnet DLR impact summary")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        plot_df["peak_dlr_benefit_percent_points"],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label="Peak DLR benefit",
    )
    ax2.plot(
        x,
        plot_df["mean_dlr_benefit_percent_points"],
        color="#17becf",
        marker="s",
        linewidth=1.6,
        linestyle="--",
        label="Mean DLR benefit",
    )
    ax2.set_ylabel("DLR benefit [percentage points]")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# The annual workflow is split into four fully independent seasonal studies and
# each one writes its own power-flow and DLR outputs.
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    net = sb.get_simbench_net(GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    max_steps = available_time_steps(abs_vals)

    weather_calendar = load_weather_calendar(max_steps)
    time_lookup = build_time_lookup(weather_calendar)
    season_time_steps = build_season_time_steps(weather_calendar)

    os.makedirs(SEASONAL_OUTPUT_ROOT, exist_ok=True)

    for season_name, time_steps in season_time_steps.items():
        print(f"Running {season_name} study with {len(time_steps)} time steps.")
        run_season_study(season_name, time_steps, SEASONAL_OUTPUT_ROOT, time_lookup)
        print(f"Completed {season_name}. Results written to: {os.path.join(SEASONAL_OUTPUT_ROOT, season_name)}")

    export_cross_season_summary_tables(SEASONAL_OUTPUT_ROOT)
    print("Seasonal 1-year study finished.")
    print(f"Seasonal results written to: {SEASONAL_OUTPUT_ROOT}")
    print(f"Cross-season comparison CSVs written to: {SEASONAL_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
