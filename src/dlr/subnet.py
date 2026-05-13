import math

import numpy as np
import pandas as pd

from .config import (
    CONDUCTOR_TEMP_C,
    DEFAULT_ABSORPTIVITY,
    DEFAULT_ALPHA,
    DEFAULT_EMISSIVITY,
)
from .network import first_valid_value, get_bus_coordinate_map, safe_name


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


def compute_line_azimuth_deg(net, from_bus, to_bus, pos_map=None):
    pos = pos_map if pos_map is not None else get_bus_coordinate_map(net, [from_bus, to_bus])
    if from_bus not in pos or to_bus not in pos:
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
        "emissivity": float(
            first_valid_value(std_line.get("emissivity"), line_row.get("emissivity"), DEFAULT_EMISSIVITY)
        ),
        "absorptivity": float(
            first_valid_value(std_line.get("absorptivity"), line_row.get("absorptivity"), DEFAULT_ABSORPTIVITY)
        ),
        "parallel_count": int(line_row.get("parallel", 1) or 1),
        "line_azimuth_deg": compute_line_azimuth_deg(net, int(line_row["from_bus"]), int(line_row["to_bus"])),
        "dlr_conductor_temp_c": CONDUCTOR_TEMP_C,
    }


def _build_conductor_props_no_geo(net, line_row, azimuth_deg):
    """Like build_conductor_properties but accepts a pre-computed azimuth_deg."""
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
        "emissivity": float(
            first_valid_value(std_line.get("emissivity"), line_row.get("emissivity"), DEFAULT_EMISSIVITY)
        ),
        "absorptivity": float(
            first_valid_value(std_line.get("absorptivity"), line_row.get("absorptivity"), DEFAULT_ABSORPTIVITY)
        ),
        "parallel_count": int(line_row.get("parallel", 1) or 1),
        "line_azimuth_deg": azimuth_deg,
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
    all_buses = list(set(line_df["from_bus"].tolist() + line_df["to_bus"].tolist()))
    cached_pos = get_bus_coordinate_map(net, all_buses)
    azimuths = {
        idx: compute_line_azimuth_deg(net, int(line_df.at[idx, "from_bus"]), int(line_df.at[idx, "to_bus"]), cached_pos)
        for idx in line_df.index
    }
    conductor_df = pd.DataFrame(
        [_build_conductor_props_no_geo(net, row, azimuths[idx]) for idx, row in line_df.iterrows()],
        index=line_df.index,
    )
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
    candidate_columns = [
        col
        for col in row.index
        if any(token in str(col).lower() for token in ["type", "profile", "source", "tech", "energy", "fuel", "name"])
    ]
    text_parts = [str(row.get(col)).strip().lower() for col in candidate_columns if not pd.isna(row.get(col))]
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
        "line_index",
        "line_name",
        "std_type",
        "conductor_id",
        "from_bus_name",
        "to_bus_name",
        "parallel",
        "parallel_count",
        "parallel_match",
        "r_ohm_per_km",
        "resistance_20c_ohm_per_km",
        "resistance_match",
        "x_ohm_per_km",
        "max_i_ka",
        "q_mm2",
        "diameter_m_est",
        "alpha_per_c",
        "emissivity",
        "absorptivity",
        "line_azimuth_deg",
        "dlr_conductor_temp_c",
    ]
    return summary[[col for col in ordered if col in summary.columns]].copy()
