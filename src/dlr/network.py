import json

import networkx as nx
import numpy as np
import pandas as pd

from .config import SUBNET_BUS_NAMES


def safe_name(value, fallback):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def available_time_steps(abs_vals):
    return len(abs_vals[("load", "p_mw")].index)


def get_bus_index_by_name(net, bus_name):
    names = net.bus["name"].fillna("").astype(str).str.strip()
    filled = names.where(names != "", other=pd.Series([f"bus_{i}" for i in net.bus.index], index=net.bus.index))
    hits = net.bus.index[filled == bus_name]
    if len(hits) == 0:
        raise KeyError(f"Bus not found in SimBench net: {bus_name}")
    return int(hits[0])


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


def get_subnet_bus_map(net, bus_names=None):
    if bus_names is None:
        bus_names = SUBNET_BUS_NAMES
    available = {safe_name(net.bus.at[idx, "name"], f"bus_{idx}"): int(idx) for idx in net.bus.index}
    missing = [name for name in bus_names if name not in available]
    if missing:
        raise KeyError(f"Subnet bus names not found in SimBench net: {missing}")
    return {name: available[name] for name in bus_names}
