import os
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import simbench
import pandapower.plotting as pp_plot


SIMBENCH_CODE = "1-HV-mixed--0-no_sw"
OUT_DIR = "HV1_export"
os.makedirs(OUT_DIR, exist_ok=True)


def safe_name(value, fallback):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def get_bus_pos(net):
    if hasattr(net, "bus_geodata") and net.bus_geodata is not None and len(net.bus_geodata) > 0:
        cols = {c.lower() for c in net.bus_geodata.columns}
        if "x" in cols and "y" in cols:
            xcol = next(c for c in net.bus_geodata.columns if c.lower() == "x")
            ycol = next(c for c in net.bus_geodata.columns if c.lower() == "y")
            pos = {
                int(b): (float(net.bus_geodata.at[b, xcol]), float(net.bus_geodata.at[b, ycol]))
                for b in net.bus.index
                if b in net.bus_geodata.index
            }
            if pos:
                return pos, "real"

    if "geo" in net.bus.columns:
        pos = {}
        for b in net.bus.index:
            g = net.bus.at[b, "geo"]
            if g is None or isinstance(g, float):
                continue
            if isinstance(g, str) and g.strip():
                try:
                    gj = json.loads(g)
                    if gj.get("type") == "Point":
                        x, y = gj["coordinates"][:2]
                        pos[int(b)] = (float(x), float(y))
                except Exception:
                    pass
        if pos:
            return pos, "real"

    pp_plot.create_generic_coordinates(net)
    if hasattr(net, "bus_geodata") and net.bus_geodata is not None and len(net.bus_geodata) > 0:
        pos = {
            int(b): (float(net.bus_geodata.at[b, "x"]), float(net.bus_geodata.at[b, "y"]))
            for b in net.bus.index
            if b in net.bus_geodata.index
        }
        if pos:
            return pos, "generic"

    raise ValueError("Could not derive any bus coordinates.")


def build_bus_summary(net):
    line_counts = pd.Series(0, index=net.bus.index, dtype=int)
    if len(net.line):
        line_counts = (
            line_counts
            .add(net.line.groupby("from_bus").size(), fill_value=0)
            .add(net.line.groupby("to_bus").size(), fill_value=0)
            .fillna(0)
            .astype(int)
        )

    trafo_counts = pd.Series(0, index=net.bus.index, dtype=int)
    if len(net.trafo):
        trafo_counts = (
            trafo_counts
            .add(net.trafo.groupby("hv_bus").size(), fill_value=0)
            .add(net.trafo.groupby("lv_bus").size(), fill_value=0)
            .fillna(0)
            .astype(int)
        )

    ext_grid_buses = set(net.ext_grid["bus"].tolist()) if len(net.ext_grid) else set()

    df = net.bus.copy()
    df["bus_index"] = df.index
    df["bus_name"] = [safe_name(net.bus.at[idx, "name"], f"bus_{idx}") for idx in net.bus.index]
    df["connected_line_count"] = line_counts.reindex(net.bus.index, fill_value=0).values
    df["connected_trafo_count"] = trafo_counts.reindex(net.bus.index, fill_value=0).values
    df["is_substation_bus"] = [idx in ext_grid_buses for idx in net.bus.index]
    return df


def build_line_summary(net):
    df = net.line.copy()
    df["line_index"] = df.index
    df["line_name"] = [safe_name(net.line.at[idx, "name"], f"line_{idx}") for idx in net.line.index]
    df["from_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.line["from_bus"]]
    df["to_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.line["to_bus"]]

    df["bus_pair"] = df.apply(
        lambda r: tuple(sorted((int(r["from_bus"]), int(r["to_bus"])))),
        axis=1
    )
    pair_counts = df.groupby("bus_pair").size()
    df["same_corridor_line_count"] = df["bus_pair"].map(pair_counts)

    if "parallel" in df.columns:
        df["parallel_count"] = df["parallel"].fillna(1).astype(int)
    else:
        df["parallel_count"] = 1

    df["has_parallel_circuit"] = (df["parallel_count"] > 1) | (df["same_corridor_line_count"] > 1)
    return df


def build_trafo_summary(net):
    if len(net.trafo) == 0:
        return pd.DataFrame()

    df = net.trafo.copy()
    df["trafo_index"] = df.index
    df["trafo_name"] = [safe_name(net.trafo.at[idx, "name"], f"trafo_{idx}") for idx in net.trafo.index]
    df["hv_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.trafo["hv_bus"]]
    df["lv_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.trafo["lv_bus"]]
    return df


def build_substation_summary(net):
    if len(net.ext_grid) == 0:
        return pd.DataFrame()

    df = net.ext_grid.copy()
    df["ext_grid_index"] = df.index
    df["bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.ext_grid["bus"]]
    return df


def build_bus_connections(net):
    records = []

    for bus_idx in net.bus.index:
        bus_name = safe_name(net.bus.at[bus_idx, "name"], f"bus_{bus_idx}")

        connected_lines = net.line[(net.line["from_bus"] == bus_idx) | (net.line["to_bus"] == bus_idx)]
        for line_idx, row in connected_lines.iterrows():
            other_bus = int(row["to_bus"]) if int(row["from_bus"]) == int(bus_idx) else int(row["from_bus"])
            records.append(
                {
                    "bus_index": int(bus_idx),
                    "bus_name": bus_name,
                    "connection_type": "line",
                    "element_index": int(line_idx),
                    "element_name": safe_name(net.line.at[line_idx, "name"], f"line_{line_idx}"),
                    "connected_to_bus": other_bus,
                    "connected_to_bus_name": safe_name(net.bus.at[other_bus, "name"], f"bus_{other_bus}"),
                    "length_km": row.get("length_km", np.nan),
                    "parallel": row.get("parallel", 1),
                }
            )

        connected_trafos = net.trafo[(net.trafo["hv_bus"] == bus_idx) | (net.trafo["lv_bus"] == bus_idx)]
        for trafo_idx, row in connected_trafos.iterrows():
            other_bus = int(row["lv_bus"]) if int(row["hv_bus"]) == int(bus_idx) else int(row["hv_bus"])
            records.append(
                {
                    "bus_index": int(bus_idx),
                    "bus_name": bus_name,
                    "connection_type": "transformer",
                    "element_index": int(trafo_idx),
                    "element_name": safe_name(net.trafo.at[trafo_idx, "name"], f"trafo_{trafo_idx}"),
                    "connected_to_bus": other_bus,
                    "connected_to_bus_name": safe_name(net.bus.at[other_bus, "name"], f"bus_{other_bus}"),
                    "length_km": np.nan,
                    "parallel": np.nan,
                }
            )

    return pd.DataFrame(records)


def export_csvs(net):
    bus_df = build_bus_summary(net)
    line_df = build_line_summary(net)
    trafo_df = build_trafo_summary(net)
    sub_df = build_substation_summary(net)
    conn_df = build_bus_connections(net)

    bus_df.to_csv(os.path.join(OUT_DIR, "bus_summary.csv"), index=False)
    line_df.to_csv(os.path.join(OUT_DIR, "line_summary.csv"), index=False)
    trafo_df.to_csv(os.path.join(OUT_DIR, "trafo_summary.csv"), index=False)
    sub_df.to_csv(os.path.join(OUT_DIR, "substation_summary.csv"), index=False)
    conn_df.to_csv(os.path.join(OUT_DIR, "bus_connections.csv"), index=False)

    parallel_df = line_df.loc[line_df["has_parallel_circuit"]].copy()
    parallel_df.to_csv(os.path.join(OUT_DIR, "parallel_lines_only.csv"), index=False)

    return line_df


def plot_diagram(net, pos, coord_mode, line_df):
    G = nx.Graph()

    for b in net.bus.index:
        G.add_node(int(b))

    edge_info = {}
    for _, row in line_df.iterrows():
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        key = (fb, tb) if fb <= tb else (tb, fb)

        edge_info.setdefault(
            key,
            {
                "length": 0.0,
                "count": 0,
                "parallel": False,
            }
        )
        edge_info[key]["length"] += float(row["length_km"]) if pd.notna(row["length_km"]) else 0.0
        edge_info[key]["count"] += 1
        edge_info[key]["parallel"] = bool(row["has_parallel_circuit"])

        G.add_edge(fb, tb)

    plt.figure(figsize=(30, 20))
    ax = plt.gca()

    normal_edges = [edge for edge, info in edge_info.items() if not info["parallel"]]
    parallel_edges = [edge for edge, info in edge_info.items() if info["parallel"]]

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1.0, edge_color="black", ax=ax)
    if parallel_edges:
        nx.draw_networkx_edges(G, pos, edgelist=parallel_edges, width=1.8, edge_color="#1f77b4", ax=ax)

    substation_buses = set(net.ext_grid["bus"].tolist()) if len(net.ext_grid) else set()
    normal_buses = [int(b) for b in net.bus.index if int(b) not in substation_buses]
    sub_buses = [int(b) for b in substation_buses]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=normal_buses,
        node_size=40,
        node_color="#1f77b4",
        ax=ax
    )

    if sub_buses:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=sub_buses,
            node_size=90,
            node_color="#8e44ad",
            ax=ax
        )

    bus_labels = {
        int(b): safe_name(net.bus.at[b, "name"], f"bus_{b}")
        for b in net.bus.index
        if int(b) in pos
    }
    nx.draw_networkx_labels(G, pos, labels=bus_labels, font_size=6, ax=ax)

    line_labels = {}
    for edge, info in edge_info.items():
        if info["count"] == 1:
            line_labels[edge] = f"{info['length']:.1f} km"
        else:
            line_labels[edge] = f"{info['length']:.1f} km ({info['count']} lines)"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=line_labels, font_size=6, ax=ax)

    for ti in net.trafo.index:
        hv = int(net.trafo.at[ti, "hv_bus"])
        lv = int(net.trafo.at[ti, "lv_bus"])
        if hv in pos and lv in pos:
            x1, y1 = pos[hv]
            x2, y2 = pos[lv]
            ax.plot([x1, x2], [y1, y2], linestyle="--", linewidth=1.2, color="#8e44ad")

    plt.title(f"SimBench {SIMBENCH_CODE} — diagram ({coord_mode} coordinates)")
    plt.axis("off")
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, "HV1_diagram_busnames_km.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    return png_path


def main():
    net = simbench.get_simbench_net(SIMBENCH_CODE)
    pos, coord_mode = get_bus_pos(net)
    line_df = export_csvs(net)
    png_path = plot_diagram(net, pos, coord_mode, line_df)

    print("Created:")
    print(os.path.abspath(os.path.join(OUT_DIR, "bus_summary.csv")))
    print(os.path.abspath(os.path.join(OUT_DIR, "line_summary.csv")))
    print(os.path.abspath(os.path.join(OUT_DIR, "parallel_lines_only.csv")))
    print(os.path.abspath(os.path.join(OUT_DIR, "trafo_summary.csv")))
    print(os.path.abspath(os.path.join(OUT_DIR, "substation_summary.csv")))
    print(os.path.abspath(os.path.join(OUT_DIR, "bus_connections.csv")))
    print(os.path.abspath(png_path))
    print("Coordinate mode:", coord_mode)


if __name__ == "__main__":
    main()
