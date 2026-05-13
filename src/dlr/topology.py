"""Network topology export and diagram tools for the HV1 SimBench grid.

Two diagram styles:
- geo: uses real geographic coordinates from bus_geodata / GeoJSON (Geoplot approach)
- tree: custom BFS tree layout with parallel-line offsets, exports PNG + PDF
"""

import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .network import safe_name

# ---------------------------------------------------------------------------
# Shared topology builders
# ---------------------------------------------------------------------------


def _build_bus_summary(net):
    line_counts = pd.Series(0, index=net.bus.index, dtype=int)
    if len(net.line):
        line_counts = (
            line_counts.add(net.line.groupby("from_bus").size(), fill_value=0)
            .add(net.line.groupby("to_bus").size(), fill_value=0)
            .fillna(0)
            .astype(int)
        )
    trafo_counts = pd.Series(0, index=net.bus.index, dtype=int)
    if len(net.trafo):
        trafo_counts = (
            trafo_counts.add(net.trafo.groupby("hv_bus").size(), fill_value=0)
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


def _build_line_summary(net):
    df = net.line.copy()
    df["line_index"] = df.index
    df["line_name"] = [safe_name(net.line.at[idx, "name"], f"line_{idx}") for idx in net.line.index]
    df["from_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.line["from_bus"]]
    df["to_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.line["to_bus"]]
    df["bus_pair"] = df.apply(lambda r: tuple(sorted((int(r["from_bus"]), int(r["to_bus"])))), axis=1)
    pair_counts = df.groupby("bus_pair").size()
    df["same_corridor_line_count"] = df["bus_pair"].map(pair_counts)
    df["parallel_count"] = df["parallel"].fillna(1).astype(int) if "parallel" in df.columns else 1
    df["has_parallel_circuit"] = (df["parallel_count"] > 1) | (df["same_corridor_line_count"] > 1)
    return df


def _build_trafo_summary(net):
    if len(net.trafo) == 0:
        return pd.DataFrame()
    df = net.trafo.copy()
    df["trafo_index"] = df.index
    df["trafo_name"] = [safe_name(net.trafo.at[idx, "name"], f"trafo_{idx}") for idx in net.trafo.index]
    df["hv_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.trafo["hv_bus"]]
    df["lv_bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.trafo["lv_bus"]]
    return df


def _build_substation_summary(net):
    if len(net.ext_grid) == 0:
        return pd.DataFrame()
    df = net.ext_grid.copy()
    df["ext_grid_index"] = df.index
    df["bus_name"] = [safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.ext_grid["bus"]]
    return df


def _build_bus_connections(net):
    records = []
    for bus_idx in net.bus.index:
        bus_name = safe_name(net.bus.at[bus_idx, "name"], f"bus_{bus_idx}")
        for line_idx, row in net.line[(net.line["from_bus"] == bus_idx) | (net.line["to_bus"] == bus_idx)].iterrows():
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
        for trafo_idx, row in net.trafo[(net.trafo["hv_bus"] == bus_idx) | (net.trafo["lv_bus"] == bus_idx)].iterrows():
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


def _export_csvs(net, out_dir, line_filename="line_summary.csv", include_connections=True):
    os.makedirs(out_dir, exist_ok=True)
    bus_df = _build_bus_summary(net)
    line_df = _build_line_summary(net)
    trafo_df = _build_trafo_summary(net)
    sub_df = _build_substation_summary(net)

    bus_df.to_csv(os.path.join(out_dir, "bus_summary.csv"), index=False)
    line_df.to_csv(os.path.join(out_dir, line_filename), index=False)
    trafo_df.to_csv(os.path.join(out_dir, "trafo_summary.csv"), index=False)
    sub_df.to_csv(os.path.join(out_dir, "substation_summary.csv"), index=False)
    line_df.loc[line_df["has_parallel_circuit"]].to_csv(os.path.join(out_dir, "parallel_lines_only.csv"), index=False)
    if include_connections:
        _build_bus_connections(net).to_csv(os.path.join(out_dir, "bus_connections.csv"), index=False)
    return bus_df, line_df, trafo_df


# ---------------------------------------------------------------------------
# Geo-coordinate diagram (Geoplot style)
# ---------------------------------------------------------------------------


def _get_bus_pos_geo(net):
    if hasattr(net, "bus_geodata") and net.bus_geodata is not None and len(net.bus_geodata) > 0:
        cols = {c.lower(): c for c in net.bus_geodata.columns}
        if "x" in cols and "y" in cols:
            pos = {
                int(b): (float(net.bus_geodata.at[b, cols["x"]]), float(net.bus_geodata.at[b, cols["y"]]))
                for b in net.bus.index
                if b in net.bus_geodata.index
            }
            if pos:
                return pos, "real"

    if "geo" in net.bus.columns:
        pos = {}
        for b in net.bus.index:
            g = net.bus.at[b, "geo"]
            if g is None or (isinstance(g, float) and np.isnan(g)):
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

    import pandapower.plotting as pp_plot

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


def run_geo_export(out_dir="HV1_export"):
    """Export topology CSVs and a geo-coordinate network diagram."""
    import simbench

    from .config import GRID_CODE

    net = simbench.get_simbench_net(GRID_CODE)
    os.makedirs(out_dir, exist_ok=True)
    bus_df, line_df, trafo_df = _export_csvs(net, out_dir)
    pos, coord_mode = _get_bus_pos_geo(net)

    G = nx.Graph()
    for b in net.bus.index:
        G.add_node(int(b))
    edge_info = {}
    for _, row in line_df.iterrows():
        fb, tb = int(row["from_bus"]), int(row["to_bus"])
        key = (fb, tb) if fb <= tb else (tb, fb)
        edge_info.setdefault(key, {"length": 0.0, "count": 0, "parallel": False})
        edge_info[key]["length"] += float(row["length_km"]) if pd.notna(row.get("length_km")) else 0.0
        edge_info[key]["count"] += 1
        edge_info[key]["parallel"] = bool(row["has_parallel_circuit"])
        G.add_edge(fb, tb)

    plt.figure(figsize=(30, 20))
    ax = plt.gca()
    normal_edges = [e for e, info in edge_info.items() if not info["parallel"]]
    parallel_edges = [e for e, info in edge_info.items() if info["parallel"]]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1.0, edge_color="black", ax=ax)
    if parallel_edges:
        nx.draw_networkx_edges(G, pos, edgelist=parallel_edges, width=1.8, edge_color="#1f77b4", ax=ax)

    ext_grid_buses = set(net.ext_grid["bus"].tolist()) if len(net.ext_grid) else set()
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[int(b) for b in net.bus.index if int(b) not in ext_grid_buses],
        node_size=40,
        node_color="#1f77b4",
        ax=ax,
    )
    if ext_grid_buses:
        nx.draw_networkx_nodes(G, pos, nodelist=list(ext_grid_buses), node_size=90, node_color="#8e44ad", ax=ax)

    nx.draw_networkx_labels(
        G,
        pos,
        labels={int(b): safe_name(net.bus.at[b, "name"], f"bus_{b}") for b in net.bus.index if int(b) in pos},
        font_size=6,
        ax=ax,
    )

    line_labels = {
        e: f"{info['length']:.1f} km" if info["count"] == 1 else f"{info['length']:.1f} km ({info['count']} lines)"
        for e, info in edge_info.items()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=line_labels, font_size=6, ax=ax)

    for ti in net.trafo.index:
        hv, lv = int(net.trafo.at[ti, "hv_bus"]), int(net.trafo.at[ti, "lv_bus"])
        if hv in pos and lv in pos:
            ax.plot([pos[hv][0], pos[lv][0]], [pos[hv][1], pos[lv][1]], linestyle="--", linewidth=1.2, color="#8e44ad")

    from .config import GRID_CODE as _GC

    plt.title(f"SimBench {_GC} — diagram ({coord_mode} coordinates)")
    plt.axis("off")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "HV1_diagram_busnames_km.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Geo export written to: {os.path.abspath(out_dir)}")
    print(f"  Coordinate mode: {coord_mode}")


# ---------------------------------------------------------------------------
# Tree-layout diagram (new ploting test style)
# ---------------------------------------------------------------------------


def _build_graph(net):
    graph = nx.Graph()
    for bus_idx in net.bus.index:
        graph.add_node(int(bus_idx))
    for line_idx, row in net.line.iterrows():
        graph.add_edge(
            int(row["from_bus"]),
            int(row["to_bus"]),
            kind="line",
            line_index=int(line_idx),
            length_km=row.get("length_km", np.nan),
            parallel=row.get("parallel", 1),
        )
    for trafo_idx, row in net.trafo.iterrows():
        graph.add_edge(int(row["hv_bus"]), int(row["lv_bus"]), kind="trafo", trafo_index=int(trafo_idx))
    return graph


def _layout_tree(net):
    graph = _build_graph(net)
    roots = net.ext_grid["bus"].tolist() if len(net.ext_grid) else [list(graph.nodes)[0]]
    pos, visited, queue, by_level = {}, set(), [], {}
    for r in roots:
        queue.append((int(r), 0))
        visited.add(int(r))
    while queue:
        node, depth = queue.pop(0)
        by_level.setdefault(depth, []).append(node)
        for nb in sorted(graph.neighbors(node), key=lambda n: graph.degree[n], reverse=True):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    for depth, nodes in by_level.items():
        xs = np.linspace(-len(nodes), len(nodes), len(nodes)) if len(nodes) > 1 else np.array([0.0])
        for x, node in zip(xs, nodes):
            pos[node] = (float(x), float(-depth * 2.4))
    missing = [n for n in graph.nodes if n not in pos]
    if missing:
        spring = nx.spring_layout(graph.subgraph(missing), seed=7)
        base_y = min(y for _, y in pos.values()) - 3.0 if pos else 0.0
        for node in missing:
            sx, sy = spring[node]
            pos[node] = (sx * 4.0, base_y + sy * 2.0)
    pos = nx.spring_layout(graph, pos=pos, fixed=[int(r) for r in roots], seed=11, k=1.8, iterations=200)
    return pos


def _offset_segment(x1, y1, x2, y2, offset, order, total):
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return (x1, y1), (x2, y2)
    nx_off, ny_off = -dy / length, dx / length
    shift = (order - (total - 1) / 2.0) * offset
    return (x1 + shift * nx_off, y1 + shift * ny_off), (x2 + shift * nx_off, y2 + shift * ny_off)


def run_tree_diagram(out_dir="hv1_diagram_like_image"):
    """Export topology CSVs and a BFS tree-layout diagram (PNG + PDF)."""
    import simbench

    from .config import GRID_CODE

    os.makedirs(out_dir, exist_ok=True)
    net = simbench.get_simbench_net(GRID_CODE)

    bus_df, line_df, trafo_df = _export_csvs(
        net, out_dir, line_filename="line_summary_with_parallel.csv", include_connections=False
    )
    pos = _layout_tree(net)

    fig, ax = plt.subplots(figsize=(24, 16))
    fig.patch.set_facecolor("#e6e6e6")
    ax.set_facecolor("#e6e6e6")

    for _, group in line_df.groupby("bus_pair"):
        group = group.sort_values("line_index")
        total = len(group)
        for order, (_, row) in enumerate(group.iterrows()):
            fb, tb = int(row["from_bus"]), int(row["to_bus"])
            x1, y1 = pos[fb]
            x2, y2 = pos[tb]
            p1, p2 = _offset_segment(x1, y1, x2, y2, 0.05, order, total)
            is_parallel = bool(row["has_parallel_circuit"])
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="#1f77b4" if is_parallel else "black",
                linewidth=1.2 if is_parallel else 0.9,
                alpha=0.95,
                zorder=1,
            )
            if pd.notna(row.get("length_km", np.nan)):
                ax.text(
                    (p1[0] + p2[0]) / 2,
                    (p1[1] + p2[1]) / 2,
                    f"{float(row['length_km']):.1f} km",
                    fontsize=5,
                    color="black",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "#e6e6e6", "alpha": 0.7, "edgecolor": "none", "pad": 0.05},
                    zorder=3,
                )

    for _, row in trafo_df.iterrows():
        hv, lv = int(row["hv_bus"]), int(row["lv_bus"])
        ax.plot(
            [pos[hv][0], pos[lv][0]], [pos[hv][1], pos[lv][1]], color="black", linewidth=0.8, linestyle="--", zorder=2
        )

    ax.scatter(
        [pos[b][0] for b in bus_df["bus_index"]],
        [pos[b][1] for b in bus_df["bus_index"]],
        s=18,
        color="#1f77b4",
        edgecolors="#1f77b4",
        linewidths=0.3,
        zorder=4,
    )
    for _, row in bus_df.iterrows():
        b = int(row["bus_index"])
        ax.text(pos[b][0] + 0.03, pos[b][1] + 0.01, row["bus_name"], fontsize=5, color="black", zorder=5)

    from .config import GRID_CODE

    ax.set_title(f"SimBench {GRID_CODE} diagram (bus names + line lengths)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    png_path = os.path.join(out_dir, "hv1_diagram_like_image.png")
    pdf_path = os.path.join(out_dir, "hv1_diagram_like_image.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Tree diagram written to: {os.path.abspath(out_dir)}")
