import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import simbench as sb


GRID_CODE = "1-HV-mixed--0-no_sw"
OUTPUT_DIR = "hv1_diagram_like_image"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_net():
    return sb.get_simbench_net(GRID_CODE)


def safe_name(series, index, fallback_prefix):
    value = series.get(index, "")
    if pd.isna(value) or str(value).strip() == "":
        return f"{fallback_prefix}_{index}"
    return str(value)


def build_bus_table(net):
    line_counts = pd.Series(0, index=net.bus.index, dtype=int)
    if len(net.line):
        line_counts = (
            line_counts
            .add(net.line.groupby("from_bus").size(), fill_value=0)
            .add(net.line.groupby("to_bus").size(), fill_value=0)
            .fillna(0)
            .astype(int)
        )

    ext_grid_buses = set(net.ext_grid["bus"].tolist()) if len(net.ext_grid) else set()

    df = net.bus.copy()
    df["bus_index"] = df.index
    df["bus_name"] = [safe_name(net.bus["name"], idx, "bus") for idx in net.bus.index]
    df["connected_line_count"] = line_counts.reindex(net.bus.index, fill_value=0).values
    df["is_substation_bus"] = [idx in ext_grid_buses for idx in net.bus.index]
    return df


def build_line_table(net):
    df = net.line.copy()
    df["line_index"] = df.index
    df["line_name"] = [safe_name(net.line["name"], idx, "line") for idx in net.line.index]
    df["from_bus_name"] = [safe_name(net.bus["name"], b, "bus") for b in net.line["from_bus"]]
    df["to_bus_name"] = [safe_name(net.bus["name"], b, "bus") for b in net.line["to_bus"]]

    df["bus_pair"] = df.apply(
        lambda r: tuple(sorted((int(r["from_bus"]), int(r["to_bus"])))),
        axis=1,
    )
    pair_counts = df.groupby("bus_pair").size()
    df["same_corridor_line_count"] = df["bus_pair"].map(pair_counts)

    if "parallel" in df.columns:
        df["parallel_count"] = df["parallel"].fillna(1).astype(int)
    else:
        df["parallel_count"] = 1

    df["has_parallel_circuit"] = (df["parallel_count"] > 1) | (df["same_corridor_line_count"] > 1)
    return df


def build_trafo_table(net):
    if len(net.trafo) == 0:
        return pd.DataFrame()

    df = net.trafo.copy()
    df["trafo_index"] = df.index
    df["trafo_name"] = [safe_name(net.trafo["name"], idx, "trafo") for idx in net.trafo.index]
    df["hv_bus_name"] = [safe_name(net.bus["name"], b, "bus") for b in net.trafo["hv_bus"]]
    df["lv_bus_name"] = [safe_name(net.bus["name"], b, "bus") for b in net.trafo["lv_bus"]]
    return df


def export_tables(net):
    bus_df = build_bus_table(net)
    line_df = build_line_table(net)
    trafo_df = build_trafo_table(net)

    bus_df.to_csv(os.path.join(OUTPUT_DIR, "bus_summary.csv"), index=False)
    line_df.to_csv(os.path.join(OUTPUT_DIR, "line_summary_with_parallel.csv"), index=False)
    trafo_df.to_csv(os.path.join(OUTPUT_DIR, "trafo_summary.csv"), index=False)

    parallel_only = line_df.loc[line_df["has_parallel_circuit"]].copy()
    parallel_only.to_csv(os.path.join(OUTPUT_DIR, "parallel_lines_only.csv"), index=False)


def build_graph(net):
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
        graph.add_edge(
            int(row["hv_bus"]),
            int(row["lv_bus"]),
            kind="trafo",
            trafo_index=int(trafo_idx),
        )

    return graph


def layout_like_reference(net):
    graph = build_graph(net)
    roots = net.ext_grid["bus"].tolist() if len(net.ext_grid) else [list(graph.nodes)[0]]

    pos = {}
    visited = set()
    queue = []

    for r in roots:
        queue.append((int(r), 0, 0.0))
        visited.add(int(r))

    levels = {}
    by_level = {}

    while queue:
        node, depth, _ = queue.pop(0)
        levels[node] = depth
        by_level.setdefault(depth, []).append(node)

        neighbors = sorted(graph.neighbors(node), key=lambda n: graph.degree[n], reverse=True)
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1, 0.0))

    max_width = 0
    for depth, nodes in by_level.items():
        max_width = max(max_width, len(nodes))
        xs = np.linspace(-len(nodes), len(nodes), len(nodes)) if len(nodes) > 1 else np.array([0.0])
        for x, node in zip(xs, nodes):
            pos[node] = (float(x), float(-depth * 2.4))

    missing = [n for n in graph.nodes if n not in pos]
    if missing:
        spring = nx.spring_layout(graph.subgraph(missing), seed=7)
        base_y = min(y for _, y in pos.values()) - 3.0 if pos else 0.0
        for i, node in enumerate(missing):
            sx, sy = spring[node]
            pos[node] = (sx * 4.0, base_y + sy * 2.0)

    # Relax positions to reduce overlap but keep the rough tree shape
    fixed_nodes = roots
    pos = nx.spring_layout(graph, pos=pos, fixed=fixed_nodes, seed=11, k=1.8, iterations=200)

    return pos


def offset_segment(x1, y1, x2, y2, offset, order, total):
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return (x1, y1), (x2, y2)

    nx_off = -dy / length
    ny_off = dx / length
    center_shift = order - (total - 1) / 2.0
    ox = center_shift * offset * nx_off
    oy = center_shift * offset * ny_off
    return (x1 + ox, y1 + oy), (x2 + ox, y2 + oy)


def draw_diagram(net):
    bus_df = build_bus_table(net)
    line_df = build_line_table(net)
    trafo_df = build_trafo_table(net)
    graph = build_graph(net)
    pos = layout_like_reference(net)

    fig, ax = plt.subplots(figsize=(24, 16))
    fig.patch.set_facecolor("#e6e6e6")
    ax.set_facecolor("#e6e6e6")

    pair_groups = line_df.groupby("bus_pair")
    for _, group in pair_groups:
        group = group.sort_values("line_index")
        total = len(group)

        for order, (_, row) in enumerate(group.iterrows()):
            fb = int(row["from_bus"])
            tb = int(row["to_bus"])
            x1, y1 = pos[fb]
            x2, y2 = pos[tb]

            p1, p2 = offset_segment(x1, y1, x2, y2, offset=0.05, order=order, total=total)

            is_parallel = bool(row["has_parallel_circuit"])
            color = "#1f77b4" if is_parallel else "black"
            width = 1.2 if is_parallel else 0.9

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=width,
                alpha=0.95,
                zorder=1,
            )

            if pd.notna(row.get("length_km", np.nan)):
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                ax.text(
                    mx,
                    my,
                    f"{float(row['length_km']):.1f} km",
                    fontsize=5,
                    color="black",
                    rotation=0,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "#e6e6e6", "alpha": 0.7, "edgecolor": "none", "pad": 0.05},
                    zorder=3,
                )

    for _, row in trafo_df.iterrows():
        hv = int(row["hv_bus"])
        lv = int(row["lv_bus"])
        x1, y1 = pos[hv]
        x2, y2 = pos[lv]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="black",
            linewidth=0.8,
            linestyle="--",
            zorder=2,
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
        x, y = pos[b]
        ax.text(
            x + 0.03,
            y + 0.01,
            row["bus_name"],
            fontsize=5,
            color="black",
            zorder=5,
        )

    ax.set_title(f"SimBench {GRID_CODE} diagram (bus names + line lengths)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "hv1_diagram_like_image.png")
    pdf_path = os.path.join(OUTPUT_DIR, "hv1_diagram_like_image.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

    print("Created:")
    print(os.path.join(OUTPUT_DIR, "bus_summary.csv"))
    print(os.path.join(OUTPUT_DIR, "line_summary_with_parallel.csv"))
    print(os.path.join(OUTPUT_DIR, "parallel_lines_only.csv"))
    print(os.path.join(OUTPUT_DIR, "trafo_summary.csv"))
    print(png_path)
    print(pdf_path)


def main():
    ensure_output_dir()
    net = load_net()
    export_tables(net)
    draw_diagram(net)


if __name__ == "__main__":
    main()
