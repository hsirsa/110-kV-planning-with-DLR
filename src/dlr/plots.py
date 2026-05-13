import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .analysis import (
    build_seasonal_dashboard_summary,
    build_subnet_loading_summary,
    build_subnet_voltage_summary,
    summarize_bus_vm,
    summarize_metric,
)
from .config import SEASON_MONTHS, TIME_STEP_HOURS, TOP_LINE_COUNT
from .dlr_calc import build_loading_comparison_timeseries
from .network import get_subnet_positions, safe_name

# --- Shared annotation helpers ---


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


def format_season_axis(ax, month_labels, month_positions):
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels)
    ax.grid(True, alpha=0.3)


# --- Per-season subnet plots ---


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
    mean_ax.scatter(
        plot_df["mean_net_injection_mw"], y_positions, color="#1f77b4", marker="D", s=36, label="Mean net injection"
    )
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
    peak_ax.scatter(
        plot_df["peak_net_injection_mw"], y_positions, color="#d62728", marker="o", s=36, label="Peak net injection"
    )
    scale = max(plot_df["peak_total_generation_mw"].max(), plot_df["peak_load_mw"].max(), 1.0) * 0.01
    for idx, row in plot_df.iterrows():
        peak_ax.text(
            max(row["peak_total_generation_mw"], row["peak_load_mw"]) + scale,
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
        ax.plot(group.sort_values(x_column)[x_column], group.sort_values(x_column)[value_column], label=label)
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
        dlr_df[["time_step", "line_index", "dlr_ka"]], on=["time_step", "line_index"], how="left"
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
    plot_grouped_series(
        axes[0],
        bus_vm.assign(hour_of_day=bus_vm["time_step"] * TIME_STEP_HOURS),
        "hour_of_day",
        "name",
        "vm_pu",
        "Voltage [p.u.]",
        "Subnet bus voltages",
    )
    plot_current_vs_dlr(axes[1], line_i_ka, dlr_df)
    plot_loading_comparison(axes[2], loading_comparison_df)
    axes[2].set_xlabel("Hour of day")
    axes[2].set_xlim(0.0, 24.0)
    axes[2].set_xticks(np.arange(0.0, 25.0, 2.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_subnet_topology(
    net,
    subnet_bus_indices,
    line_summary,
    trafo_summary,
    bus_vm,
    line_i_ka,
    line_loading,
    trafo_loading,
    dlr_df,
    output_path,
):
    graph = nx.Graph()
    for bus_idx in subnet_bus_indices:
        graph.add_node(int(bus_idx))
    for _, row in line_summary.iterrows():
        graph.add_edge(int(row["from_bus"]), int(row["to_bus"]), edge_type="line")
    for _, row in trafo_summary.iterrows():
        graph.add_edge(int(row["hv_bus"]), int(row["lv_bus"]), edge_type="trafo")
    pos = get_subnet_positions(net, subnet_bus_indices, graph)
    fig, ax = plt.subplots(figsize=(14, 10))
    line_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["edge_type"] == "line"]
    trafo_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["edge_type"] == "trafo"]
    if line_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=line_edges, width=2.0, edge_color="#1f77b4", ax=ax)
    if trafo_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=trafo_edges, width=2.0, edge_color="#d35400", style="dashed", ax=ax)
    ext_grid_buses = set(net.ext_grid["bus"].tolist())
    node_colors = ["#f5b041" if idx in ext_grid_buses else "#85c1e9" for idx in graph.nodes]
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
        ax.text(
            x,
            y,
            label,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
        )
    ax.set_title("Subnet power flow and DLR summary")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# --- Seasonal plots ---


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


def plot_seasonal_current_vs_dlr(ax, line_i_ka, dlr_df, month_labels, month_positions):
    ax.set_title("Subnet line current vs dynamic rating")
    ax.set_ylabel("Current / DLR [kA]")
    if line_i_ka.empty or dlr_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    merged = line_i_ka.merge(dlr_df[["time_step", "line_index", "dlr_ka"]], on=["time_step", "line_index"], how="left")
    for label, group in merged.groupby("name"):
        ordered = group.sort_values("season_position")
        ax.plot(ordered["season_position"], ordered["i_ka"], label=f"{label} I")
        ax.plot(ordered["season_position"], ordered["dlr_ka"], linestyle="--", label=f"{label} DLR")
    format_season_axis(ax, month_labels, month_positions)
    ax.legend(fontsize=7, ncol=2)


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
    annotate_time_points(ax, highlighted_points, "season_position", "max_loading_without_dlr_percent", "#d62728")
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
    annotate_time_points(ax, highlighted_points, "season_position", "max_dlr_benefit_percent_points", "#1f77b4")
    format_season_axis(ax, month_labels, month_positions)
    ax.legend(fontsize=8)


def plot_subnet_generation_voltage_analysis(analysis_df, time_steps, time_lookup, season_name, output_path):
    from .analysis import attach_rank_labels, attach_season_position, build_season_axis_lookup, top_n_time_points

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
    injection_ax.plot(
        plot_df["season_position"], plot_df["net_injection_mw"], color="#1f77b4", alpha=0.8, label="Net injection"
    )
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
        plot_df["net_injection_mw"], plot_df["max_vm_pu"], c=plot_df["season_position"], cmap="viridis", s=20, alpha=0.8
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


def plot_subnet_timeseries_for_season(
    bus_vm, line_i_ka, line_loading, dlr_df, generation_analysis_df, time_lookup, time_steps, season_name, output_path
):
    from .analysis import (
        attach_rank_labels,
        attach_season_position,
        attach_time_utc,
        build_season_axis_lookup,
        top_n_time_points,
    )

    bus_vm = attach_time_utc(bus_vm, time_lookup)
    line_loading = attach_time_utc(line_loading, time_lookup)
    loading_comparison_df = attach_time_utc(build_loading_comparison_timeseries(line_loading, dlr_df), time_lookup)
    generation_analysis_df = attach_time_utc(generation_analysis_df, time_lookup)
    voltage_summary_df = attach_time_utc(build_subnet_voltage_summary(bus_vm), time_lookup)
    loading_summary_df = attach_time_utc(build_subnet_loading_summary(loading_comparison_df), time_lookup)
    dashboard_df = build_seasonal_dashboard_summary(voltage_summary_df, loading_summary_df, generation_analysis_df)
    season_axis_lookup, month_labels, month_positions = build_season_axis_lookup(time_steps, time_lookup)
    dashboard_df = attach_season_position(dashboard_df, season_axis_lookup)

    top_voltage_points = attach_rank_labels(top_n_time_points(dashboard_df, "max_vm_pu", n=10), "V")
    top_loading_points = attach_rank_labels(
        top_n_time_points(dashboard_df, "max_loading_without_dlr_percent", n=10), "L"
    )
    top_benefit_points = attach_rank_labels(
        top_n_time_points(dashboard_df, "max_dlr_benefit_percent_points", n=10), "B"
    )

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

    axes[3].plot(
        dashboard_df["season_position"], dashboard_df["net_injection_mw"], color="#1f77b4", label="Net injection"
    )
    axes[3].axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    axes[3].set_ylabel("Net injection [MW]")
    axes[3].set_xlabel("Season month")
    format_season_axis(axes[3], month_labels, month_positions)
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# --- HV1-wide and cross-season plots ---


def plot_top_lines(ax, summary_df, value_column, title, xlabel):
    ax.set_title(title)
    if summary_df.empty or value_column not in summary_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    plot_df = summary_df.nlargest(TOP_LINE_COUNT, value_column).sort_values(value_column)
    ax.barh(plot_df["name"], plot_df[value_column], color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)


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
            col = f"{season_name}_{metric_suffix}"
            values.append(
                pd.to_numeric(summary_df[col], errors="coerce").max() if col in summary_df.columns else float("nan")
            )
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
