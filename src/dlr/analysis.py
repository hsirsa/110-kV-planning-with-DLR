import numpy as np
import pandas as pd

from .config import TIME_STEP_HOURS
from .network import safe_name
from .subnet import classify_subnet_bus, infer_sgen_generation_type


def _classify_sgen_generation_type_vectorized(sgen_df):
    """Vectorized replacement for infer_sgen_generation_type applied row-by-row."""
    tokens = ["type", "profile", "source", "tech", "energy", "fuel", "name"]
    candidate_cols = [c for c in sgen_df.columns if any(t in str(c).lower() for t in tokens)]
    if not candidate_cols:
        return pd.Series("other", index=sgen_df.index)
    combined = sgen_df[candidate_cols].fillna("").astype(str).apply(
        lambda row: " ".join(row.str.lower()), axis=1
    )
    result = pd.Series("other", index=sgen_df.index)
    result[combined.str.contains("wind", na=False)] = "wind"
    result[combined.str.contains(r"pv|photovolta|solar", regex=True, na=False)] = "pv"
    return result


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
        id_vars=["time_step"], var_name="element_index", value_name="p_mw",
    )

    meta = filtered[["bus"]].copy()
    meta["element_index"] = meta.index.astype(int)
    meta["bus_index"] = meta["bus"].astype(int)
    meta["bus_name"] = [safe_name(net.bus.at[bus, "name"], f"bus_{bus}") for bus in meta["bus_index"]]

    if element_type == "sgen":
        meta["generation_type"] = _classify_sgen_generation_type_vectorized(filtered).values
    elif element_type == "gen":
        meta["generation_type"] = "other"
    else:
        meta["generation_type"] = "load"

    merged = long_df.merge(meta[["element_index", "bus_index", "bus_name", "generation_type"]], on="element_index", how="left")
    merged["component_type"] = element_type
    return merged[["time_step", "bus_index", "bus_name", "component_type", "generation_type", "p_mw"]]


def build_subnet_bus_generation_mix(net, abs_vals, time_steps, subnet_bus_indices):
    columns = [
        "bus_index", "bus_name",
        "mean_load_mw", "peak_load_mw",
        "mean_pv_mw", "peak_pv_mw",
        "mean_wind_mw", "peak_wind_mw",
        "mean_other_gen_mw", "peak_other_gen_mw",
        "mean_total_generation_mw", "peak_total_generation_mw",
        "mean_net_injection_mw", "peak_net_injection_mw",
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

    detailed_df = pd.concat(frames, ignore_index=True).copy()
    detailed_df["category"] = np.select(
        [
            detailed_df["component_type"] == "load",
            (detailed_df["component_type"] == "sgen") & (detailed_df["generation_type"] == "pv"),
            (detailed_df["component_type"] == "sgen") & (detailed_df["generation_type"] == "wind"),
        ],
        ["load_mw", "pv_mw", "wind_mw"],
        default="other_gen_mw",
    )
    pivoted = (
        detailed_df.groupby(["time_step", "bus_index", "bus_name", "category"])["p_mw"]
        .sum()
        .unstack("category", fill_value=0.0)
        .reset_index()
    )
    for col in ["load_mw", "pv_mw", "wind_mw", "other_gen_mw"]:
        if col not in pivoted.columns:
            pivoted[col] = 0.0

    bus_names = {int(idx): safe_name(net.bus.at[idx, "name"], f"bus_{idx}") for idx in subnet_bus_indices}
    all_time_bus = pd.DataFrame({
        "time_step": np.repeat(np.array(list(time_steps), dtype=int), len(subnet_bus_indices)),
        "bus_index": np.tile(np.array(subnet_bus_indices, dtype=int), len(time_steps)),
    })
    all_time_bus["bus_name"] = all_time_bus["bus_index"].map(bus_names)

    time_series_df = all_time_bus.merge(pivoted, on=["time_step", "bus_index", "bus_name"], how="left")
    for col in ["load_mw", "pv_mw", "wind_mw", "other_gen_mw"]:
        time_series_df[col] = time_series_df[col].fillna(0.0)
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
            row["mean_load_mw"], row["mean_total_generation_mw"], row["mean_pv_mw"], row["mean_wind_mw"],
        ),
        axis=1,
    )
    return summary_df[columns].sort_values(["bus_index", "bus_name"]).reset_index(drop=True)


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
    grouped = (
        merged.groupby(["time_step", bus_column], as_index=False)[["p_mw", "q_mvar"]]
        .sum()
        .rename(columns={bus_column: "bus_index"})
    )
    grouped["bus_name"] = [safe_name(net.bus.at[idx, "name"], f"bus_{idx}") for idx in grouped["bus_index"]]
    grouped["source_type"] = source_label
    return grouped[["time_step", "bus_index", "bus_name", "source_type", "p_mw", "q_mvar"]]


def build_subnet_injection_timeseries(net, abs_vals, time_steps, subnet_bus_indices):
    frames = []
    for element_type in ("load", "sgen", "gen"):
        frame = aggregate_abs_values_by_bus(net, abs_vals, time_steps, element_type, "bus", subnet_bus_indices, element_type)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["time_step", "bus_index", "bus_name", "source_type", "p_mw", "q_mvar"])
    return pd.concat(frames, ignore_index=True)


def summarize_metric(df, index_column, value_column):
    if df.empty:
        return pd.DataFrame()
    return df.groupby(index_column)[value_column].agg(["min", "max", "mean"])


def summarize_bus_vm(bus_vm):
    if bus_vm.empty:
        return pd.DataFrame()
    return bus_vm.groupby("bus_index")["vm_pu"].agg(["min", "max", "mean"])


def build_subnet_generation_analysis(injection_df, bus_vm):
    columns = ["time_step", "total_generation_mw", "total_load_mw", "net_injection_mw", "max_vm_pu"]
    if injection_df.empty or bus_vm.empty:
        return pd.DataFrame(columns=columns)
    generation = (
        injection_df[injection_df["source_type"].isin(["sgen", "gen"])]
        .groupby("time_step", as_index=False)["p_mw"]
        .sum().rename(columns={"p_mw": "total_generation_mw"})
    )
    load = (
        injection_df[injection_df["source_type"] == "load"]
        .groupby("time_step", as_index=False)["p_mw"]
        .sum().rename(columns={"p_mw": "total_load_mw"})
    )
    max_vm = (
        bus_vm.groupby("time_step", as_index=False)["vm_pu"]
        .max().rename(columns={"vm_pu": "max_vm_pu"})
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
        summary_parts.append(
            f"{len(exceedances)} steps above {threshold:.2f} pu with mean net injection "
            f"{exceedances['net_injection_mw'].mean():.2f} MW and peak net injection "
            f"{exceedances['net_injection_mw'].max():.2f} MW"
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
        "max_loading_without_dlr_percent", "mean_loading_without_dlr_percent",
        "max_loading_with_dlr_percent", "mean_loading_with_dlr_percent",
        "max_dlr_benefit_percent_points", "mean_dlr_benefit_percent_points",
    ]
    if loading_comparison_df.empty:
        return pd.DataFrame(columns=columns)
    summary = loading_comparison_df.groupby("time_step", as_index=False).agg(
        max_loading_without_dlr_percent=("loading_percent_without_dlr", "max"),
        mean_loading_without_dlr_percent=("loading_percent_without_dlr", "mean"),
        max_loading_with_dlr_percent=("loading_percent_with_dlr", "max"),
        mean_loading_with_dlr_percent=("loading_percent_with_dlr", "mean"),
        max_dlr_benefit_percent_points=("dlr_benefit_percent_points", "max"),
        mean_dlr_benefit_percent_points=("dlr_benefit_percent_points", "mean"),
    )
    return summary.sort_values("time_step").reset_index(drop=True)


def attach_rank_labels(top_df, rank_prefix):
    ranked = top_df.copy().reset_index(drop=True)
    ranked["rank_label"] = [f"{rank_prefix}{i + 1}" for i in range(len(ranked))]
    return ranked


def top_n_time_points(df, value_column, n=10):
    if df.empty or value_column not in df.columns:
        return pd.DataFrame()
    return df.nlargest(min(n, len(df)), value_column).sort_values(value_column, ascending=False).reset_index(drop=True)


def build_seasonal_dashboard_summary(voltage_summary, loading_summary, generation_analysis_df):
    merged = voltage_summary.merge(loading_summary, on="time_step", how="outer")
    if not generation_analysis_df.empty:
        merged = merged.merge(
            generation_analysis_df[["time_step", "net_injection_mw", "total_generation_mw", "total_load_mw"]],
            on="time_step",
            how="left",
        )
    return merged.sort_values("time_step").reset_index(drop=True)


def build_line_peak_summary(line_i_ka, line_loading, loading_comparison_df, time_lookup):
    if line_i_ka.empty or line_loading.empty:
        return pd.DataFrame()
    current_peak = line_i_ka.loc[line_i_ka.groupby("line_index")["i_ka"].idxmax()].copy()
    current_peak = current_peak[["line_index", "name", "time_step", "i_ka"]].rename(
        columns={"time_step": "time_step_max_i_ka", "i_ka": "max_i_ka"}
    )
    loading_peak = line_loading.loc[line_loading.groupby("line_index")["loading_percent"].idxmax()].copy()
    loading_peak = loading_peak[["line_index", "name", "time_step", "loading_percent"]].rename(
        columns={"time_step": "time_step_max_loading_without_dlr", "loading_percent": "max_loading_without_dlr_percent"}
    )
    summary = current_peak.merge(loading_peak, on=["line_index", "name"], how="outer")
    if not loading_comparison_df.empty:
        with_dlr_source = loading_comparison_df.dropna(subset=["loading_percent_with_dlr"]).copy()
        if not with_dlr_source.empty:
            with_dlr_peak = with_dlr_source.loc[
                with_dlr_source.groupby("line_index")["loading_percent_with_dlr"].idxmax(),
                ["line_index", "name", "time_step", "loading_percent_with_dlr"],
            ].rename(columns={"time_step": "time_step_max_loading_with_dlr", "loading_percent_with_dlr": "max_loading_with_dlr_percent"})
            summary = summary.merge(with_dlr_peak, on=["line_index", "name"], how="left")
    for step_col in ["time_step_max_i_ka", "time_step_max_loading_without_dlr", "time_step_max_loading_with_dlr"]:
        if step_col in summary.columns:
            summary[f"{step_col}_utc"] = pd.to_datetime(summary[step_col].map(time_lookup), utc=True)
    return summary.sort_values("max_loading_without_dlr_percent", ascending=False).reset_index(drop=True)


def attach_time_utc(df, time_lookup):
    if df.empty or "time_step" not in df.columns:
        return df.copy()
    enriched = df.copy()
    enriched["time_utc"] = pd.to_datetime(enriched["time_step"].map(time_lookup), utc=True)
    return enriched


def build_season_axis_lookup(time_steps, time_lookup):
    axis_df = pd.DataFrame({"time_step": pd.Index(time_steps, dtype=int).unique()})
    axis_df["time_utc"] = pd.to_datetime(axis_df["time_step"].map(time_lookup), utc=True)
    axis_df = axis_df.dropna().sort_values("time_utc").reset_index(drop=True)
    axis_df["season_position"] = np.arange(len(axis_df))
    axis_df["month_label"] = axis_df["time_utc"].dt.strftime("%b")
    month_ticks = axis_df.groupby("month_label", sort=False)["season_position"].min()
    return axis_df[["time_step", "season_position"]], month_ticks.index.tolist(), month_ticks.tolist()


def attach_season_position(df, season_axis_lookup):
    if df.empty:
        return df.copy()
    return df.merge(season_axis_lookup, on="time_step", how="left")
