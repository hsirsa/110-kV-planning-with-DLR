import os
import warnings
from collections import OrderedDict

import pandas as pd
import simbench as sb
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries import run_timeseries

from .analysis import (
    attach_time_utc,
    build_line_peak_summary,
    build_subnet_bus_generation_mix,
    build_subnet_generation_analysis,
    build_subnet_injection_timeseries,
    build_subnet_loading_summary,
    summarize_subnet_overvoltage_correlation,
)
from .boiler import add_configured_electric_boilers
from .config import (
    AITOLAHTI_WEATHER_CSV,
    GRID_CODE,
    SEASON_MONTHS,
    SEASONAL_OUTPUT_ROOT,
    SUBNET_DIR_NAME,
)
from .dlr_calc import build_loading_comparison_timeseries, build_subnet_dlr_timeseries, load_aitolahti_weather
from .export import export_named_results, filter_named_results
from .network import available_time_steps, get_subnet_bus_map
from .plots import (
    plot_cross_season_overview,
    plot_hv1_line_peak_figure,
    plot_seasonal_dlr_benefit_overview,
    plot_subnet_bus_generation_mix,
    plot_subnet_generation_voltage_analysis,
    plot_subnet_timeseries_for_season,
    plot_subnet_topology,
)
from .powerflow import diverged_time_steps, prepare_output_writer, robust_runpp
from .subnet import (
    build_conductor_validation_summary,
    build_subnet_bus_summary,
    build_subnet_component_summary,
    build_subnet_line_summary,
    build_subnet_trafo_summary,
)

SEASON_OUTPUT_ROOTS = {
    "winter": "results_1year_winter_subnet_dlr_only",
    "spring": "results_1year_spring_subnet_dlr_only",
    "summer": "results_1year_summer_subnet_dlr_only",
    "autumn": "results_1year_autumn_subnet_dlr_only",
}


def load_weather_calendar(max_steps):
    if not AITOLAHTI_WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"Aitolahti weather file not found: {AITOLAHTI_WEATHER_CSV}. Run dlr_weather_tampere_full.py first."
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


def build_season_time_steps(weather_calendar):
    season_steps = OrderedDict()
    for season_name, months in SEASON_MONTHS.items():
        steps = weather_calendar.loc[weather_calendar["month"].isin(months), "time_step"].astype(int).tolist()
        season_steps[season_name] = steps
    return season_steps


def build_time_lookup(weather_calendar):
    return weather_calendar.set_index("time_step")["time_utc"]


def export_subnet_results(net, abs_vals, time_steps, output_path):
    subnet_dir = os.path.join(output_path, SUBNET_DIR_NAME)
    os.makedirs(subnet_dir, exist_ok=True)
    bus_map = get_subnet_bus_map(net)
    subnet_bus_indices = list(bus_map.values())
    subnet_bus_set = set(subnet_bus_indices)
    subnet_lines = net.line[net.line["from_bus"].isin(subnet_bus_set) & net.line["to_bus"].isin(subnet_bus_set)].copy()
    subnet_trafos = net.trafo[
        net.trafo["hv_bus"].isin(subnet_bus_set) & net.trafo["lv_bus"].isin(subnet_bus_set)
    ].copy()
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
    line_loading = filter_named_results(
        os.path.join(output_path, "line_loading_percent_named.csv"), "line_index", line_indices
    )
    line_i_ka.to_csv(os.path.join(subnet_dir, "subnet_line_i_ka.csv"), index=False)
    line_loading.to_csv(os.path.join(subnet_dir, "subnet_line_loading_percent.csv"), index=False)
    trafo_loading = pd.DataFrame()
    if not subnet_trafos.empty:
        trafo_loading = filter_named_results(
            os.path.join(output_path, "trafo_loading_percent_named.csv"),
            "trafo_index",
            subnet_trafos.index.astype(int).tolist(),
        )
        trafo_loading.to_csv(os.path.join(subnet_dir, "subnet_trafo_loading_percent.csv"), index=False)
    injection_df = build_subnet_injection_timeseries(net, abs_vals, time_steps, subnet_bus_indices)
    injection_df.to_csv(os.path.join(subnet_dir, "subnet_bus_injections.csv"), index=False)
    dlr_df = build_subnet_dlr_timeseries(line_summary, line_i_ka, weather_df)
    loading_comparison_df = build_loading_comparison_timeseries(line_loading, dlr_df)
    conductor_validation_df = build_conductor_validation_summary(line_summary)
    dlr_df.to_csv(os.path.join(subnet_dir, "subnet_line_dlr_ka.csv"), index=False)
    loading_comparison_df.to_csv(os.path.join(subnet_dir, "subnet_line_loading_comparison.csv"), index=False)
    conductor_validation_df.to_csv(os.path.join(subnet_dir, "subnet_dlr_conductor_check.csv"), index=False)

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


def export_subnet_results_for_season(net, abs_vals, time_steps, output_path, time_lookup, season_name, weather_df=None):
    subnet_dir = os.path.join(output_path, SUBNET_DIR_NAME)
    os.makedirs(subnet_dir, exist_ok=True)

    bus_map = get_subnet_bus_map(net)
    subnet_bus_indices = list(bus_map.values())
    subnet_bus_set = set(subnet_bus_indices)
    subnet_lines = net.line[net.line["from_bus"].isin(subnet_bus_set) & net.line["to_bus"].isin(subnet_bus_set)].copy()
    subnet_trafos = net.trafo[
        net.trafo["hv_bus"].isin(subnet_bus_set) & net.trafo["lv_bus"].isin(subnet_bus_set)
    ].copy()
    if weather_df is None:
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
    line_loading = filter_named_results(
        os.path.join(output_path, "line_loading_percent_named.csv"), "line_index", line_indices
    )
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
    plot_subnet_bus_generation_mix(bus_generation_mix_df, os.path.join(subnet_dir, "subnet_bus_generation_mix.png"))
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


def export_hv1_results_for_season(net, time_steps, output_path, time_lookup, season_name, weather_df=None):
    hv1_dir = os.path.join(output_path, "hv1_network")
    os.makedirs(hv1_dir, exist_ok=True)

    if weather_df is None:
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


def run_season_study(season_name, time_steps, output_root, time_lookup):
    output_path = os.path.join(output_root, season_name)
    net = sb.get_simbench_net(GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    add_configured_electric_boilers(net, abs_vals)
    sb.apply_const_controllers(net, abs_vals)
    prepare_output_writer(net, time_steps, output_path)

    diverged_time_steps.clear()
    try:
        run_timeseries(
            net,
            time_steps=time_steps,
            run=robust_runpp,
            continue_on_divergence=True,
            verbose=True,
        )
    except LoadflowNotConverged as err:
        print(f"Season {season_name} stopped on a non-converged step: {err}")

    _report_diverged(season_name, time_steps, time_lookup)

    weather_df = load_aitolahti_weather(time_steps)
    export_named_results(net, output_path)
    export_subnet_results_for_season(net, abs_vals, time_steps, output_path, time_lookup, season_name, weather_df)
    export_hv1_results_for_season(net, time_steps, output_path, time_lookup, season_name, weather_df)


def _report_diverged(season_name, time_steps, time_lookup):
    if not diverged_time_steps:
        print(f"  [{season_name}] All {len(time_steps)} time steps converged.")
        return
    pct = 100 * len(diverged_time_steps) / len(time_steps)
    print(f"  [{season_name}] {len(diverged_time_steps)}/{len(time_steps)} time steps diverged ({pct:.1f}%).")
    for ts in diverged_time_steps[:10]:
        label = time_lookup.get(ts, ts)
        print(f"    step {ts}: {label}")
    if len(diverged_time_steps) > 10:
        print(f"    ... and {len(diverged_time_steps) - 10} more.")


def run_single_season_study(season_name, output_root=None):
    if season_name not in SEASON_MONTHS:
        raise ValueError(f"Unknown season '{season_name}'. Expected one of: {list(SEASON_MONTHS)}")

    warnings.filterwarnings("ignore", category=FutureWarning)
    output_path = output_root or SEASON_OUTPUT_ROOTS[season_name]

    net = sb.get_simbench_net(GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    max_steps = available_time_steps(abs_vals)
    weather_calendar = load_weather_calendar(max_steps)
    time_lookup = build_time_lookup(weather_calendar)
    time_steps = build_season_time_steps(weather_calendar)[season_name]

    add_configured_electric_boilers(net, abs_vals)
    sb.apply_const_controllers(net, abs_vals)
    prepare_output_writer(net, time_steps, output_path)

    diverged_time_steps.clear()
    try:
        run_timeseries(
            net,
            time_steps=time_steps,
            run=robust_runpp,
            continue_on_divergence=True,
            verbose=True,
        )
    except LoadflowNotConverged as err:
        print(f"Season {season_name} stopped on a non-converged step: {err}")

    _report_diverged(season_name, time_steps, time_lookup)
    weather_df = load_aitolahti_weather(time_steps)
    export_named_results(net, output_path)
    export_subnet_results_for_season(net, abs_vals, time_steps, output_path, time_lookup, season_name, weather_df)

    print(
        f"Finished {season_name} study with {len(time_steps)} time steps. "
        f"Subnet comparison before/after DLR written to: {os.path.join(output_path, SUBNET_DIR_NAME)}"
    )


def build_cross_season_peak_table(season_frames):
    if not season_frames:
        return pd.DataFrame()
    combined = pd.concat(season_frames, ignore_index=True)
    value_columns = ["max_i_ka", "max_loading_without_dlr_percent", "max_loading_with_dlr_percent"]
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


def export_cross_season_summary_tables(output_root):
    subnet_summary = collect_subnet_cross_season_summary(output_root)
    hv1_summary = collect_hv1_cross_season_summary(output_root)
    dlr_summary = collect_seasonal_dlr_benefit_summary(output_root)

    if not subnet_summary.empty:
        subnet_summary.to_csv(os.path.join(output_root, "seasonal_subnet_line_peak_comparison.csv"), index=False)
        plot_cross_season_overview(
            subnet_summary, "Subnet", os.path.join(output_root, "seasonal_subnet_line_peak_comparison.png")
        )
    if not hv1_summary.empty:
        hv1_summary.to_csv(os.path.join(output_root, "seasonal_hv1_line_peak_comparison.csv"), index=False)
        plot_cross_season_overview(
            hv1_summary, "HV1 network", os.path.join(output_root, "seasonal_hv1_line_peak_comparison.png")
        )
    if not dlr_summary.empty:
        dlr_summary.to_csv(os.path.join(output_root, "seasonal_subnet_dlr_summary.csv"), index=False)
        plot_seasonal_dlr_benefit_overview(dlr_summary, os.path.join(output_root, "seasonal_subnet_dlr_summary.png"))


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    max_steps = len(pd.read_csv(AITOLAHTI_WEATHER_CSV, usecols=[0]))
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
