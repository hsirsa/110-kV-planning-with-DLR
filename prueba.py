import os
import warnings

import pandas as pd
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
import simbench as sb


GRID_CODE = "1-HV-mixed--0-no_sw"
TIME_STEP_HOURS = 0.25


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


def available_time_steps(abs_vals):
    return len(abs_vals[("load", "p_mw")].index)


def prepare_output_writer(net, time_steps, output_path):
    os.makedirs(output_path, exist_ok=True)

    ow = OutputWriter(
        net,
        time_steps=time_steps,
        output_path=output_path,
        output_file_type=".csv",
    )
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "i_ka")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    return ow


def export_result_with_names(csv_path, meta, name_column, index_label, value_column, output_path):
    df = pd.read_csv(csv_path, sep=";", decimal=".", index_col=0)
    df.index.name = "time_step"
    df.columns = [int(col) for col in df.columns]

    meta_export = meta.copy()
    meta_export[index_label] = meta_export.index
    meta_export[name_column] = meta_export[name_column].fillna("").astype(str)

    long_df = df.reset_index().melt(
        id_vars=["time_step"],
        var_name=index_label,
        value_name=value_column,
    )
    merged = long_df.merge(meta_export[[index_label, name_column]], on=index_label, how="left")
    merged = merged[["time_step", index_label, name_column, value_column]]
    merged.to_csv(output_path, index=False)


def export_named_results(net, output_path):
    export_result_with_names(
        csv_path=os.path.join(output_path, "res_bus", "vm_pu.csv"),
        meta=net.bus,
        name_column="name",
        index_label="bus_index",
        value_column="vm_pu",
        output_path=os.path.join(output_path, "bus_vm_pu_named.csv"),
    )

    export_result_with_names(
        csv_path=os.path.join(output_path, "res_line", "i_ka.csv"),
        meta=net.line,
        name_column="name",
        index_label="line_index",
        value_column="i_ka",
        output_path=os.path.join(output_path, "line_i_ka_named.csv"),
    )

    export_result_with_names(
        csv_path=os.path.join(output_path, "res_line", "loading_percent.csv"),
        meta=net.line,
        name_column="name",
        index_label="line_index",
        value_column="loading_percent",
        output_path=os.path.join(output_path, "line_loading_percent_named.csv"),
    )

    export_result_with_names(
        csv_path=os.path.join(output_path, "res_trafo", "loading_percent.csv"),
        meta=net.trafo,
        name_column="name",
        index_label="trafo_index",
        value_column="loading_percent",
        output_path=os.path.join(output_path, "trafo_loading_percent_named.csv"),
    )


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    net = sb.get_simbench_net(GRID_CODE)

    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, abs_vals)

    max_steps = available_time_steps(abs_vals)
    day_steps = min(int(24 / TIME_STEP_HOURS), max_steps)
    time_steps = list(range(day_steps))

    output_path = "results_1day"
    prepare_output_writer(net, time_steps, output_path)

    try:
        run_timeseries(
            net,
            time_steps=time_steps,
            run=robust_runpp,
            continue_on_divergence=False,
            verbose=True,
        )
    except LoadflowNotConverged as err:
        print(f"Time-series power flow did not converge: {err}")
        return

    export_named_results(net, output_path)
    print("1-day study finished. Results written to", output_path)


if __name__ == "__main__":
    main()
