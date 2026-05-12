import warnings

import simbench as sb
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries import run_timeseries

from .config import GRID_CODE, TIME_STEP_HOURS
from .export import export_named_results
from .network import available_time_steps
from .powerflow import prepare_output_writer, robust_runpp


def run_day_study(output_path="results_1day"):
    """1-day (96 time step) power flow, no boilers, no DLR."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    net = sb.get_simbench_net(GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, abs_vals)
    max_steps = available_time_steps(abs_vals)
    day_steps = min(int(24 / TIME_STEP_HOURS), max_steps)
    time_steps = list(range(day_steps))
    prepare_output_writer(net, time_steps, output_path)
    try:
        run_timeseries(net, time_steps=time_steps, run=robust_runpp, continue_on_divergence=False, verbose=True)
    except LoadflowNotConverged as err:
        print(f"Power flow did not converge: {err}")
        return
    export_named_results(net, output_path)
    print(f"1-day study finished. Results written to {output_path}")


def run_year_study(output_path="results_1year"):
    """Full 1-year power flow, no boilers, no DLR, no seasonal split."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    net = sb.get_simbench_net(GRID_CODE)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, abs_vals)
    time_steps = list(range(available_time_steps(abs_vals)))
    prepare_output_writer(net, time_steps, output_path)
    try:
        run_timeseries(net, time_steps=time_steps, run=robust_runpp, continue_on_divergence=False, verbose=True)
    except LoadflowNotConverged as err:
        print(f"Power flow did not converge: {err}")
        return
    export_named_results(net, output_path)
    print(f"1-year study finished. Results written to {output_path}")
