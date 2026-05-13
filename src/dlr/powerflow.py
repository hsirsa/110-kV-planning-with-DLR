import contextlib
import logging
import os
import warnings

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries.output_writer import OutputWriter


@contextlib.contextmanager
def _silence_logger(logger):
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(old_level)


_SOLVER_OPTIONS = [
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

# Tracks which time steps failed all solver attempts (populated during run_timeseries).
diverged_time_steps: list[int] = []


def robust_runpp(net, **kwargs):
    last_error = None
    for options in _SOLVER_OPTIONS:
        current_options = {**options, **kwargs}
        try:
            pp_logger = logging.getLogger("pandapower")
            with warnings.catch_warnings(), _silence_logger(pp_logger):
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message=".*Matrix is exactly singular.*")
                pp.runpp(net, **current_options)
            return
        except Exception as err:
            last_error = err

    # All three solvers failed — record and re-raise so run_timeseries can handle it.
    time_step = getattr(net, "_ppc", {}).get("time_step", "?")
    diverged_time_steps.append(time_step)
    raise LoadflowNotConverged(f"All solver attempts failed. Last error: {last_error}")


def prepare_output_writer(net, time_steps, output_path):
    os.makedirs(output_path, exist_ok=True)
    ow = OutputWriter(net, time_steps=time_steps, output_path=output_path, output_file_type=".csv")
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "i_ka")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    return ow
