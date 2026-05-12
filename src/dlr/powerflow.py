import os

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.timeseries.output_writer import OutputWriter


def robust_runpp(net, **kwargs):
    solver_options = [
        {"algorithm": "nr", "init": "dc", "check_connectivity": True,
         "calculate_voltage_angles": True, "max_iteration": 50, "tolerance_mva": 1e-6, "numba": False},
        {"algorithm": "nr", "init": "flat", "check_connectivity": True,
         "calculate_voltage_angles": True, "max_iteration": 50, "tolerance_mva": 1e-6, "numba": False},
        {"algorithm": "fdbx", "init": "flat", "check_connectivity": True,
         "calculate_voltage_angles": True, "max_iteration": 100, "tolerance_mva": 1e-6, "numba": False},
    ]
    last_error = None
    for options in solver_options:
        current_options = {**options, **kwargs}
        try:
            pp.runpp(net, **current_options)
            return
        except Exception as err:
            last_error = err
    raise LoadflowNotConverged(f"All solver attempts failed. Last error: {last_error}")


def prepare_output_writer(net, time_steps, output_path):
    os.makedirs(output_path, exist_ok=True)
    ow = OutputWriter(net, time_steps=time_steps, output_path=output_path, output_file_type=".csv")
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "i_ka")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    return ow
