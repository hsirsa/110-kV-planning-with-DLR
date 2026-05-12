import os

import pandas as pd


def export_result_with_names(csv_path, meta, name_column, index_label, value_column, output_path):
    df = pd.read_csv(csv_path, sep=";", decimal=".", index_col=0)
    df.index.name = "time_step"
    df.columns = [int(col) for col in df.columns]
    meta_export = meta.copy()
    meta_export[index_label] = meta_export.index
    meta_export[name_column] = meta_export[name_column].fillna("").astype(str)
    long_df = df.reset_index().melt(id_vars=["time_step"], var_name=index_label, value_name=value_column)
    merged = long_df.merge(meta_export[[index_label, name_column]], on=index_label, how="left")
    merged = merged[["time_step", index_label, name_column, value_column]]
    merged.to_csv(output_path, index=False)


def export_named_results(net, output_path):
    export_result_with_names(
        os.path.join(output_path, "res_bus", "vm_pu.csv"), net.bus, "name", "bus_index", "vm_pu",
        os.path.join(output_path, "bus_vm_pu_named.csv"),
    )
    export_result_with_names(
        os.path.join(output_path, "res_line", "i_ka.csv"), net.line, "name", "line_index", "i_ka",
        os.path.join(output_path, "line_i_ka_named.csv"),
    )
    export_result_with_names(
        os.path.join(output_path, "res_line", "loading_percent.csv"), net.line, "name", "line_index", "loading_percent",
        os.path.join(output_path, "line_loading_percent_named.csv"),
    )
    export_result_with_names(
        os.path.join(output_path, "res_trafo", "loading_percent.csv"), net.trafo, "name", "trafo_index", "loading_percent",
        os.path.join(output_path, "trafo_loading_percent_named.csv"),
    )


def filter_named_results(csv_path, index_label, allowed_indices):
    if not allowed_indices:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df[df[index_label].isin(allowed_indices)].copy()
