import numpy as np
import pandas as pd
import pytest

from dlr.analysis import (
    attach_rank_labels,
    attach_season_position,
    attach_time_utc,
    build_season_axis_lookup,
    build_subnet_generation_analysis,
    build_subnet_loading_summary,
    build_subnet_voltage_summary,
    summarize_bus_vm,
    summarize_metric,
    top_n_time_points,
)


# ---------------------------------------------------------------------------
# attach_rank_labels
# ---------------------------------------------------------------------------

class TestAttachRankLabels:
    def test_labels_use_prefix_and_1_based_index(self):
        df = pd.DataFrame({"value": [3, 1, 2]})
        result = attach_rank_labels(df, "T")
        assert list(result["rank_label"]) == ["T1", "T2", "T3"]

    def test_original_order_is_preserved(self):
        df = pd.DataFrame({"value": [10, 20]})
        result = attach_rank_labels(df, "X")
        assert result["value"].tolist() == [10, 20]

    def test_empty_df(self):
        result = attach_rank_labels(pd.DataFrame(), "T")
        assert result.empty


# ---------------------------------------------------------------------------
# top_n_time_points
# ---------------------------------------------------------------------------

class TestTopNTimePoints:
    def test_returns_n_rows(self):
        df = pd.DataFrame({"time_step": range(20), "value": range(20)})
        assert len(top_n_time_points(df, "value", n=5)) == 5

    def test_top_row_is_largest(self):
        df = pd.DataFrame({"time_step": range(10), "value": range(10)})
        result = top_n_time_points(df, "value", n=3)
        assert result["value"].iloc[0] == 9

    def test_returns_all_if_fewer_than_n(self):
        df = pd.DataFrame({"time_step": [0, 1], "value": [5.0, 3.0]})
        assert len(top_n_time_points(df, "value", n=10)) == 2

    def test_empty_df_returns_empty(self):
        assert top_n_time_points(pd.DataFrame(), "value").empty

    def test_missing_column_returns_empty(self):
        df = pd.DataFrame({"time_step": [0, 1]})
        assert top_n_time_points(df, "nonexistent").empty

    def test_descending_order(self):
        df = pd.DataFrame({"time_step": range(5), "value": [3.0, 1.0, 5.0, 2.0, 4.0]})
        result = top_n_time_points(df, "value", n=3)
        assert result["value"].tolist() == [5.0, 4.0, 3.0]


# ---------------------------------------------------------------------------
# summarize_metric / summarize_bus_vm
# ---------------------------------------------------------------------------

class TestSummarizeMetric:
    def test_min_max_mean(self):
        df = pd.DataFrame({"line_index": [0, 0, 1, 1], "i_ka": [0.5, 1.0, 0.2, 0.8]})
        result = summarize_metric(df, "line_index", "i_ka")
        assert result.at[0, "max"] == pytest.approx(1.0)
        assert result.at[0, "min"] == pytest.approx(0.5)
        assert result.at[0, "mean"] == pytest.approx(0.75)
        assert result.at[1, "min"] == pytest.approx(0.2)

    def test_empty_returns_empty(self):
        assert summarize_metric(pd.DataFrame(), "line_index", "i_ka").empty


class TestSummarizeBusVm:
    def test_groups_by_bus_index(self):
        bus_vm = pd.DataFrame({
            "bus_index": [0, 0, 1],
            "vm_pu": [1.0, 1.05, 0.98],
        })
        result = summarize_bus_vm(bus_vm)
        assert result.at[0, "max"] == pytest.approx(1.05)
        assert result.at[0, "min"] == pytest.approx(1.0)
        assert result.at[1, "mean"] == pytest.approx(0.98)

    def test_empty_returns_empty(self):
        assert summarize_bus_vm(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# build_subnet_voltage_summary
# ---------------------------------------------------------------------------

class TestBuildSubnetVoltageSummary:
    def test_output_columns(self):
        bus_vm = pd.DataFrame({"time_step": [0, 0, 1, 1], "vm_pu": [1.0, 1.05, 0.98, 1.02]})
        result = build_subnet_voltage_summary(bus_vm)
        assert "max_vm_pu" in result.columns
        assert "mean_vm_pu" in result.columns

    def test_max_per_step(self):
        bus_vm = pd.DataFrame({"time_step": [0, 0], "vm_pu": [1.0, 1.05]})
        result = build_subnet_voltage_summary(bus_vm)
        assert result.loc[result["time_step"] == 0, "max_vm_pu"].iloc[0] == pytest.approx(1.05)

    def test_empty_returns_columns(self):
        result = build_subnet_voltage_summary(pd.DataFrame())
        assert list(result.columns) == ["time_step", "max_vm_pu", "mean_vm_pu"]


# ---------------------------------------------------------------------------
# build_subnet_loading_summary
# ---------------------------------------------------------------------------

class TestBuildSubnetLoadingSummary:
    @pytest.fixture
    def loading_df(self):
        return pd.DataFrame({
            "time_step": [0, 0, 1, 1],
            "loading_percent_without_dlr": [60.0, 50.0, 70.0, 80.0],
            "loading_percent_with_dlr":    [45.0, 40.0, 55.0, 60.0],
            "dlr_benefit_percent_points":  [15.0, 10.0, 15.0, 20.0],
        })

    def test_max_loading_without_dlr(self, loading_df):
        result = build_subnet_loading_summary(loading_df)
        assert result.loc[result["time_step"] == 0, "max_loading_without_dlr_percent"].iloc[0] == pytest.approx(60.0)

    def test_mean_dlr_benefit(self, loading_df):
        result = build_subnet_loading_summary(loading_df)
        assert result.loc[result["time_step"] == 0, "mean_dlr_benefit_percent_points"].iloc[0] == pytest.approx(12.5)

    def test_empty_returns_columns(self):
        result = build_subnet_loading_summary(pd.DataFrame())
        assert "time_step" in result.columns


# ---------------------------------------------------------------------------
# build_subnet_generation_analysis
# ---------------------------------------------------------------------------

class TestBuildSubnetGenerationAnalysis:
    @pytest.fixture
    def injection_df(self):
        return pd.DataFrame({
            "time_step": [0, 0, 1, 1],
            "source_type": ["sgen", "load", "sgen", "load"],
            "p_mw": [50.0, 30.0, 60.0, 40.0],
        })

    @pytest.fixture
    def bus_vm(self):
        return pd.DataFrame({
            "time_step": [0, 0, 1, 1],
            "bus_index": [0, 1, 0, 1],
            "vm_pu": [1.02, 1.05, 0.98, 1.01],
        })

    def test_net_injection_calculation(self, injection_df, bus_vm):
        result = build_subnet_generation_analysis(injection_df, bus_vm)
        row0 = result[result["time_step"] == 0].iloc[0]
        assert row0["total_generation_mw"] == pytest.approx(50.0)
        assert row0["total_load_mw"] == pytest.approx(30.0)
        assert row0["net_injection_mw"] == pytest.approx(20.0)

    def test_max_vm_per_step(self, injection_df, bus_vm):
        result = build_subnet_generation_analysis(injection_df, bus_vm)
        row0 = result[result["time_step"] == 0].iloc[0]
        assert row0["max_vm_pu"] == pytest.approx(1.05)

    def test_empty_returns_columns(self):
        result = build_subnet_generation_analysis(pd.DataFrame(), pd.DataFrame())
        assert list(result.columns) == ["time_step", "total_generation_mw", "total_load_mw", "net_injection_mw", "max_vm_pu"]


# ---------------------------------------------------------------------------
# attach_time_utc
# ---------------------------------------------------------------------------

class TestAttachTimeUtc:
    @pytest.fixture
    def time_lookup(self):
        return pd.Series(
            pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:15:00"], utc=True),
            index=[0, 1],
        )

    def test_column_added(self, time_lookup):
        df = pd.DataFrame({"time_step": [0, 1], "value": [1.0, 2.0]})
        result = attach_time_utc(df, time_lookup)
        assert "time_utc" in result.columns

    def test_correct_timestamp(self, time_lookup):
        df = pd.DataFrame({"time_step": [0], "value": [1.0]})
        result = attach_time_utc(df, time_lookup)
        assert result["time_utc"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")

    def test_empty_returns_empty(self, time_lookup):
        result = attach_time_utc(pd.DataFrame(), time_lookup)
        assert result.empty

    def test_no_time_step_column_returns_copy(self, time_lookup):
        df = pd.DataFrame({"value": [1.0]})
        result = attach_time_utc(df, time_lookup)
        assert "time_utc" not in result.columns


# ---------------------------------------------------------------------------
# build_season_axis_lookup / attach_season_position
# ---------------------------------------------------------------------------

class TestBuildSeasonAxisLookup:
    @pytest.fixture
    def time_lookup(self):
        times = pd.date_range("2023-01-01", periods=200, freq="15min", tz="UTC")
        return pd.Series(times, index=range(200))

    def test_season_position_is_sequential(self, time_lookup):
        time_steps = list(range(96))
        axis_df, month_labels, month_positions = build_season_axis_lookup(time_steps, time_lookup)
        assert axis_df["season_position"].is_monotonic_increasing

    def test_returns_three_items(self, time_lookup):
        result = build_season_axis_lookup(list(range(10)), time_lookup)
        assert len(result) == 3

    def test_month_labels_and_positions_same_length(self, time_lookup):
        _, month_labels, month_positions = build_season_axis_lookup(list(range(200)), time_lookup)
        assert len(month_labels) == len(month_positions)

    def test_attach_season_position_merges(self, time_lookup):
        time_steps = list(range(10))
        axis_df, _, _ = build_season_axis_lookup(time_steps, time_lookup)
        df = pd.DataFrame({"time_step": [0, 1, 2]})
        result = attach_season_position(df, axis_df)
        assert "season_position" in result.columns
