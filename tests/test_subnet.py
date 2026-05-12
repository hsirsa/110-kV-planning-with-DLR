import math
import types

import numpy as np
import pandas as pd
import pytest

from dlr.subnet import (
    aggregate_count_by_bus,
    aggregate_sum_by_bus,
    build_line_conductor_id,
    classify_subnet_bus,
    compute_line_azimuth_deg,
    infer_sgen_generation_type,
)


# ---------------------------------------------------------------------------
# classify_subnet_bus
# ---------------------------------------------------------------------------

class TestClassifySubnetBus:
    def test_load_dominated(self):
        assert classify_subnet_bus(100.0, 20.0, 0.0, 0.0) == "Load-dominated"

    def test_generation_dominated_pv(self):
        assert classify_subnet_bus(10.0, 100.0, 80.0, 0.0) == "Generation-dominated (PV/Wind)"

    def test_generation_dominated_wind(self):
        assert classify_subnet_bus(10.0, 100.0, 0.0, 80.0) == "Generation-dominated (PV/Wind)"

    def test_generation_dominated_other(self):
        assert classify_subnet_bus(10.0, 100.0, 0.0, 0.0) == "Generation-dominated (Other)"

    def test_mixed_equal_load_gen(self):
        assert classify_subnet_bus(100.0, 100.0, 0.0, 0.0) == "Mixed"

    def test_both_zero_is_mixed(self):
        assert classify_subnet_bus(0.0, 0.0, 0.0, 0.0) == "Mixed"

    def test_none_inputs_treated_as_zero(self):
        assert classify_subnet_bus(None, None, None, None) == "Mixed"

    def test_just_above_load_threshold(self):
        # generation > load * 1.2 → generation-dominated
        assert classify_subnet_bus(10.0, 13.0, 0.0, 0.0) == "Generation-dominated (Other)"


# ---------------------------------------------------------------------------
# compute_line_azimuth_deg
# ---------------------------------------------------------------------------

class TestComputeLineAzimuthDeg:
    def _net(self):
        return types.SimpleNamespace()

    def test_missing_bus_returns_default(self):
        assert compute_line_azimuth_deg(self._net(), 0, 1, pos_map={}) == 90.0

    def test_east_direction(self):
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        assert compute_line_azimuth_deg(self._net(), 0, 1, pos_map=pos) == pytest.approx(0.0, abs=1e-6)

    def test_north_direction(self):
        pos = {0: (0.0, 0.0), 1: (0.0, 1.0)}
        assert compute_line_azimuth_deg(self._net(), 0, 1, pos_map=pos) == pytest.approx(90.0, abs=1e-6)

    def test_west_direction_wraps_to_0_to_180(self):
        pos = {0: (1.0, 0.0), 1: (0.0, 0.0)}
        # atan2(0, -1) = 180° → 180 % 180 = 0
        result = compute_line_azimuth_deg(self._net(), 0, 1, pos_map=pos)
        assert 0.0 <= result < 180.0

    def test_diagonal(self):
        pos = {0: (0.0, 0.0), 1: (1.0, 1.0)}
        result = compute_line_azimuth_deg(self._net(), 0, 1, pos_map=pos)
        assert result == pytest.approx(45.0, abs=1e-6)


# ---------------------------------------------------------------------------
# aggregate_count_by_bus / aggregate_sum_by_bus
# ---------------------------------------------------------------------------

class TestAggregateFunctions:
    def test_count_by_bus(self):
        df = pd.DataFrame({"bus": [0, 0, 1, 2]})
        result = aggregate_count_by_bus(df, "bus", [0, 1, 2])
        assert result == [2, 1, 1]

    def test_count_missing_bus_returns_zero(self):
        df = pd.DataFrame({"bus": [0]})
        result = aggregate_count_by_bus(df, "bus", [0, 99])
        assert result == [1, 0]

    def test_count_empty_df(self):
        result = aggregate_count_by_bus(pd.DataFrame(), "bus", [0, 1])
        assert result == [0, 0]

    def test_sum_by_bus(self):
        df = pd.DataFrame({"bus": [0, 0, 1], "p_mw": [10.0, 20.0, 5.0]})
        result = aggregate_sum_by_bus(df, "bus", "p_mw", [0, 1])
        assert result == pytest.approx([30.0, 5.0])

    def test_sum_missing_bus_is_zero(self):
        df = pd.DataFrame({"bus": [0], "p_mw": [10.0]})
        result = aggregate_sum_by_bus(df, "bus", "p_mw", [0, 99])
        assert result == pytest.approx([10.0, 0.0])

    def test_sum_empty_df(self):
        result = aggregate_sum_by_bus(pd.DataFrame(), "bus", "p_mw", [0, 1])
        assert result == pytest.approx([0.0, 0.0])

    def test_sum_missing_value_column(self):
        df = pd.DataFrame({"bus": [0]})
        result = aggregate_sum_by_bus(df, "bus", "p_mw", [0])
        assert result == pytest.approx([0.0])


# ---------------------------------------------------------------------------
# infer_sgen_generation_type
# ---------------------------------------------------------------------------

class TestInferSgenGenerationType:
    def test_pv_from_name(self):
        row = pd.Series({"type": "PV_profile"})
        assert infer_sgen_generation_type(row) == "pv"

    def test_solar_keyword(self):
        row = pd.Series({"profile": "solar_2023"})
        assert infer_sgen_generation_type(row) == "pv"

    def test_wind_keyword(self):
        row = pd.Series({"type": "wind_farm"})
        assert infer_sgen_generation_type(row) == "wind"

    def test_unknown_returns_other(self):
        row = pd.Series({"type": "CHP_plant"})
        assert infer_sgen_generation_type(row) == "other"

    def test_no_candidate_columns(self):
        row = pd.Series({"bus": 5, "p_mw": 10.0})
        assert infer_sgen_generation_type(row) == "other"


# ---------------------------------------------------------------------------
# build_line_conductor_id
# ---------------------------------------------------------------------------

class TestBuildLineConductorId:
    def test_uses_std_type_when_present(self):
        df = pd.DataFrame({
            "std_type":        ["NAYY 4x50 SE"],
            "r_ohm_per_km":   [0.641],
            "x_ohm_per_km":   [0.083],
            "max_i_ka":       [0.142],
        })
        result = build_line_conductor_id(df)
        assert result[0] == "NAYY 4x50 SE"

    def test_falls_back_to_r_x_imax(self):
        df = pd.DataFrame({
            "std_type":        [None],
            "r_ohm_per_km":   [0.1],
            "x_ohm_per_km":   [0.4],
            "max_i_ka":       [0.5],
        })
        result = build_line_conductor_id(df)
        assert "r=" in result[0]
        assert "x=" in result[0]
        assert "imax=" in result[0]

    def test_mixed_rows(self):
        df = pd.DataFrame({
            "std_type":       ["TypeA", None],
            "r_ohm_per_km":  [0.0, 0.2],
            "x_ohm_per_km":  [0.0, 0.3],
            "max_i_ka":      [0.0, 0.4],
        })
        result = build_line_conductor_id(df)
        assert result[0] == "TypeA"
        assert "r=" in result[1]
