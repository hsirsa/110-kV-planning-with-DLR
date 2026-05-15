import math

import numpy as np
import pandas as pd
import pytest

from dlr.dlr_calc import build_loading_comparison_timeseries, ieee738_ampacity_ka, ieee738_ampacity_ka_batch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_LINE = {
    "dlr_conductor_temp_c": 75.0,
    "diameter_m_est": 0.03,
    "line_azimuth_deg": 0.0,
    "resistance_20c_ohm_per_km": 0.1,
    "alpha_per_c": 0.00403,
    "emissivity": 0.8,
    "absorptivity": 0.8,
    "parallel_count": 1,
}

BASE_WEATHER = {
    "ambient_temp_c": 20.0,
    "wind_speed_mps": 2.0,
    "wind_angle_deg": 90.0,
    "solar_wm2": 0.0,
}


# ---------------------------------------------------------------------------
# ieee738_ampacity_ka  (scalar)
# ---------------------------------------------------------------------------

class TestIeee738AmpacityKaScalar:
    def test_result_is_positive(self):
        assert ieee738_ampacity_ka(BASE_LINE, BASE_WEATHER) > 0.0

    def test_result_in_realistic_range(self):
        result = ieee738_ampacity_ka(BASE_LINE, BASE_WEATHER)
        assert 0.1 < result < 5.0

    def test_higher_wind_raises_ampacity(self):
        low  = ieee738_ampacity_ka(BASE_LINE, {**BASE_WEATHER, "wind_speed_mps": 0.5})
        high = ieee738_ampacity_ka(BASE_LINE, {**BASE_WEATHER, "wind_speed_mps": 8.0})
        assert high > low

    def test_higher_ambient_lowers_ampacity(self):
        cold = ieee738_ampacity_ka(BASE_LINE, {**BASE_WEATHER, "ambient_temp_c": -10.0})
        hot  = ieee738_ampacity_ka(BASE_LINE, {**BASE_WEATHER, "ambient_temp_c": 40.0})
        assert cold > hot

    def test_two_parallel_conductors(self):
        single = ieee738_ampacity_ka({**BASE_LINE, "parallel_count": 1}, BASE_WEATHER)
        double = ieee738_ampacity_ka({**BASE_LINE, "parallel_count": 2}, BASE_WEATHER)
        # Two parallels halve resistance → ampacity scales by sqrt(2)
        assert double == pytest.approx(single * math.sqrt(2), rel=1e-4)

    def test_solar_load_reduces_ampacity(self):
        no_sun  = ieee738_ampacity_ka(BASE_LINE, {**BASE_WEATHER, "solar_wm2": 0.0})
        full_sun = ieee738_ampacity_ka(BASE_LINE, {**BASE_WEATHER, "solar_wm2": 1000.0})
        assert no_sun > full_sun

    def test_extreme_solar_returns_zero(self):
        # Solar so large the heat balance goes negative → must return 0.0
        result = ieee738_ampacity_ka(
            {**BASE_LINE, "absorptivity": 1.0},
            {**BASE_WEATHER, "solar_wm2": 1e9},
        )
        assert result == 0.0

    def test_conductor_at_ambient_temp_still_positive(self):
        # delta_t floored at 0.1 — function must not return 0 or raise
        line = {**BASE_LINE, "dlr_conductor_temp_c": 20.0}
        result = ieee738_ampacity_ka(line, BASE_WEATHER)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# ieee738_ampacity_ka_batch  (vectorised)
# ---------------------------------------------------------------------------

class TestIeee738AmpacityKaBatch:
    def _make_df(self, overrides=None):
        row = {**BASE_LINE, **BASE_WEATHER}
        if overrides:
            row.update(overrides)
        return pd.DataFrame([row])

    def test_single_row_matches_scalar(self):
        df = self._make_df()
        batch  = ieee738_ampacity_ka_batch(df)[0]
        scalar = ieee738_ampacity_ka(BASE_LINE, BASE_WEATHER)
        assert batch == pytest.approx(scalar, rel=1e-6)

    def test_batch_matches_scalar_across_conditions(self):
        rows = [
            {**BASE_LINE, **BASE_WEATHER, "ambient_temp_c": t, "wind_speed_mps": w}
            for t in [0.0, 10.0, 20.0, 35.0]
            for w in [0.5, 2.0, 7.0]
        ]
        df = pd.DataFrame(rows)
        batch = ieee738_ampacity_ka_batch(df)
        for i, row in enumerate(rows):
            line_row    = {k: row[k] for k in BASE_LINE}
            weather_row = {k: row[k] for k in BASE_WEATHER}
            # Override weather wind/angle values that were written into row
            weather_row["wind_speed_mps"] = row["wind_speed_mps"]
            weather_row["ambient_temp_c"] = row["ambient_temp_c"]
            scalar = ieee738_ampacity_ka(line_row, weather_row)
            assert batch[i] == pytest.approx(scalar, rel=1e-5), f"Mismatch at row {i}"

    def test_extreme_solar_returns_zero(self):
        df = self._make_df({"absorptivity": 1.0, "solar_wm2": 1e9})
        assert ieee738_ampacity_ka_batch(df)[0] == 0.0

    def test_output_length_matches_input(self):
        rows = [{**BASE_LINE, **BASE_WEATHER}] * 7
        df = pd.DataFrame(rows)
        assert len(ieee738_ampacity_ka_batch(df)) == 7


# ---------------------------------------------------------------------------
# build_loading_comparison_timeseries
# ---------------------------------------------------------------------------

class TestBuildLoadingComparisonTimeseries:
    @pytest.fixture
    def line_loading(self):
        return pd.DataFrame({
            "time_step":        [0, 1, 0, 1],
            "line_index":       [0, 0, 1, 1],
            "name":             ["L0", "L0", "L1", "L1"],
            "loading_percent":  [60.0, 70.0, 40.0, 50.0],
        })

    @pytest.fixture
    def dlr_df(self):
        return pd.DataFrame({
            "time_step":               [0, 1, 0, 1],
            "line_index":              [0, 0, 1, 1],
            "name":                    ["L0", "L0", "L1", "L1"],
            "hour_of_day":             [0.0, 0.25, 0.0, 0.25],
            "dlr_utilization_percent": [45.0, 50.0, 35.0, 38.0],
        })

    def test_output_columns(self, line_loading, dlr_df):
        result = build_loading_comparison_timeseries(line_loading, dlr_df)
        assert "loading_percent_without_dlr" in result.columns
        assert "loading_percent_with_dlr" in result.columns
        assert "dlr_benefit_percent_points" in result.columns

    def test_benefit_calculation(self, line_loading, dlr_df):
        result = build_loading_comparison_timeseries(line_loading, dlr_df)
        row = result[(result["time_step"] == 0) & (result["line_index"] == 0)].iloc[0]
        assert row["loading_percent_without_dlr"] == pytest.approx(60.0)
        assert row["loading_percent_with_dlr"]    == pytest.approx(45.0)
        assert row["dlr_benefit_percent_points"]  == pytest.approx(15.0)

    def test_empty_line_loading_returns_empty(self, dlr_df):
        result = build_loading_comparison_timeseries(pd.DataFrame(), dlr_df)
        assert result.empty

    def test_empty_dlr_returns_empty(self, line_loading):
        result = build_loading_comparison_timeseries(line_loading, pd.DataFrame())
        assert result.empty

    def test_row_count(self, line_loading, dlr_df):
        result = build_loading_comparison_timeseries(line_loading, dlr_df)
        assert len(result) == len(line_loading)
