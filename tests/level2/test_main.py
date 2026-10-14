import numpy as np
import pytest
import xarray as xr

from src.level2.main import (
    check_plasma_current,
    create_parser,
    get_sample_durations,
    is_grid_consistent_with_uniform_sampling,
    set_mapping_time_bounds,
    time_weighted_running_mean,
    trim_ip_range,
)


def make_plasma_current(cadence: float, high_current_duration: float) -> xr.DataArray:
    time = np.arange(-0.1, 0.3 + cadence / 2, cadence)
    values = np.where(
        (time >= 0) & (time < high_current_duration),
        250_000.0,
        0.0,
    )
    return xr.DataArray(values, dims=["time"], coords={"time": time})


def make_multirate_plasma_current(high_current_duration: float) -> xr.DataArray:
    time = np.concatenate(
        (
            np.arange(-0.1, 0.0, 0.001),
            np.arange(0.0, 0.2, 0.0002),
            np.arange(0.2, 0.301, 0.001),
        )
    )
    values = np.where(
        (time >= 0) & (time < high_current_duration),
        250_000.0,
        0.0,
    )
    return xr.DataArray(values, dims=["time"], coords={"time": time})


def make_union_grid_plasma_current(high_current_duration: float) -> xr.DataArray:
    fine = make_plasma_current(0.0002, high_current_duration)
    coarse_time = np.arange(-0.1, 0.301, 0.001) + 5e-8
    time = np.concatenate((fine.time.values, coarse_time))
    values = np.concatenate((fine.values, np.full(coarse_time.size, np.nan)))
    order = np.argsort(time)
    return xr.DataArray(values[order], dims=["time"], coords={"time": time[order]})


@pytest.mark.parametrize(
    ("high_current_duration", "expected"),
    [(0.05, False), (0.07, True)],
)
def test_plasma_current_gate_is_invariant_across_native_cadences(
    high_current_duration, expected
):
    results = [
        check_plasma_current(make_plasma_current(cadence, high_current_duration))
        for cadence in (0.0002, 0.00025, 0.0005, 0.001)
    ]

    assert results == [expected] * 4


@pytest.mark.parametrize(
    ("high_current_duration", "expected"),
    [(0.05, False), (0.07, True)],
)
def test_plasma_current_gate_is_invariant_on_multirate_and_union_grids(
    high_current_duration, expected
):
    traces = (
        make_plasma_current(0.0002, high_current_duration),
        make_multirate_plasma_current(high_current_duration),
        make_union_grid_plasma_current(high_current_duration),
    )

    assert [check_plasma_current(trace) for trace in traces] == [expected] * 3


def test_float32_two_rate_grid_uses_physical_interval_durations():
    ideal_widths = np.resize([0.00024995, 0.00025005], 999)
    time = np.concatenate(([0.0], np.cumsum(ideal_widths))).astype(np.float32)
    values = np.zeros(time.size)
    long_intervals = np.flatnonzero(ideal_widths > ideal_widths.mean())[:250]
    values[long_intervals] = 250_000.0
    plasma_current = xr.DataArray(values, dims=["time"], coords={"time": time})
    plasma_current, time, durations, _ = get_sample_durations(plasma_current)
    observed_duration = durations[:-1][long_intervals].sum()

    assert observed_duration == pytest.approx(0.06251251042704098)
    assert not is_grid_consistent_with_uniform_sampling(plasma_current.time.dtype, time)
    assert check_plasma_current(plasma_current)


@pytest.mark.parametrize(
    ("cadence", "half_spread", "offset", "high_interval_count"),
    [
        (0.00025, 0.000000125, 2.0, 250),
        (0.0002, 0.0000004, 8.0, 312),
    ],
)
def test_float32_observable_multirate_grids_are_not_treated_as_uniform(
    cadence, half_spread, offset, high_interval_count
):
    ideal_widths = np.resize([cadence - half_spread, cadence + half_spread], 999)
    time = (offset + np.concatenate(([0.0], np.cumsum(ideal_widths)))).astype(
        np.float32
    )
    long_intervals = np.flatnonzero(ideal_widths > cadence)[:high_interval_count]
    values = np.zeros(time.size)
    values[long_intervals] = 250_000.0
    plasma_current = xr.DataArray(values, dims=["time"], coords={"time": time})
    plasma_current, parsed_time, durations, _ = get_sample_durations(plasma_current)

    assert durations[:-1][long_intervals].sum() > 0.0625
    assert not is_grid_consistent_with_uniform_sampling(
        plasma_current.time.dtype, parsed_time
    )
    assert check_plasma_current(plasma_current)


@pytest.mark.parametrize(
    ("offset", "half_spread", "expected_duration"),
    [
        (-100.0, 1.1220184543019653e-13, 0.06250000002803802),
        (-0.1, 4.466835921509635e-16, 0.0625000000001111),
    ],
)
def test_float64_rate_change_is_not_hidden_by_timeline_arithmetic(
    offset, half_spread, expected_duration
):
    cadence = 0.00025
    ideal_widths = np.resize([cadence - half_spread, cadence + half_spread], 999)
    time = offset + np.concatenate(([0.0], np.cumsum(ideal_widths)))
    long_intervals = np.flatnonzero(ideal_widths > cadence)[:250]
    values = np.zeros(time.size)
    values[long_intervals] = 250_000.0
    plasma_current = xr.DataArray(values, dims=["time"], coords={"time": time})
    plasma_current, parsed_time, durations, _ = get_sample_durations(plasma_current)

    assert durations[:-1][long_intervals].sum() == pytest.approx(expected_duration)
    assert not is_grid_consistent_with_uniform_sampling(
        plasma_current.time.dtype, parsed_time
    )
    assert check_plasma_current(plasma_current)


def test_plasma_current_gate_preserves_strict_legacy_duration():
    time = np.arange(400) * 0.00025
    values = np.zeros(time.size)
    values[:250] = 250_000.0
    plasma_current = xr.DataArray(values, dims=["time"], coords={"time": time})

    assert not check_plasma_current(plasma_current)

    plasma_current.values[250] = 250_000.0
    assert check_plasma_current(plasma_current)


def test_float32_quantisation_preserves_strict_legacy_duration():
    time = (-2.0615 + np.arange(400) * 0.00025).astype(np.float32)
    values = np.zeros(time.size)
    values[:250] = 250_000.0
    plasma_current = xr.DataArray(values, dims=["time"], coords={"time": time})

    # The float32 endpoints measure this nominal 62.5 ms span as
    # 62.500119 ms; timestamp precision must not turn equality into a pass.
    assert not check_plasma_current(plasma_current)

    plasma_current.values[250] = 250_000.0
    assert check_plasma_current(plasma_current)


@pytest.mark.parametrize("offset", [-100.0, -2.5, 0.0, 4.0, 4.522943873, 50.0])
def test_float32_widest_intervals_do_not_turn_250_samples_into_a_pass(offset):
    time = (offset + np.arange(1000) * 0.00025).astype(np.float32)
    plasma_current = xr.DataArray(
        np.zeros(time.size), dims=["time"], coords={"time": time}
    )
    _, _, durations, _ = get_sample_durations(plasma_current)
    interval_order = np.argsort(durations[:-1])
    widest_250 = interval_order[-250:]
    plasma_current.values[widest_250] = 250_000.0

    assert durations[widest_250].sum() > 0.0625
    assert not check_plasma_current(plasma_current)

    plasma_current.values[interval_order[-251]] = 250_000.0
    assert check_plasma_current(plasma_current)


def test_terminal_sample_does_not_invent_qualification_duration():
    plasma_current = xr.DataArray(
        [0.0, 250_000.0],
        dims=["time"],
        coords={"time": [0.0, 0.063]},
    )

    assert not check_plasma_current(plasma_current)


def test_smoothing_uses_the_full_75_ms_window_at_0_2_ms():
    time = np.arange(500) * 0.0002
    current = np.zeros(time.size)
    current[374] = 375.0
    plasma_current = xr.DataArray(current, dims=["time"], coords={"time": time})
    _, time, durations, tolerance = get_sample_durations(plasma_current)

    smoothed = time_weighted_running_mean(current, time, durations, tolerance)

    # The 375th sample completes 75 ms and must be included. The former
    # float-to-int truncation produced 374 samples and a zero first average.
    assert smoothed[0] == pytest.approx(1.0)


def test_mapping_tmax_is_invariant_across_native_cadences(mocker, sample_mapping):
    read_profile = mocker.patch("src.level2.main.DatasetReader.read_profile")
    bounds = []
    cadences = (0.0002, 0.00025, 0.0005, 0.001)
    traces = [make_plasma_current(cadence, 0.15) for cadence in cadences]
    traces.extend(
        (make_multirate_plasma_current(0.15), make_union_grid_plasma_current(0.15))
    )

    for trace in traces:
        mapping = sample_mapping.model_copy(deep=True)
        read_profile.return_value = trace
        set_mapping_time_bounds(mapping, 30420, mocker.Mock())
        bounds.append(mapping.tmax)

    assert all(check_plasma_current(trace) for trace in traces)
    # Both the pulse edge and smoothed threshold crossing are quantised to the
    # native grid, so two native samples bound the expected discretisation.
    native_resolution = 2 * max(cadences)
    assert max(bounds) - min(bounds) <= native_resolution
    assert bounds == pytest.approx([0.2205] * len(traces), abs=native_resolution)


def test_float32_uda_time_quantisation_is_usable():
    time = np.arange(-2.0, 4.0, 0.0002).astype(np.float32)
    values = np.where((time >= 0) & (time < 0.07), 250_000.0, 0.0)
    plasma_current = xr.DataArray(values, dims=["time"], coords={"time": time})

    assert check_plasma_current(plasma_current)


def test_level2_cli_has_no_global_interpolation_delta():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["mapping.yml", "--dt", "0.001"])


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([0.0, 0.001, 0.0005], "strictly increasing"),
        ([0.0, np.nan, 0.001], "finite values"),
    ],
)
def test_plasma_current_rejects_unusable_time_coordinates(time, message):
    plasma_current = xr.DataArray(
        np.full(len(time), 250_000.0),
        dims=["time"],
        coords={"time": time},
    )

    with pytest.raises(ValueError, match=message):
        check_plasma_current(plasma_current)


def test_plasma_current_requires_time_dimension():
    plasma_current = xr.DataArray(np.full(10, 250_000.0), dims=["sample"])

    with pytest.raises(ValueError, match="time dimension"):
        check_plasma_current(plasma_current)


def test_plasma_current_requires_time_coordinate():
    plasma_current = xr.DataArray(np.full(10, 250_000.0), dims=["time"])

    with pytest.raises(ValueError, match="time coordinate"):
        check_plasma_current(plasma_current)


def test_trim_requires_a_complete_smoothing_window():
    time = np.arange(0, 0.05, 0.001)
    plasma_current = xr.DataArray(
        np.full(time.size, 250_000.0),
        dims=["time"],
        coords={"time": time},
    )

    with pytest.raises(ValueError, match="enough time coverage"):
        trim_ip_range(plasma_current)


def test_plasma_current_rejects_gaps_larger_than_smoothing_window():
    time = [0.0, 0.001, 0.1, 0.101]
    plasma_current = xr.DataArray(
        np.full(len(time), 250_000.0),
        dims=["time"],
        coords={"time": time},
    )

    with pytest.raises(ValueError, match="time gaps"):
        check_plasma_current(plasma_current)
