import argparse
import math
import sys
import uuid
from pathlib import Path

import numpy as np
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

from src.core.config import ReaderConfig, load_config
from src.core.load import (
    BaseLoader,
    loader_registry,
)
from src.core.log import logger
from src.core.model import Mapping, load_model
from src.core.upload import UploadS3
from src.core.workflow_manager import WorkflowManager
from src.core.writer import dataset_writer_registry
from src.level2.reader import DatasetReader

MINIMUM_PLASMA_CURRENT = 200_000
# Preserve the historical default policy as physical durations, rather than
# scaling raw-signal sample counts from an unrelated interpolation cadence.
MINIMUM_PLASMA_CURRENT_DURATION = 0.0625  # 250 samples at the legacy 0.25 ms
PLASMA_CURRENT_SMOOTHING_DURATION = 0.075  # 300 samples at the legacy 0.25 ms
# Integrating the full duration deliberately fixes the former float-to-int
# truncation that could turn 0.075 / 0.0002 into a 374-sample window.


def get_sample_durations(
    plasma_current: xr.DataArray,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray, float]:
    """Return finite samples, timestamps, ZOH widths, and time precision."""
    if plasma_current.ndim != 1 or plasma_current.dims != ("time",):
        raise ValueError("Plasma current must be one-dimensional with a time dimension")

    if "time" not in plasma_current.coords:
        raise ValueError("Plasma current must have a time coordinate")

    source_time = np.asarray(plasma_current.time.values)
    try:
        time = np.asarray(source_time, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Plasma current time coordinate must be numeric") from error

    if time.size < 2:
        raise ValueError(
            "Plasma current time coordinate must contain at least two samples"
        )
    if not np.isfinite(time).all():
        raise ValueError(
            "Plasma current time coordinate must contain only finite values"
        )

    sample_intervals = np.diff(time)
    if np.any(sample_intervals <= 0):
        raise ValueError("Plasma current time coordinate must be strictly increasing")

    try:
        current = np.asarray(plasma_current.values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Plasma current samples must be numeric") from error

    finite_samples = np.isfinite(current)
    if np.count_nonzero(finite_samples) < 2:
        raise ValueError("Plasma current must contain at least two finite samples")

    plasma_current = plasma_current.isel(time=np.flatnonzero(finite_samples))
    time = time[finite_samples]
    sample_intervals = np.diff(time)
    time_tolerance = get_time_quantization_tolerance(
        source_time.dtype, time, sample_intervals
    )
    if np.any(sample_intervals > PLASMA_CURRENT_SMOOTHING_DURATION + time_tolerance):
        raise ValueError(
            "Plasma current time gaps must not exceed the 0.075 s smoothing window"
        )

    # Treat each sample as zero-order held until the next timestamp. The final
    # sample has no right-hand timestamp, so use the last observed interval.
    sample_durations = np.append(sample_intervals, sample_intervals[-1])
    return plasma_current, time, sample_durations, time_tolerance


def get_time_quantization_tolerance(
    time_dtype: np.dtype, time: np.ndarray, sample_intervals: np.ndarray
) -> float:
    """Bound duration comparisons by timestamp precision, not a full sample."""
    scale = max(1.0, float(np.max(np.abs(time))))
    if np.issubdtype(time_dtype, np.floating):
        source_precision = 4 * np.finfo(time_dtype).eps * scale
    else:
        source_precision = 0.0

    source_precision = max(source_precision, 4 * math.ulp(scale))
    # Keep the equality band far below a native interval, so one additional
    # above-threshold sample always changes the gate result.
    interval_cap = float(np.min(sample_intervals)) * 0.01
    return min(source_precision, interval_cap)


def _clip_polygon_to_half_plane(
    polygon: list[tuple[float, float]],
    normal_x: float,
    normal_y: float,
    limit: float,
) -> list[tuple[float, float]]:
    """Clip a convex polygon to normal_x * x + normal_y * y <= limit."""
    if not polygon:
        return []

    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(limit))
    clipped = []
    previous = polygon[-1]
    previous_value = normal_x * previous[0] + normal_y * previous[1] - limit
    previous_inside = previous_value <= tolerance

    for current in polygon:
        current_value = normal_x * current[0] + normal_y * current[1] - limit
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            fraction = previous_value / (previous_value - current_value)
            clipped.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_inside:
            clipped.append(current)

        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return clipped


def _get_exact_affine_residual(time: np.ndarray) -> np.ndarray:
    """Return exact stored-time residuals from the line through the endpoints."""
    ratios = [float(value).as_integer_ratio() for value in time]
    common_denominator = max(denominator for _, denominator in ratios)
    ticks = [
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    ]
    interval_count = len(ticks) - 1
    first_tick = ticks[0]
    span = ticks[-1] - first_tick
    residual_denominator = interval_count * common_denominator
    return np.fromiter(
        (
            (interval_count * (tick - first_tick) - index * span) / residual_denominator
            for index, tick in enumerate(ticks)
        ),
        dtype=float,
        count=len(ticks),
    )


def is_grid_consistent_with_uniform_sampling(
    time_dtype: np.dtype, time: np.ndarray
) -> bool:
    """Return whether one affine timeline intersects every timestamp cell."""
    if not np.issubdtype(time_dtype, np.floating):
        sample_intervals = np.diff(time)
        return bool(np.all(sample_intervals == sample_intervals[0]))

    # A stored floating-point timestamp represents the rounding cell between
    # the midpoints to its adjacent values. Intersect the corresponding linear
    # constraints on one shared pair of latent endpoints. A non-empty polygon
    # is therefore an actual feasibility result, not an interval-spread guess.
    source_time = np.asarray(time, dtype=time_dtype)
    previous = np.nextafter(source_time, -np.inf)
    following = np.nextafter(source_time, np.inf)
    lower_radii = (source_time - previous).astype(float) / 2
    upper_radii = (following - source_time).astype(float) / 2

    weights = np.arange(time.size, dtype=float) / (time.size - 1)
    # Python exposes every finite IEEE float as an exact integer ratio. Use a
    # common dyadic scale so a large absolute clock offset cannot introduce a
    # numerical tolerance that hides an observable cadence change.
    residual = _get_exact_affine_residual(time)
    constraint_scale = max(
        float(np.max(lower_radii)),
        float(np.max(upper_radii)),
    )
    lower_limits = (residual - lower_radii) / constraint_scale
    upper_limits = (residual + upper_radii) / constraint_scale

    endpoint_0_lower = -lower_radii[0] / constraint_scale
    endpoint_0_upper = upper_radii[0] / constraint_scale
    endpoint_n_lower = -lower_radii[-1] / constraint_scale
    endpoint_n_upper = upper_radii[-1] / constraint_scale
    feasible_polygon = [
        (endpoint_0_lower, endpoint_n_lower),
        (endpoint_0_upper, endpoint_n_lower),
        (endpoint_0_upper, endpoint_n_upper),
        (endpoint_0_lower, endpoint_n_upper),
    ]

    for index in range(1, time.size - 1):
        endpoint_n_weight = weights[index]
        endpoint_0_weight = 1 - endpoint_n_weight
        feasible_polygon = _clip_polygon_to_half_plane(
            feasible_polygon,
            endpoint_0_weight,
            endpoint_n_weight,
            upper_limits[index],
        )
        feasible_polygon = _clip_polygon_to_half_plane(
            feasible_polygon,
            -endpoint_0_weight,
            -endpoint_n_weight,
            -lower_limits[index],
        )
        if not feasible_polygon:
            return False

    return True


def time_weighted_running_mean(
    current: np.ndarray,
    time: np.ndarray,
    sample_durations: np.ndarray,
    time_tolerance: float,
) -> np.ndarray:
    """Average a zero-order-held signal over exact forward 75 ms windows."""
    coverage_end = time[-1] + sample_durations[-1]
    coverage_duration = math.fsum(sample_durations)
    rounding_tolerance = max(
        math.ulp(PLASMA_CURRENT_SMOOTHING_DURATION) * 8, time_tolerance
    )
    if coverage_duration < PLASMA_CURRENT_SMOOTHING_DURATION - rounding_tolerance:
        raise ValueError(
            "Plasma current does not contain enough time coverage for the 0.075 s "
            "smoothing window"
        )

    valid_starts = (
        time + PLASMA_CURRENT_SMOOTHING_DURATION <= coverage_end + rounding_tolerance
    )
    window_starts = time[valid_starts]
    window_ends = np.minimum(
        window_starts + PLASMA_CURRENT_SMOOTHING_DURATION, coverage_end
    )

    edges = np.append(time, coverage_end)
    cumulative_area = np.concatenate(
        ([0.0], np.cumsum(current * sample_durations, dtype=float))
    )
    area_at_window_end = np.interp(window_ends, edges, cumulative_area)
    area_at_window_start = cumulative_area[: window_starts.size]
    return (
        area_at_window_end - area_at_window_start
    ) / PLASMA_CURRENT_SMOOTHING_DURATION


def trim_ip_range(plasma_current: xr.DataArray) -> xr.DataArray:
    plasma_current, time, sample_durations, time_tolerance = get_sample_durations(
        plasma_current
    )
    current = plasma_current.values / 1000
    rm = time_weighted_running_mean(current, time, sample_durations, time_tolerance)

    sign_samples = rm[np.abs(rm) > 30]
    if sign_samples.size == 0 or not np.isfinite(sign_samples).all():
        raise ValueError("Cannot determine plasma current sign from smoothed signal")
    sign_ip = np.sign(sign_samples.mean())
    if sign_ip == 0:
        raise ValueError("Cannot determine plasma current sign from smoothed signal")
    rm *= sign_ip

    plasma_indices = np.flatnonzero(np.abs(rm) >= 15)
    if plasma_indices.size == 0:
        raise ValueError("No plasma current remains after smoothing")

    start = plasma_indices[0]
    window_end = time[plasma_indices[-1]] + PLASMA_CURRENT_SMOOTHING_DURATION
    stop = np.searchsorted(time, window_end, side="left")
    return plasma_current[start:stop]


def check_plasma_current(plasma_current: xr.DataArray) -> bool:
    plasma_current, time, sample_durations, time_tolerance = get_sample_durations(
        plasma_current
    )
    # Qualification only counts intervals bounded by two observed timestamps.
    # The repeated final interval is needed by smoothing, but must not invent
    # evidence for the terminal sample.
    observed_durations = sample_durations[:-1]
    above_threshold = np.abs(plasma_current.values[:-1]) > MINIMUM_PLASMA_CURRENT
    if is_grid_consistent_with_uniform_sampling(plasma_current.time.dtype, time):
        # If all rounding cells admit a uniform latent grid, count samples at
        # the mean cadence. This preserves the historical policy and is stable
        # under timestamp quantisation; sub-resolution rate changes are
        # observationally indistinguishable and deliberately follow this path.
        mean_cadence = math.fsum(observed_durations) / observed_durations.size
        above_threshold_duration = np.count_nonzero(above_threshold) * mean_cadence
    else:
        # Observably multi-rate traces require their physical interval widths.
        above_threshold_duration = math.fsum(
            observed_durations[above_threshold].tolist()
        )
    rounding_tolerance = max(
        math.ulp(MINIMUM_PLASMA_CURRENT_DURATION) * 8,
        time_tolerance,
    )
    return (
        above_threshold_duration - MINIMUM_PLASMA_CURRENT_DURATION > rounding_tolerance
    )


def load_mapping_file(mapping_file: str):
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        logger.error(f'No mapping file exists called "{mapping_file}"')
        sys.exit(-1)

    mapping = load_model(mapping_file)
    return mapping


def load_config_file(config_file: str):
    mapping_file = Path(config_file)
    if not mapping_file.exists():
        logger.error(f'No mapping file exists called "{mapping_file}"')
        sys.exit(-1)

    mapping = load_config(mapping_file)
    return mapping


def create_uuid(oid_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, oid_name))


def get_default_loader(config: ReaderConfig) -> BaseLoader:
    loader_type = config.type
    loader_params = config.options
    loader = loader_registry.create(loader_type, **loader_params)
    return loader


def set_mapping_time_bounds(
    mapping: Mapping, shot: int, loader: BaseLoader, force: bool = False
):
    if mapping.plasma_current is None:
        return

    dataset_name, signal_name = mapping.plasma_current.split("/")
    reader = DatasetReader(mapping, loader)
    plasma_current = reader.read_profile(shot, dataset_name, signal_name)

    if plasma_current is None:
        raise RuntimeError("Cannot load Plasma Current")

    if not check_plasma_current(plasma_current):
        if force:
            logger.warning(
                f"Ip check failed for shot {shot}, continuing anyway due to --force-ip-check"
            )
        else:
            raise RuntimeError(f"No Plasma Current for shot {shot}")

    plasma_current = trim_ip_range(plasma_current)
    mapping.tmax = float(plasma_current.time.values.max())


def process_shot(shot: int, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.verbose:
        logger.setLevel("DEBUG")

    dataset_names = args.include_datasets

    mapping_file = Path(args.mapping_file)
    if mapping_file.is_dir():
        mapping_file = mapping_file / f"{shot}.yml"

    mapping = load_mapping_file(mapping_file)

    config = load_config_file(args.config_file)
    if args.output_path is not None:
        config.writer.options["output_path"] = args.output_path

    writer = dataset_writer_registry.create(config.writer.type, **config.writer.options)

    file_name = f"{shot}.{writer.file_extension}"
    local_file = config.writer.options["output_path"] / Path(file_name)

    loader = get_default_loader(config.readers[mapping.default_loader])
    set_mapping_time_bounds(mapping, shot, loader, force=args.force_ip_check)

    for group_name in mapping.datasets.keys():
        if len(dataset_names) == 0 or group_name in dataset_names:
            if group_name not in args.exclude_datasets:
                logger.info(
                    f"Processing {group_name} for shot {shot} from {mapping.facility}"
                )

                reader = DatasetReader(mapping, loader, skip_geometry=args.skip_geometry)
                dataset = reader.read_dataset(shot, group_name)
                if len(dataset) == 0:
                    continue

                logger.info(
                    f"Writing {group_name} for shot {shot} from {mapping.facility}"
                )
                writer.write(file_name, group_name, dataset)

    if config.upload is not None:
        remote_file = f"{config.upload.base_path}/"

        uploader = UploadS3(config.upload)
        uploader.upload(local_file, remote_file)

    logger.info(f"Done shot {shot}!")


def safe_process_shot(shot, *args, **kwargs):
    try:
        process_shot(shot, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to process shot {shot}: {e}")
        logger.debug("Exception information", exc_info=True)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")

    if args.shots is not None:
        shots = [int(s) for s in args.shots]
    elif args.shot is None:
        if args.shot_min is None or args.shot_max is None:
            logger.error(
                "Must provide both a minimum and maximum shot (--shot-min/--shot-max)"
            )
            sys.exit(-1)
        shots = range(args.shot_min, args.shot_max)
    else:
        shots = [args.shot]

    kwargs = vars(args)
    kwargs.pop("shot")
    workflow_manager = WorkflowManager(safe_process_shot)
    workflow_manager.run_workflows(shots, **kwargs)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_file", type=str)
    parser.add_argument("-c", "--config-file", type=str, default="./configs/level2.yml")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--shot-min", type=int, default=None)
    parser.add_argument("--shot-max", type=int, default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("-i", "--include-datasets", nargs="+", default=[])
    parser.add_argument("-e", "--exclude-datasets", nargs="+", default=[])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--force-ip-check", action="store_true")
    parser.add_argument("-o", "--output-path", type=str, default=None)
    parser.add_argument("-n", "--n-workers", type=int, default=None)
    parser.add_argument("--skip-geometry", action="store_true")
    return parser


if __name__ == "__main__":
    main()
