from pathlib import Path
import xarray as xr
import numpy as np
import argparse
from loguru import logger
from src.data import UDALoader


def subsample_dataset(dataset: xr.Dataset, method: str, num_samples: int) -> xr.Dataset:
    time = dataset.dropna(dim="time", how="all").time
    if method == "random":
        sampled_times = np.random.choice(time, size=num_samples, replace=False)
    elif method == "grid":
        sampled_times = time[:: max(1, len(time) // num_samples)]
    return dataset.sel(time=sampled_times)


def load_dataset(
    shot_id: int, sample_method: str = "grid", num_samples: int = 10
) -> xr.Dataset:
    loader = UDALoader()

    ## SS Profiles
    ss_fit_ratio = loader.get_radial_profile(
        name="fit_ratio",
        signal_name="ACT/CEL3/SS/PVB/FIT_RATIO",
        shot_id=shot_id,
    )
    ss_emissivity = loader.get_radial_profile(
        name="emissivity",
        signal_name="ACT/CEL3/SS/PVB/C5291/EMISSIVITY",
        shot_id=shot_id,
    )
    ss_velocity = loader.get_radial_profile(
        name="velocity",
        signal_name="ACT/CEL3/SS/PVB/C5291/VELOCITY",
        shot_id=shot_id,
    )
    ss_temperature = loader.get_radial_profile(
        name="temperature",
        signal_name="ACT/CEL3/SS/PVB/C5291/TEMPERATURE",
        shot_id=shot_id,
    )

    # Interpolate all profiles to common time base
    min_time = -0.1
    max_time = 2.0
    dt = 0.005

    time_base = np.arange(min_time, max_time + dt, dt)
    radial_profiles = [ss_fit_ratio, ss_emissivity, ss_velocity, ss_temperature]
    radial_profiles = [profile.interp(time=time_base) for profile in radial_profiles]
    radial_profiles = xr.merge(radial_profiles)

    ## SS Fits
    ss_fits = loader.get_volume_data(
        name="ss_fits",
        signal_name="ACT/CEL3/SS/PVB/SS_FITS",
        shot_id=shot_id,
    )
    ss_counts = loader.get_volume_data(
        name="ss_counts",
        signal_name="ACT/CEL3/SS/COUNTS",
        shot_id=shot_id,
    )
    ss_bg_counts = loader.get_volume_data(
        name="ss_bg_counts",
        signal_name="ACT/CEL3/SS/PVB/SCALED_BG_COUNTS",
        shot_id=shot_id,
    )

    ## Sub Fits
    ss_sub_fits = loader.get_volume_data(
        name="ss_sub_fits",
        signal_name="ACT/CEL3/SS/PVB/SUB_FITS",
        shot_id=shot_id,
    )
    ss_sub_counts = loader.get_volume_data(
        name="ss_sub_counts",
        signal_name="ACT/CEL3/SS/PVB/SUB_COUNTS",
        shot_id=shot_id,
    )

    # Interpolate all wavelength profiles to common time base
    wavelength_profiles = [ss_fits, ss_counts, ss_bg_counts, ss_sub_fits, ss_sub_counts]
    wavelength_profiles = [
        profile.interp(time=time_base) for profile in wavelength_profiles
    ]
    wavelength_profiles = xr.merge(wavelength_profiles)

    # Subsample both wavelength profiles in time
    wavelength_profiles = subsample_dataset(
        wavelength_profiles, method=sample_method, num_samples=num_samples
    )

    dataset = xr.merge(
        [
            radial_profiles,
            wavelength_profiles,
        ]
    )

    dataset = dataset.expand_dims({"shot_id": [shot_id]})
    return dataset


def write_dataset(dataset: xr.Dataset, shot_id: int, output_path: Path):
    output_file = output_path / "shots.zarr"
    if output_file.exists():
        dataset.to_zarr(output_file, mode="a", append_dim="shot_id")
    else:
        dataset.to_zarr(output_file)
    logger.info(f"Saved dataset for shot {shot_id} to {output_file}")


def process_shot(
    shot_id: int, output_path: Path, sample_method: str = "grid", num_samples: int = 10
):
    try:
        logger.info(f"Processing shot {shot_id}")
        dataset = load_dataset(shot_id, sample_method, num_samples)
        write_dataset(dataset, shot_id, output_path)
    except Exception as e:
        logger.error(f"Failed to process shot {shot_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a CXRS frame dataset from UDA."
    )
    parser.add_argument("shot_min", type=int, help="Minimum shot number")
    parser.add_argument("shot_max", type=int, help="Maximum shot number")
    parser.add_argument(
        "--output-path", type=str, default=".", help="Path to save the output dataset"
    )
    parser.add_argument(
        "--sample-method",
        type=str,
        choices=["random", "grid"],
        default="grid",
        help="Sampling method",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to select"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    np.random.seed(args.seed)
    shot_ids = range(args.shot_min, args.shot_max + 1)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for shot_id in shot_ids:
        process_shot(
            shot_id,
            output_path,
            sample_method=args.sample_method,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
