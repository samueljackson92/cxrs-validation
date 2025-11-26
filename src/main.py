import argparse
import numpy as np
import panel as pn
from loguru import logger
import holoviews as hv
import hvplot.xarray  # noqa: F401
import xarray as xr

pn.extension(design="material", sizing_mode="stretch_width")


def plot_profile_slice(
    name: str,
    ds: xr.Dataset,
    time_point: float,
    radial_point: float,
    plot_options: dict,
) -> None:
    ds = ds.sel(time=time_point)
    ds = ds.rename({f"{name}_data": "data", f"{name}_error": "error"})

    ymin, ymax = ds.data.values.min(), ds.data.values.max()
    plot = ds.hvplot.line(
        x="major_radius",
        y="data",
        title=f"{name} at t={time_point:.2f}s",
        ylim=(ymin * 0.8, ymax * 1.2),
        ylabel=name,
    ).opts(**plot_options)

    error_bars = hv.ErrorBars(
        (ds["major_radius"].values, ds["data"].values, ds["error"].values),
    ).opts(line_alpha=0)

    radial_point_line = hv.VLine(radial_point).opts(
        color="red", line_width=2, line_dash="dashed"
    )

    plot = plot * error_bars * radial_point_line
    return plot


def plot_volume(
    name: str,
    ds: xr.Dataset,
    time_point: float,
    radial_point: float,
    wavelength_range: tuple[float, float],
    plot_options: dict,
    no_padding: bool = False,
    rename: bool = True,
) -> None:
    ds = ds.sel(time=time_point, major_radius=radial_point)
    ds = ds.sel(wavelength=slice(wavelength_range[0], wavelength_range[1]))
    if rename:
        ds = ds.rename({f"{name}_data": "data", f"{name}_error": "error"})

    ymin, ymax = np.nanmin(ds.data.values), np.nanmax(ds.data.values)
    ylim = (ymin * 0.7, ymax * 1.3)
    if no_padding:
        ylim = (ymin, ymax)

    plot = ds.hvplot.line(
        x="wavelength",
        y="data",
        title=f"{name} at t={time_point:.2f}s and r={radial_point:.2f}m",
        ylim=ylim,
        ylabel=name,
    ).opts(**plot_options)

    error_bars = hv.ErrorBars(
        (ds["wavelength"].values, ds["data"].values, ds["error"].values),
    ).opts(line_alpha=0)

    plot = plot * error_bars
    return plot


def make_line_plots(
    ss_fit_ratio,
    ss_emissivity,
    ss_velocity,
    ss_temperature,
    time_point,
    radial_point,
    plot_options,
):
    fit_ratio_plot = plot_profile_slice(
        "fit_ratio", ss_fit_ratio, time_point, radial_point, plot_options
    )
    emissivity_plot = plot_profile_slice(
        "emissivity", ss_emissivity, time_point, radial_point, plot_options
    )
    velocity_plot = plot_profile_slice(
        "velocity", ss_velocity, time_point, radial_point, plot_options
    )
    temperature_plot = plot_profile_slice(
        "temperature", ss_temperature, time_point, radial_point, plot_options
    )

    left_column = (
        (fit_ratio_plot + emissivity_plot + velocity_plot + temperature_plot)
        .cols(1)
        .opts(shared_axes=False)
    )

    return left_column


def make_volume_plots(
    ss_fits,
    ss_counts,
    ss_bg_counts,
    ss_sub_fits,
    ss_sub_counts,
    time_point,
    radial_point,
    wavelength_range,
    plot_options,
):
    ss_fits_plot = plot_volume(
        "ss_fits", ss_fits, time_point, radial_point, wavelength_range, plot_options
    )
    ss_counts_plot = plot_volume(
        "ss_counts", ss_counts, time_point, radial_point, wavelength_range, plot_options
    )
    ss_bg_counts_plot = plot_volume(
        "ss_bg_counts",
        ss_bg_counts,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
    )

    ss_sub_fits_plot = plot_volume(
        "ss_sub_fits",
        ss_sub_fits,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
    )
    ss_sub_counts_plot = plot_volume(
        "ss_sub_counts",
        ss_sub_counts,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
    )

    ss_sub_residual = ss_sub_counts.rename(
        {"ss_sub_counts_data": "data", "ss_sub_counts_error": "error"}
    ) - ss_sub_fits.rename({"ss_sub_fits_data": "data", "ss_sub_fits_error": "error"})
    ss_sub_residual_plot = plot_volume(
        "",
        ss_sub_residual,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
        no_padding=True,
        rename=False,
    )
    zero_line = hv.HLine(0).opts(color="black", line_width=2, line_dash="solid")
    ss_sub_residual_plot = ss_sub_residual_plot * zero_line

    ss_plot = ss_fits_plot * ss_counts_plot * ss_bg_counts_plot
    ss_sub_plot = (ss_sub_fits_plot * ss_sub_counts_plot).opts(padding=(0.1, 0.1))

    right_column = (
        (ss_plot + ss_sub_plot + ss_sub_residual_plot).opts(shared_axes=False).cols(1)
    )
    return right_column


def heatmap_plot(ss_temperature, time_point, radial_point):
    heatmap = ss_temperature.dropna(dim="time").hvplot(
        title="SS Temperature Profile", x="time", y="major_radius", z="data", height=150
    )
    heatmap = (
        heatmap
        * hv.VLine(time_point).opts(color="red", line_width=2, line_dash="dashed")
        * hv.HLine(radial_point).opts(color="red", line_width=2, line_dash="dashed")
    )
    return heatmap


def main():
    parser = argparse.ArgumentParser(description="CXRS Validation UI")
    parser.add_argument("path", type=str, help="Path to CXRS frame file data")
    parser.add_argument("--seed", type=int, default=50, help="Random seed for sampling")

    args = parser.parse_args()

    np.random.seed(args.seed)
    dataset = xr.open_zarr(args.path)
    dataset = dataset.dropna(dim="time", how="all")

    # Find regions of time where we keep wavelength profile data
    subsampled_frames = dataset.ss_fits_data.dropna(dim="time", how="all")
    subsampled_frames = subsampled_frames.stack(
        frame=("shot_id", "time", "major_radius")
    )
    frames = subsampled_frames.frame

    # Randomize and chose a frame to show
    frame_indices = np.arange(len(frames))
    np.random.shuffle(frame_indices)

    frame = frames.isel(frame=frame_indices[0])

    shot_id = frame.shot_id.item()
    time_point = frame.time.item()
    radial_point = frame.major_radius.item()

    # Plotting options
    height = 250
    wavelength_range = (528, 531)
    plot_options = dict(fontsize={"ylabel": 10, "xlabel": 10}, height=height)

    logger.info(f"Starting app for shot {shot_id} at time {time_point}s")

    # Load data for the selected frame

    # Load radial profile data for the shot
    radial_profiles = dataset.sel(shot_id=shot_id)
    ss_fit_ratio = radial_profiles[["fit_ratio_data", "fit_ratio_error"]]
    ss_emissivity = radial_profiles[["emissivity_data", "emissivity_error"]]
    ss_velocity = radial_profiles[["velocity_data", "velocity_error"]]
    ss_temperature = radial_profiles[["temperature_data", "temperature_error"]]

    # Load wavelength profile data for the shot
    wavelength_profiles = dataset.sel(shot_id=shot_id)
    ss_fits = wavelength_profiles[["ss_fits_data", "ss_fits_error"]]
    ss_counts = wavelength_profiles[["ss_counts_data", "ss_counts_error"]]
    ss_bg_counts = wavelength_profiles[["ss_bg_counts_data", "ss_bg_counts_error"]]

    ss_sub_fits = wavelength_profiles[["ss_sub_fits_data", "ss_sub_fits_error"]]
    ss_sub_counts = wavelength_profiles[["ss_sub_counts_data", "ss_sub_counts_error"]]

    title = pn.pane.Markdown(
        """
        ## Controls
        """
    )

    next_button = pn.widgets.Button(name="Next", button_type="primary")
    prev_button = pn.widgets.Button(name="Previous", button_type="primary")
    navigation = pn.Row(prev_button, next_button)

    emissivity_title = pn.pane.Markdown(
        """
        ### Emissivity Quality Label
        """
    )

    emissivity_label = pn.widgets.RadioBoxGroup(
        name="Emissivity Quality",
        options=["0 - Bad", "1 - Average", "2 - Good"],
        inline=False,
    )

    velocity_title = pn.pane.Markdown(
        """
        ### Velocity Quality Label
        """
    )

    velocity_label = pn.widgets.RadioBoxGroup(
        name="Velocity Quality",
        options=["0 - Bad", "1 - Average", "2 - Good"],
        inline=False,
    )

    temperature_title = pn.pane.Markdown(
        """
        ### Temperature Quality Label
        """
    )

    temperature_label = pn.widgets.RadioBoxGroup(
        name="Temperature Quality",
        options=["0 - Bad", "1 - Average", "2 - Good"],
        inline=False,
    )

    sidebar = pn.Column(
        title,
        navigation,
        emissivity_title,
        emissivity_label,
        velocity_title,
        velocity_label,
        temperature_title,
        temperature_label,
    )

    line_plot_options = plot_options.copy()
    line_plot_options.update(dict(height=210))

    volume_plot_options = plot_options.copy()
    volume_plot_options.update(dict(height=280))

    left_column = make_line_plots(
        ss_fit_ratio,
        ss_emissivity,
        ss_velocity,
        ss_temperature,
        time_point,
        radial_point,
        plot_options=line_plot_options,
    )

    right_column = make_volume_plots(
        ss_fits,
        ss_counts,
        ss_bg_counts,
        ss_sub_fits,
        ss_sub_counts,
        time_point,
        radial_point,
        wavelength_range,
        plot_options=volume_plot_options,
    )

    right_column = pn.Column(right_column)

    contents = pn.Row(left_column, right_column)

    app = pn.template.MaterialTemplate(
        title="CXRS Validation UI",
        header_background="#009900",
        sidebar=sidebar,
        main=[contents],
    )

    app.servable()


main()
