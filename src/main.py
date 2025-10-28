import numpy as np
import panel as pn
from loguru import logger
import holoviews as hv
import hvplot.xarray  # noqa: F401
import xarray as xr
from src.data import UDALoader

pn.extension(design="material", sizing_mode="stretch_width")


def plot_profile_slice(
    name: str,
    ds: xr.Dataset,
    time_point: float,
    radial_point: float,
    plot_options: dict,
) -> None:
    ds = ds.sel(time=time_point, method="nearest")

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
) -> None:
    ds = ds.sel(time=time_point, method="nearest").sel(
        major_radius=radial_point, method="nearest"
    )
    ds = ds.sel(wavelength=slice(wavelength_range[0], wavelength_range[1]))

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
        "Fit Ratio", ss_fit_ratio, time_point, radial_point, plot_options
    )
    emissivity_plot = plot_profile_slice(
        "Emissivity", ss_emissivity, time_point, radial_point, plot_options
    )
    velocity_plot = plot_profile_slice(
        "Velocity", ss_velocity, time_point, radial_point, plot_options
    )
    temperature_plot = plot_profile_slice(
        "Temperature", ss_temperature, time_point, radial_point, plot_options
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
        "SS Fits", ss_fits, time_point, radial_point, wavelength_range, plot_options
    )
    ss_counts_plot = plot_volume(
        "SS Counts", ss_counts, time_point, radial_point, wavelength_range, plot_options
    )
    ss_bg_counts_plot = plot_volume(
        "SS BG Counts",
        ss_bg_counts,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
    )

    ss_sub_fits_plot = plot_volume(
        "SS Sub Fits",
        ss_sub_fits,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
    )
    ss_sub_counts_plot = plot_volume(
        "SS Sub Counts",
        ss_sub_counts,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
    )

    ss_sub_residual = ss_sub_counts - ss_sub_fits
    ss_sub_residual_plot = plot_volume(
        "SS Sub Residual",
        ss_sub_residual,
        time_point,
        radial_point,
        wavelength_range,
        plot_options,
        no_padding=True,
    )
    zero_line = hv.HLine(0).opts(color="black", line_width=2, line_dash="solid")
    ss_sub_residual_plot = ss_sub_residual_plot * zero_line

    ss_plot = ss_fits_plot * ss_counts_plot * ss_bg_counts_plot
    ss_sub_plot = (ss_sub_counts_plot * ss_sub_fits_plot).opts(padding=(0.1, 0.1))

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
    shot_id = 46963
    time_point = 0.305
    radial_point = 0.9
    height = 250
    wavelength_range = (528, 531)
    plot_options = dict(fontsize={"ylabel": 10, "xlabel": 10}, height=height)

    logger.info(f"Starting app for shot {shot_id} at time {time_point}s")

    loader = UDALoader()

    ## SS Profiles
    ss_fit_ratio = pn.state.as_cached(
        "fit_ratio",
        loader.get_radial_profile,
        name="ACT/CEL3/SS/PVB/FIT_RATIO",
        shot_id=shot_id,
    )
    ss_emissivity = pn.state.as_cached(
        "emissivity",
        loader.get_radial_profile,
        name="ACT/CEL3/SS/PVB/C5291/EMISSIVITY",
        shot_id=shot_id,
    )
    ss_velocity = pn.state.as_cached(
        "velocity",
        loader.get_radial_profile,
        name="ACT/CEL3/SS/PVB/C5291/VELOCITY",
        shot_id=shot_id,
    )
    ss_temperature = pn.state.as_cached(
        "temperature",
        loader.get_radial_profile,
        name="ACT/CEL3/SS/PVB/C5291/TEMPERATURE",
        shot_id=shot_id,
    )

    ## SS Fits
    ss_fits = pn.state.as_cached(
        "fits", loader.get_volume_data, name="ACT/CEL3/SS/PVB/SS_FITS", shot_id=shot_id
    )
    ss_counts = pn.state.as_cached(
        "counts", loader.get_volume_data, name="ACT/CEL3/SS/COUNTS", shot_id=shot_id
    )
    ss_bg_counts = pn.state.as_cached(
        "bg_counts",
        loader.get_volume_data,
        name="ACT/CEL3/SS/PVB/SCALED_BG_COUNTS",
        shot_id=shot_id,
    )

    ## Sub Fits
    ss_sub_fits = pn.state.as_cached(
        "sub_fits",
        loader.get_volume_data,
        name="ACT/CEL3/SS/PVB/SUB_FITS",
        shot_id=shot_id,
    )
    ss_sub_counts = pn.state.as_cached(
        "sub_counts",
        loader.get_volume_data,
        name="ACT/CEL3/SS/PVB/SUB_COUNTS",
        shot_id=shot_id,
    )

    ss_temperature = ss_temperature.dropna(dim="time")
    ss_emissivity = ss_emissivity.dropna(dim="time")
    ss_velocity = ss_velocity.dropna(dim="time")

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

    left_column = pn.bind(
        make_line_plots,
        ss_fit_ratio,
        ss_emissivity,
        ss_velocity,
        ss_temperature,
        time_point,
        radial_point,
        plot_options=line_plot_options,
    )

    right_column = pn.bind(
        make_volume_plots,
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

    # heatmap = pn.bind(
    #     heatmap_plot, ss_temperature, time_point_slider, radial_point_slider
    # )
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
