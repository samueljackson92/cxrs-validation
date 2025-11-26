import argparse
import numpy as np
import pandas as pd
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
    ds = ds.sel(time=time_point).copy()
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
    plot_options: dict,
    no_padding: bool = False,
    rename: bool = True,
) -> None:
    ds = ds.sel(time=time_point, major_radius=radial_point)
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
    plot_options,
):
    ss_fits_plot = plot_volume(
        "ss_fits", ss_fits, time_point, radial_point, plot_options
    )
    ss_counts_plot = plot_volume(
        "ss_counts", ss_counts, time_point, radial_point, plot_options
    )
    ss_bg_counts_plot = plot_volume(
        "ss_bg_counts",
        ss_bg_counts,
        time_point,
        radial_point,
        plot_options,
    )

    ss_sub_fits_plot = plot_volume(
        "ss_sub_fits",
        ss_sub_fits,
        time_point,
        radial_point,
        plot_options,
    )
    ss_sub_counts_plot = plot_volume(
        "ss_sub_counts",
        ss_sub_counts,
        time_point,
        radial_point,
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


class CXRSValidationApp:
    def __init__(self, path):
        self.dataset = xr.open_zarr(path)
        valid_times = self.dataset.temperature_data.dropna(dim="time", how="all").time
        self.dataset = self.dataset.sel(time=valid_times)

        # Find regions of time where we kept wavelength profile data
        subsampled_frames = self.dataset.stack(
            frame=("shot_id", "time", "major_radius")
        )
        self.frames = subsampled_frames.frame

        # Randomize and choose a frame to show
        self.frame_indices = np.arange(len(self.frames))
        np.random.shuffle(self.frame_indices)

        self.current_index = 0
        self._get_frame(self.current_index)
        self.ratings = []

    def _get_frame(self, index: int):
        self.frame = self.frames.isel(frame=self.frame_indices[index])
        self.shot_id = self.frame.shot_id.item()
        self.time_point = self.frame.time.item()
        self.radial_point = self.frame.major_radius.item()

    def plot(self):
        # Plotting options

        logger.info(f"Starting app for shot {self.shot_id} at time {self.time_point}s")

        title = pn.pane.Markdown(
            """
            ## Controls
            """
        )

        def handle_next_click(event):
            self.save_state()
            self.current_index += 1
            self._get_frame(self.current_index)
            self.contents[:] = self.replot_data()

        def handle_prev_click(event):
            self.current_index -= 1
            self._get_frame(self.current_index)
            self.contents[:] = self.replot_data()

        next_button = pn.widgets.Button(name="Next", button_type="primary")
        next_button.on_click(handle_next_click)

        prev_button = pn.widgets.Button(name="Previous", button_type="primary")
        prev_button.on_click(handle_prev_click)

        navigation = pn.Row(prev_button, next_button)

        emissivity_title = pn.pane.Markdown(
            """
            ### Emissivity Quality Label
            """
        )

        self.emissivity_label = pn.widgets.RadioBoxGroup(
            name="Emissivity Quality",
            options=["0 - Bad", "1 - Average", "2 - Good"],
            inline=False,
        )

        velocity_title = pn.pane.Markdown(
            """
            ### Velocity Quality Label
            """
        )

        self.velocity_label = pn.widgets.RadioBoxGroup(
            name="Velocity Quality",
            options=["0 - Bad", "1 - Average", "2 - Good"],
            inline=False,
        )

        temperature_title = pn.pane.Markdown(
            """
            ### Temperature Quality Label
            """
        )

        self.temperature_label = pn.widgets.RadioBoxGroup(
            name="Temperature Quality",
            options=["0 - Bad", "1 - Average", "2 - Good"],
            inline=False,
        )

        sidebar = pn.Column(
            title,
            navigation,
            emissivity_title,
            self.emissivity_label,
            velocity_title,
            self.velocity_label,
            temperature_title,
            self.temperature_label,
        )

        left, right = self.replot_data()
        self.contents = pn.Row(left, right)

        self.app = pn.template.MaterialTemplate(
            title="CXRS Validation UI",
            header_background="#009900",
            sidebar=sidebar,
            main=[self.contents],
        )

        self.app.servable()

    def replot_data(self):
        plot_options = dict(fontsize={"ylabel": 10, "xlabel": 10}, height=250)

        line_plot_options = plot_options.copy()
        line_plot_options.update(dict(height=210))

        volume_plot_options = plot_options.copy()
        volume_plot_options.update(dict(height=280))

        # Load data for the selected frame

        # Load radial profile data for the shot
        radial_profiles = self.dataset.sel(shot_id=self.shot_id)
        ss_fit_ratio = radial_profiles[["fit_ratio_data", "fit_ratio_error"]]
        ss_emissivity = radial_profiles[["emissivity_data", "emissivity_error"]]
        ss_velocity = radial_profiles[["velocity_data", "velocity_error"]]
        ss_temperature = radial_profiles[["temperature_data", "temperature_error"]]

        # Load wavelength profile data for the shot
        wavelength_profiles = self.dataset.sel(shot_id=self.shot_id)
        ss_fits = wavelength_profiles[["ss_fits_data", "ss_fits_error"]]
        ss_counts = wavelength_profiles[["ss_counts_data", "ss_counts_error"]]
        ss_bg_counts = wavelength_profiles[["ss_bg_counts_data", "ss_bg_counts_error"]]

        ss_sub_fits = wavelength_profiles[["ss_sub_fits_data", "ss_sub_fits_error"]]
        ss_sub_counts = wavelength_profiles[
            ["ss_sub_counts_data", "ss_sub_counts_error"]
        ]

        self.left_column = make_line_plots(
            ss_fit_ratio,
            ss_emissivity,
            ss_velocity,
            ss_temperature,
            self.time_point,
            self.radial_point,
            plot_options=line_plot_options,
        )

        self.right_column = make_volume_plots(
            ss_fits,
            ss_counts,
            ss_bg_counts,
            ss_sub_fits,
            ss_sub_counts,
            self.time_point,
            self.radial_point,
            plot_options=volume_plot_options,
        )
        return self.left_column, self.right_column

    def save_state(self):
        logger.info(
            f"Current frame index: {self.current_index}, shot_id: {self.shot_id}, time: {self.time_point}, radial_point: {self.radial_point}"
        )

        info = {
            "current_index": self.current_index,
            "shot_id": self.shot_id,
            "time_point": self.time_point,
            "radial_point": self.radial_point,
            "emissivity_label": self.emissivity_label.value,
            "velocity_label": self.velocity_label.value,
            "temperature_label": self.temperature_label.value,
        }
        self.ratings.append(info)

        logger.info(f"Current labels: {info}")

        pd.DataFrame(self.ratings).to_csv("cxrs_validation_ratings.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="CXRS Validation UI")
    parser.add_argument("path", type=str, help="Path to CXRS frame file data")
    parser.add_argument("--seed", type=int, default=50, help="Random seed for sampling")

    args = parser.parse_args()

    np.random.seed(args.seed)

    app = CXRSValidationApp(args.path)
    app.plot()


main()
