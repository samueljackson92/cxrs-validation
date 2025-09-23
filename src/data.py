import xarray as xr
from loguru import logger
import os
os.environ["UDA_HOST"] = os.environ.get("UDA_HOST", "uda2.mast.l")
os.environ["UDA_META_PLUGINNAME"] = os.environ.get("UDA_META_PLUGINNAME", "MASTU_DB")
os.environ["UDA_METANEW_PLUGINNAME"] = os.environ.get("UDA_METANEW_PLUGINNAME", "MAST_DB")

from pyuda import Client


class UDALoader:
    def __init__(self):
        self.client = Client()

    def get_radial_profile(self, name: str, shot_id: int) -> xr.DataArray:
        logger.info(f"Loading {name} for shot {shot_id}")
        signal = self.client.get(name, shot_id)
        data = xr.DataArray(
            signal.data,
            dims=("time", "major_radius"),
            coords={"time": signal.dims[0].data, "major_radius": signal.dims[1].data},
            name=name,
        )
        error = xr.DataArray(
            signal.errors, dims=("time", "major_radius"), coords=data.coords
        )
        ds = xr.Dataset({"data": data, "error": error})
        return ds

    def get_volume_data(self, name: str, shot_id: int) -> xr.DataArray:
        logger.info(f"Loading {name} for shot {shot_id}")
        signal = self.client.get(name, shot_id)
        data = xr.DataArray(
            signal.data,
            dims=("time", "major_radius", "wavelength"),
            coords={"time": signal.dims[0].data, "major_radius": signal.dims[1].data, "wavelength": signal.dims[2].data},
            name=name,
        )
        error = xr.DataArray(signal.errors, dims=("time", "major_radius", "wavelength"), coords=data.coords)

        ds = xr.Dataset({"data": data, "error": error})
        return ds
