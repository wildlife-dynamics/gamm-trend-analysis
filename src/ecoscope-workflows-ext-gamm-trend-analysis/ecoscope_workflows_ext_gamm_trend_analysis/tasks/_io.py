"""I/O tasks for GAMM trend analysis."""

from typing import Annotated, Optional
import geopandas as gpd
from ecoscope_workflows_core.decorators import task
from pydantic import Field
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def read_geopackage(
    path: Annotated[str, Field(description="Path to the GeoPackage file")],
    layer: Annotated[
        Optional[str],
        Field(default=None, description="Layer name to read from the GeoPackage"),
    ] = None,
) -> AnyDataFrame:
    """
    Read a GeoPackage file into a GeoDataFrame.
    """
    roi = gpd.read_file(path, layer=layer)
    loita_forest = roi[roi["name"] == "Loita Forest"]
    return loita_forest
