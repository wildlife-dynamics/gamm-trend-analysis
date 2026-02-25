"""Forest cover trend analysis tasks."""
from typing import Annotated
import geopandas as gpd
import pandas as pd
from ecoscope_workflows_core.decorators import task
from pydantic import Field
from ecoscope.analysis.trend_analysis import get_forest_cover_trends


@task
def extract_forest_cover_trends(
    aoi: Annotated[
        gpd.GeoDataFrame,
        Field(description="Area of interest geometry (must have CRS set)")
    ],
    tree_cover_threshold: Annotated[
        float,
        Field(default=60.0, description="Minimum tree cover percentage (0-100)")
    ] = 60.0,
    scale: Annotated[
        int,
        Field(default=30, description="Pixel scale in meters for reduction")
    ] = 30,
    max_pixels: Annotated[
        float,
        Field(default=1e9, description="Maximum pixels for reduction")
    ] = 1e9,
) -> pd.DataFrame:
    """
    Extract forest cover trends from Google Earth Engine dataset.
    
    Returns DataFrame with columns:
    - year: Year of observation
    - loss_area: Forest loss area in acres
    - cumsum_loss_area: Cumulative loss area in acres
    - survival_area: Remaining forest area in acres
    """
    return get_forest_cover_trends(
        aoi=aoi,
        tree_cover_threshold=tree_cover_threshold,
        scale=scale,
        max_pixels=max_pixels,
    )
