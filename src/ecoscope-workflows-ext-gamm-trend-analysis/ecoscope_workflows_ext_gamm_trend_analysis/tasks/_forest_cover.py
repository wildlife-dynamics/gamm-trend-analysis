"""Forest cover trend analysis tasks."""

from typing import Annotated, Optional
from ecoscope_workflows_ext_ecoscope.connections import EarthEngineClient
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from pydantic import Field
from ecoscope_workflows_core.annotations import AnyDataFrame, AdvancedField, AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.results._map import (
    BitmapLayerDefinition,
    LegendSegment,
    LegendValue,
)


@task
def extract_forest_cover_trends(
    client: EarthEngineClient,
    aoi: Annotated[AnyDataFrame, Field(description="Area of interest geometry (must have CRS set)")],
    time_range: Annotated[
        Optional[TimeRange], Field(description="Time range for the trend analysis")
    ] = None,
    tree_cover_threshold: Annotated[
        float, Field(default=60.0, description="Minimum tree cover percentage (0-100)")
    ] = 60.0,
    scale: Annotated[
        int,
        AdvancedField(
            default=30,
            description="Pixel scale in meters for reduction",
        ),
    ] = 30,
    max_pixels: Annotated[
        float,
        AdvancedField(
            default=1e9,
            description="Maximum pixels for reduction",
        ),
    ] = 1e9,
    image: Annotated[
        str,
        Field(
            default="UMD/hansen/global_forest_change_2023_v1_11",
            description="Google Earth Engine image name. Note that the Hansen dataset baseline starting point is always the year 2000.",
        ),
    ] = "UMD/hansen/global_forest_change_2023_v1_11",
) -> AnyDataFrame:
    """
    Extract forest cover trends from Google Earth Engine dataset.

    Returns DataFrame with columns:
    - year: Year of observation
    - loss_area: Forest loss area in acres
    - cumsum_loss_area: Cumulative loss area in acres
    - survival_area: Remaining forest area in acres
    """
    import ee
    import logging
    import pandas as pd

    if aoi.crs is None:
        logging.warning("AOI CRS not set. Assuming WGS84.")
        aoi = aoi.set_crs(4326)
    elif aoi.crs.to_epsg() != 4326:
        aoi = aoi.to_crs(4326)

    feat_coll = ee.FeatureCollection(aoi.__geo_interface__)
    gfc = ee.Image(image)

    # Calculate forested area in 2000
    treecover2000 = gfc.select(["treecover2000"])
    treecover2000_mask = treecover2000.gte(tree_cover_threshold)
    treecover2000 = treecover2000.unmask().updateMask(treecover2000_mask)
    treecover2000 = treecover2000.And(treecover2000)  # Convert pixel values to 1's
    treecover2000_area_img = treecover2000.multiply(ee.Image.pixelArea())
    treecover2000_area = treecover2000_area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=feat_coll,
        scale=scale,
        maxPixels=max_pixels,
    )

    forested_area = treecover2000_area.getInfo()["treecover2000"]
    forested_area = (forested_area or 0.0) * 0.000247105  # Convert sq.meters to acres

    # Calculate forest loss by year
    loss_img = gfc.select(["loss"])
    loss_img = loss_img.updateMask(treecover2000_mask)
    loss_area_img = loss_img.multiply(ee.Image.pixelArea())
    loss_year = gfc.select(["lossyear"])

    # Determine year filters from TimeRange
    start_year = pd.to_datetime(time_range.since).year if time_range else None
    end_year = pd.to_datetime(time_range.until).year if time_range else None

    # Apply server-side time filter via lossyear mask if range is provided
    if end_year:
        end_year_short = max(0, end_year - 2000)
        # Note: we always calculate loss from 2000 for correct survival_area,
        # but we can mask out pixels beyond the end_year to optimize the reduction.
        loss_area_img = loss_area_img.updateMask(loss_year.lte(end_year_short))

    loss_by_year = loss_area_img.addBands(loss_year).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1),
        geometry=feat_coll,
        scale=scale,
        maxPixels=max_pixels,
    )

    groups = loss_by_year.getInfo()["groups"]
    if not groups:
        return pd.DataFrame(columns=["year", "loss_area", "cumsum_loss_area", "survival_area"])

    forest_survival = pd.DataFrame([x for x in groups])
    forest_survival.rename(columns={"group": "year", "sum": "loss_area"}, inplace=True)
    forest_survival["year"] = forest_survival["year"] + 2000
    forest_survival["loss_area"] = forest_survival["loss_area"] * 0.000247105  # Convert sq.meters to acres

    # Ensure data is sorted by year for cumulative calculations
    forest_survival = forest_survival.sort_values("year")
    forest_survival["cumsum_loss_area"] = forest_survival["loss_area"].cumsum()
    forest_survival["survival_area"] = forested_area - forest_survival["cumsum_loss_area"]

    # Apply filters for final return
    if start_year:
        forest_survival = forest_survival[forest_survival["year"] >= start_year]
    if end_year:
        forest_survival = forest_survival[forest_survival["year"] <= end_year]

    return forest_survival


@task
def create_forest_layers(
    client: EarthEngineClient,
    aoi: Annotated[AnyGeoDataFrame, Field(description="Area of interest geometry")],
    time_range: Annotated[
        Optional[TimeRange], Field(description="Time range for the trend analysis")
    ] = None,
    tree_cover_threshold: Annotated[float, Field(default=60.0)] = 60.0,
    image: Annotated[str, Field(default="UMD/hansen/global_forest_change_2023_v1_11")] = (
        "UMD/hansen/global_forest_change_2023_v1_11"
    ),
    opacity: Annotated[float, Field(default=1.0)] = 1.0,
) -> Annotated[BitmapLayerDefinition, Field()]:
    """Creates a forest cover and loss tile layer for the map."""
    import base64
    import ee
    import requests
    import pandas as pd

    roi_gdf = aoi.to_crs("EPSG:4326")
    roi_geometry = roi_gdf.dissolve().geometry.iloc[0]
    ee_geometry = ee.Geometry(roi_geometry.__geo_interface__)

    gfc = ee.Image(image)
    forest_cover_img = gfc.select(["treecover2000"])
    forest_cover_mask = forest_cover_img.gte(tree_cover_threshold)
    forest_cover_img = forest_cover_img.unmask().updateMask(forest_cover_mask)
    forest_cover_img = forest_cover_img.And(forest_cover_img)

    loss_year_img = gfc.select("lossyear")
    start_year = pd.to_datetime(time_range.since).year if time_range else None
    end_year = pd.to_datetime(time_range.until).year if time_range else None

    # Filter loss pixels by the year range
    mask = loss_year_img.gt(0)
    if start_year:
        mask = mask.And(loss_year_img.gte(start_year - 2000))
    if end_year:
        mask = mask.And(loss_year_img.lte(end_year - 2000))

    forest_loss_img = loss_year_img.updateMask(mask).updateMask(forest_cover_mask)

    # Blend cover and loss into a single image for the BitmapLayer
    cover_vis = forest_cover_img.clip(ee_geometry).visualize(palette=["#236F21"])
    loss_vis = forest_loss_img.clip(ee_geometry).visualize(palette=["#FF0000"])
    blended_img = cover_vis.blend(loss_vis)

    bounds = list(roi_gdf.total_bounds)

    thumb_url = blended_img.getThumbURL(
        {
            "region": ee_geometry,
            "dimensions": "1024x1024",
            "format": "png",
        }
    )
    response = requests.get(thumb_url)
    response.raise_for_status()
    data_uri = "data:image/png;base64," + base64.b64encode(response.content).decode()

    legend = LegendSegment(
        title="Forest Change",
        values=[
            LegendValue(label="Forest Cover", color="rgba(35, 111, 33, 1)"),
            LegendValue(label="Forest Loss", color="rgba(255, 0, 0, 1)"),
        ],
    )

    return BitmapLayerDefinition(
        image=data_uri, bounds=bounds, opacity=opacity, legend=legend
    )
