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


def _parse_year_range(time_range: Optional[TimeRange]) -> tuple[Optional[int], Optional[int]]:
    """Return (start_year, end_year) from a TimeRange, or (None, None) if not provided."""
    import pandas as pd

    start_year = pd.to_datetime(time_range.since).year if time_range else None
    end_year = pd.to_datetime(time_range.until).year if time_range else None
    return start_year, end_year


def _make_treecover_mask(gfc, tree_cover_threshold: float):
    """Return (cover_img, cover_mask) from a Hansen GFC image.

    cover_img  — binary pixel image (values → 1) masked to the threshold
    cover_mask — the raw boolean mask used to filter loss pixels
    """
    cover = gfc.select(["treecover2000"])
    mask = cover.gte(tree_cover_threshold)
    cover = cover.unmask().updateMask(mask)
    cover = cover.And(cover)
    return cover, mask


@task
def extract_forest_cover_trends(
    client: EarthEngineClient,
    aoi: Annotated[AnyDataFrame, Field(description="Area of interest geometry (must have CRS set)")],
    image: str,
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
    start_year, end_year = _parse_year_range(time_range)

    # Calculate forested area in 2000
    treecover2000, treecover2000_mask = _make_treecover_mask(gfc, tree_cover_threshold)
    treecover2000_area = treecover2000.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=feat_coll,
        scale=scale,
        maxPixels=max_pixels,
    )

    forested_area = treecover2000_area.getInfo()["treecover2000"]
    forested_area = (forested_area or 0.0) * 0.000247105  # Convert sq.meters to acres

    # Calculate forest loss by year
    loss_year = gfc.select(["lossyear"])
    loss_area_img = gfc.select(["loss"]).updateMask(treecover2000_mask).multiply(ee.Image.pixelArea())

    # Apply server-side time filter via lossyear mask if range is provided
    # Note: we always calculate loss from 2000 for correct survival_area,
    # but we can mask out pixels beyond the end_year to optimize the reduction.
    if end_year:
        loss_area_img = loss_area_img.updateMask(loss_year.lte(max(0, end_year - 2000)))

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
    image: str,
    time_range: Annotated[
        Optional[TimeRange], Field(description="Time range for the trend analysis")
    ] = None,
    tree_cover_threshold: Annotated[float, Field(default=60.0)] = 60.0,
    opacity: Annotated[float, Field(default=1.0)] = 1.0,
) -> Annotated[BitmapLayerDefinition, Field()]:
    """Creates a forest cover and loss tile layer for the map."""
    import base64
    import ee
    import requests

    roi_gdf = aoi.to_crs("EPSG:4326")
    roi_geometry = roi_gdf.dissolve().geometry.iloc[0]
    ee_geometry = ee.Geometry(roi_geometry.__geo_interface__)

    gfc = ee.Image(image)
    forest_cover_img, forest_cover_mask = _make_treecover_mask(gfc, tree_cover_threshold)
    start_year, end_year = _parse_year_range(time_range)

    loss_year_img = gfc.select("lossyear")

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
