"""Forest cover trend analysis tasks."""

from typing import Annotated
from ecoscope_workflows_ext_ecoscope.connections import EarthEngineClient
from ecoscope_workflows_core.decorators import task
from pydantic import Field
from ecoscope.analysis.trend_analysis import get_forest_cover_trends
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def extract_forest_cover_trends(
    client: EarthEngineClient,
    aoi: Annotated[
        AnyDataFrame, Field(description="Area of interest geometry (must have CRS set)")
    ],
    tree_cover_threshold: Annotated[
        float, Field(default=60.0, description="Minimum tree cover percentage (0-100)")
    ] = 60.0,
    scale: Annotated[
        int, Field(default=30, description="Pixel scale in meters for reduction")
    ] = 30,
    max_pixels: Annotated[
        float, Field(default=1e9, description="Maximum pixels for reduction")
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
    return get_forest_cover_trends(
        aoi=aoi,
        tree_cover_threshold=tree_cover_threshold,
        scale=scale,
        max_pixels=max_pixels,
    )


# @task
# def plot_forest_model_historic_timeseries(
#     df: AnyDataFrame,
#     model: dict,
#     time_column: str = "year",
#     value_column: str = "survival_area",
#     model_key: str = "model",
#     model_title: str = "Model",
# ):
#     import numpy as np
#     from ecoscope.plotting.plot import draw_historic_timeseries
#
#     X = np.asarray(df[time_column], dtype=float)
#     mean, ci_lower, ci_upper = model.predict_with_ci(X)
#
#     df_plot = df.copy()
#     df_plot[f"{model_key}_mean"] = mean
#     df_plot["ci_lower"] = ci_lower
#     df_plot["ci_upper"] = ci_upper
#
#     y_axis_units = "acres"
#     y_axis_title = f"{value_column.replace('_', ' ').title()} ({y_axis_units})"
#
#     fig = draw_historic_timeseries(
#         df_plot,
#         current_value_column=value_column,
#         current_value_title=f"Observed {value_column}",
#         historic_min_column="ci_lower",
#         historic_max_column="ci_upper",
#         historic_band_title=f"{model_title} 95% CI",
#         historic_mean_column=f"{model_key}_mean",
#         historic_mean_title=f"{model_title} mean",
#         time_column=time_column,
#         layout_kwargs={
#             "title": f"Forest survival area trend ({model_title})",
#             "xaxis_title": "Year",
#             "yaxis_title": y_axis_title,
#         },
#     )
#     return fig
#
