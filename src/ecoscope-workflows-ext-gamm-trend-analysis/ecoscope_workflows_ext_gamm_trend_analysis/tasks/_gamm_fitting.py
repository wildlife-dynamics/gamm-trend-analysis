"""GAMM trend analysis tasks."""

from typing import Annotated, Literal, Optional
import numpy as np
from ecoscope_workflows_core.decorators import task
from pydantic import Field
from ecoscope.analysis.trend_analysis import GAMRegressor, optimize_gam
from .__gamm_utils import prepare_time_series_data, extract_trend_results
from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_core.skip import SKIP_SENTINEL, SkipSentinel


@task
def fit_gamm_model(
    dataframe: Annotated[AnyDataFrame, Field(description="DataFrame containing time series data")],
    time_column: Annotated[str, Field(description="Column name containing time/date values")] = "time",
    value_column: Annotated[str, Field(description="Column name containing values to analyze")] = "value",
    alpha: Annotated[
        Optional[float],
        Field(default=None, description="Smoothing parameter. If None, will be optimized."),
    ] = None,
    optimize_alpha: Annotated[bool, Field(default=True, description="Whether to optimize alpha parameter")] = True,
    metric: Annotated[
        Literal["AIC", "BIC", "Euclidean", "MSE", "R-Squared"],
        Field(default="AIC", description="Metric for optimization"),
    ] = "AIC",
    degree_of_freedom: Annotated[int, Field(default=20, description="Degrees of freedom for spline basis")] = 20,
    degree: Annotated[int, Field(default=3, description="Degree of B-spline basis")] = 3,
    family: Annotated[
        Literal["Gaussian", "Poisson", "Binomial"],
        Field(default="Gaussian", description="Distribution family for GLM"),
    ] = "Gaussian",
    lower_bound: Annotated[Optional[float], Field(default=None, description="Lower bound for spline knots")] = None,
    upper_bound: Annotated[Optional[float], Field(default=None, description="Upper bound for spline knots")] = None,
) -> dict | SkipSentinel:
    """
    Fit a GAM model to time series data.

    Returns a dictionary containing:
    - model_params: Model parameters (alpha, degree_of_freedom, etc.)
    - X: Time values used for fitting
    - y: Original values
    - metrics: Model metrics (AIC, BIC, R-squared, MSE)
    """
    # Prepare data
    X, y = prepare_time_series_data(dataframe, time_column, value_column)

    _metric = metric.lower().replace("-", "_")  # Ensure metric is in correct format for optimization
    _family = family.lower()  # Ensure family is in correct format for GAMRegressor

    if len(np.unique(y)) < 2:
        return SKIP_SENTINEL

    # Fit model
    if optimize_alpha and alpha is None:
        best_alpha, gam = optimize_gam(
            X=X,
            y=y,
            metric=_metric,
            degree_of_freedom=degree_of_freedom,
            degree=degree,
            family=_family,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    else:
        if alpha is None:
            alpha = 0.1  # default
        gam = GAMRegressor(
            alpha=alpha,
            degree_of_freedom=degree_of_freedom,
            degree=degree,
            family=_family,
        ).fit(X, y, lower_bound=lower_bound, upper_bound=upper_bound)
        best_alpha = alpha

    # Calculate metrics
    metrics = {
        "alpha": best_alpha,
        "aic": gam.aic(),
        "bic": gam.bic(),
        "r_squared": gam.r_squared(X, y),
        "mse": gam.mse(X, y),
    }

    # Return results (note: GAMRegressor can't be directly serialized)
    # Store model parameters instead
    return {
        "model_params": {
            "alpha": best_alpha,
            "degree_of_freedom": degree_of_freedom,
            "degree": degree,
            "family": _family,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
        "X": X.tolist(),
        "y": y.tolist(),
        "metrics": metrics,
    }


@task
def predict_gamm_trends(
    model_params: Annotated[dict, Field(description="Model parameters from fit_gamm_model")],
    time_values: Annotated[
        Optional[list],
        Field(
            default=None,
            description="Time values for prediction. If None, uses original training times.",
        ),
    ] = None,
    include_ci: Annotated[bool, Field(default=True, description="Include confidence intervals")] = True,
) -> AnyDataFrame:
    """
    Generate trend predictions from fitted GAM model.

    Note: This task refits the model with stored parameters since
    GAMRegressor objects cannot be serialized directly.
    """
    import numpy as np
    from ecoscope.analysis.trend_analysis import GAMRegressor

    # Reconstruct model
    params = model_params["model_params"]
    X_train = np.array(model_params["X"]).reshape(-1, 1)
    y_train = np.array(model_params["y"])

    gam = GAMRegressor(
        alpha=params["alpha"],
        degree_of_freedom=params["degree_of_freedom"],
        degree=params["degree"],
        family=params["family"],
    ).fit(
        X_train,
        y_train,
        lower_bound=params.get("lower_bound"),
        upper_bound=params.get("upper_bound"),
    )

    # Predict - use provided time_values or original training times
    if time_values is None:
        X_pred = X_train
        time_values = model_params["X"]
    else:
        X_pred = np.array(time_values).reshape(-1, 1)

    # Use utility function to extract results
    result_df = extract_trend_results(gam, X_pred, y_train, include_ci=include_ci)

    # Update time column if we used custom time_values (utility uses X.flatten())
    if time_values != model_params["X"]:
        result_df["time"] = time_values

    return result_df
