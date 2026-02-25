"""Utility functions for GAMM trend analysis tasks."""
import numpy as np
import pandas as pd
from typing import Tuple
from ecoscope.analysis.trend_analysis import GAMRegressor


def prepare_time_series_data(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract time and value arrays from DataFrame for GAM fitting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    time_column : str
        Column name containing time/date values
    value_column : str
        Column name containing values to analyze
        
    Returns
    -------
    X : np.ndarray
        Time values as numpy array
    y : np.ndarray
        Value array for fitting
    """
    X = df[time_column].values
    y = df[value_column].values
    
    # Convert datetime to numeric if needed
    if pd.api.types.is_datetime64_any_dtype(df[time_column]):
        X = pd.to_numeric(df[time_column]).values
    
    return X, y


def extract_trend_results(
    gam: GAMRegressor,
    X: np.ndarray,
    include_ci: bool = True,
) -> pd.DataFrame:
    """
    Extract trend predictions and confidence intervals from fitted GAM.
    
    Parameters
    ----------
    gam : GAMRegressor
        Fitted GAM model
    X : np.ndarray
        Time values for prediction
    include_ci : bool
        Whether to include confidence intervals
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and optionally confidence intervals
    """
    if include_ci:
        mean, ci_lower, ci_upper = gam.predict_with_ci(X)
        return pd.DataFrame({
            'time': X.flatten(),
            'predicted': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })
    else:
        predictions = gam.predict(X)
        return pd.DataFrame({
            'time': X.flatten(),
            'predicted': predictions,
        })
