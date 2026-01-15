# src/utils/statistics.py

import numpy as np
import pandas as pd


def compute_univariate_statistics(series: pd.Series) -> dict:
    """
    Compute classical and robust univariate statistics.

    Parameters
    ----------
    series : pd.Series
        Numeric pandas series

    Returns
    -------
    dict
        Dictionary containing statistical measures
    """
    series = series.dropna()

    mean = series.mean()
    median = series.median()
    variance = series.var(ddof=1)
    std_dev = series.std(ddof=1)

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    mad = np.median(np.abs(series - median))

    skewness = series.skew()
    excess_kurtosis = series.kurt()  # Fisher definition (normal = 0)

    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "std_dev": std_dev,
        "iqr": iqr,
        "mad": mad,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "min": series.min(),
        "max": series.max(),
        "count": series.count()
    }


def compute_stats_for_dataframe(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """
    Compute univariate statistics for multiple columns.

    Returns a DataFrame (rows = variables, columns = stats)
    """
    results = {}

    for col in columns:
        results[col] = compute_univariate_statistics(df[col])

    return pd.DataFrame(results).T


from statsmodels.tsa.seasonal import STL


# ======================================================
# TEMPORAL STATISTICS
# ======================================================

def perform_stl_decomposition(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: int
) -> pd.DataFrame:
    """
    Performs STL decomposition on a time series.

    Returns dataframe with:
    - trend
    - seasonal
    - residual
    """

    ts_df = df[[date_col, value_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col])
    ts_df = ts_df.sort_values(date_col)
    ts_df = ts_df.set_index(date_col)

    stl = STL(ts_df[value_col], period=period, robust=True)
    result = stl.fit()

    out = ts_df.copy()
    out["trend"] = result.trend
    out["seasonal"] = result.seasonal
    out["residual"] = result.resid

    return out.reset_index()


# ======================================================
# BAYESIAN CHANGE POINT DETECTION
# ======================================================

def bayesian_change_point_detection(
    series: pd.Series,
    window: int = 30,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Simple Bayesian-style change point detection using
    rolling posterior mean + std.

    Returns dataframe with:
    - posterior_mean
    - posterior_std
    - z_score
    - change_point_flag
    """

    df = pd.DataFrame({"value": series}).copy()

    df["posterior_mean"] = df["value"].rolling(window).mean()
    df["posterior_std"] = df["value"].rolling(window).std()

    df["z_score"] = (
        (df["value"] - df["posterior_mean"]) / df["posterior_std"]
    )

    df["change_point"] = df["z_score"].abs() > threshold

    return df


def build_district_month_matrix(
    df: pd.DataFrame,
    district_col: str,
    date_col: str,
    value_col: str
) -> pd.DataFrame:
    """
    Creates a District × Month matrix (cells = total updates)
    """
    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col])
    data["month"] = data[date_col].dt.to_period("M").astype(str)

    matrix = (
        data
        .groupby([district_col, "month"])[value_col]
        .sum()
        .unstack(fill_value=0)
    )

    return matrix

def compute_matrix_stats(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Computes row/column level statistics
    """
    stats = pd.DataFrame({
        "row_sum_total_updates": matrix.sum(axis=1),
        "row_variance": matrix.var(axis=1)
    })

    return stats


def compute_zscore_anomalies(
    matrix: pd.DataFrame,
    z_thresh: float = 3.0
) -> pd.DataFrame:
    """
    Detects abnormal district-months using Z-score
    """
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1)

    z_scores = matrix.sub(mean, axis=0).div(std, axis=0)

    anomalies = (
        z_scores
        .stack()
        .reset_index()
        .rename(columns={0: "z_score"})
    )

    anomalies["is_anomaly"] = anomalies["z_score"].abs() > z_thresh

    return anomalies



