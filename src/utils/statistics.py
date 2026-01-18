import numpy as np
import pandas as pd
import os
import json
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc  # For memory management

# Set style for visually appealing plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ======================================================
# POLICY THRESHOLDS - CONFIGURE THESE BASED ON TARGETS
# ======================================================
POLICY_THRESHOLDS = {
    'critical_threshold': 1000,     # Below this = crisis (less than ~100/month over 12 months)
    'target_threshold': 3000,       # Minimum acceptable (~250/month)
    'excellence_threshold': 8000,   # High performance (~650/month)
    'cost_per_training': 200000,    # ₹2L per district training
    'value_per_update': 5000,       # ₹5000 per enrollment value (more realistic)
    'monthly_target': 250,          # Target updates per district per month (was 50, too low!)
    'expected_improvement': 120,    # Expected monthly increase after training
}

def compute_univariate_statistics(series: pd.Series) -> dict:
    """
    Compute classical and robust univariate statistics WITH POLICY CONTEXT.
    """
    series = series.dropna()
    
    if len(series) == 0:
        return {}

    mean = series.mean()
    median = series.median()
    variance = series.var(ddof=1)
    std_dev = series.std(ddof=1)

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    mad = np.median(np.abs(series - median))
    skewness = series.skew()
    excess_kurtosis = series.kurt()
    
    # POLICY-RELEVANT ADDITIONS
    p10 = series.quantile(0.10)
    p90 = series.quantile(0.90)
    p95 = series.quantile(0.95)
    
    # Performance categories
    critical_count = (series < POLICY_THRESHOLDS['critical_threshold']).sum()
    below_target_count = ((series >= POLICY_THRESHOLDS['critical_threshold']) & 
                          (series < POLICY_THRESHOLDS['target_threshold'])).sum()
    on_track_count = ((series >= POLICY_THRESHOLDS['target_threshold']) & 
                      (series < POLICY_THRESHOLDS['excellence_threshold'])).sum()
    excellent_count = (series >= POLICY_THRESHOLDS['excellence_threshold']).sum()
    
    total_count = series.count()
    
    # Inequality measure (Gini coefficient)
    def gini_coefficient(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        if cumsum[-1] == 0:
            return 0
        return (2 * np.sum((n - np.arange(1, n+1) + 1) * sorted_x)) / (n * cumsum[-1]) - 1
    
    gini = gini_coefficient(series.values)
    
    # Concentration: what % do top 10% handle?
    sorted_series = series.sort_values(ascending=False)
    top_10_pct_idx = max(1, int(len(sorted_series) * 0.1))
    total_sum = sorted_series.sum()
    top_10_pct_contribution = sorted_series.head(top_10_pct_idx).sum() / total_sum * 100 if total_sum > 0 else 0
    
    bottom_50_pct_idx = int(len(sorted_series) * 0.5)
    bottom_50_pct_contribution = sorted_series.tail(bottom_50_pct_idx).sum() / total_sum * 100 if total_sum > 0 else 0

    return {
        # Traditional statistics
        "mean": float(mean),
        "median": float(median),
        "variance": float(variance),
        "std_dev": float(std_dev),
        "iqr": float(iqr),
        "mad": float(mad),
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
        "min": float(series.min()),
        "max": float(series.max()),
        "count": int(total_count),
        
        # Policy-relevant percentiles
        "p10_bottom_10pct": float(p10),
        "q1_25th_percentile": float(q1),
        "q3_75th_percentile": float(q3),
        "p90_top_10pct": float(p90),
        "p95_top_5pct": float(p95),
        
        # Performance categories
        "critical_count": int(critical_count),
        "critical_percentage": float(critical_count / total_count * 100),
        "below_target_count": int(below_target_count),
        "below_target_percentage": float(below_target_count / total_count * 100),
        "on_track_count": int(on_track_count),
        "on_track_percentage": float(on_track_count / total_count * 100),
        "excellent_count": int(excellent_count),
        "excellent_percentage": float(excellent_count / total_count * 100),
        
        # Inequality measures
        "gini_coefficient": float(gini),
        "inequality_interpretation": "HIGH" if gini > 0.6 else "MODERATE" if gini > 0.4 else "LOW",
        "top_10pct_contribution": float(top_10_pct_contribution),
        "bottom_50pct_contribution": float(bottom_50_pct_contribution),
        
        # Gap analysis
        "median_mean_gap": float(mean - median),
        "gap_interpretation": "High inequality - few outliers driving average up" if mean > 2 * median else "Moderate inequality",
        
        # Intervention estimates
        "districts_needing_intervention": int(critical_count + below_target_count),
        "intervention_percentage": float((critical_count + below_target_count) / total_count * 100),
        "estimated_training_cost_lakhs": float((critical_count + below_target_count) * POLICY_THRESHOLDS['cost_per_training'] / 100000),
    }


def compute_stats_for_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute univariate statistics for multiple columns."""
    results = {}
    for col in columns:
        results[col] = compute_univariate_statistics(df[col])
    return pd.DataFrame(results).T


# ======================================================
# DISTRICT-LEVEL AGGREGATION AND ANALYSIS
# ======================================================

def aggregate_by_district(df: pd.DataFrame, district_col: str, value_col: str) -> pd.DataFrame:
    """
    Aggregate data by district with performance categorization.
    Returns district-level summary with names and categories.
    """
    district_agg = df.groupby(district_col).agg({
        value_col: ['sum', 'mean', 'median', 'std', 'count']
    }).reset_index()
    
    district_agg.columns = [district_col, 'total_updates', 'mean_daily_updates', 
                            'median_daily_updates', 'std_updates', 'days_active']
    
    # Add performance category
    def categorize_performance(total):
        if total < POLICY_THRESHOLDS['critical_threshold']:
            return 'Critical'
        elif total < POLICY_THRESHOLDS['target_threshold']:
            return 'Below Target'
        elif total < POLICY_THRESHOLDS['excellence_threshold']:
            return 'On Track'
        else:
            return 'Excellent'
    
    district_agg['performance_category'] = district_agg['total_updates'].apply(categorize_performance)
    district_agg['needs_intervention'] = district_agg['performance_category'].isin(['Critical', 'Below Target'])
    
    return district_agg.sort_values('total_updates', ascending=False)


def get_district_month_summary(df: pd.DataFrame, district_col: str, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Get month-wise summary for each district with anomaly flags.
    """
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.strftime('%B')
    df_copy['year_month'] = pd.to_datetime(df_copy[date_col]).dt.to_period('M')
    
    monthly_summary = df_copy.groupby([district_col, 'month', 'year_month']).agg({
        value_col: 'sum'
    }).reset_index()
    
    monthly_summary.columns = [district_col, 'month', 'year_month', 'monthly_updates']
    
    # Add performance flags
    monthly_summary['meets_target'] = monthly_summary['monthly_updates'] >= POLICY_THRESHOLDS['monthly_target']
    monthly_summary['critical'] = monthly_summary['monthly_updates'] < POLICY_THRESHOLDS['critical_threshold']
    
    # Calculate month-over-month change for anomaly detection
    monthly_summary = monthly_summary.sort_values([district_col, 'year_month'])
    monthly_summary['pct_change'] = monthly_summary.groupby(district_col)['monthly_updates'].pct_change()
    monthly_summary['concerning_drop'] = monthly_summary['pct_change'] < -0.3  # 30% drop
    monthly_summary['sudden_spike'] = monthly_summary['pct_change'] > 1.0  # 100% increase
    
    return monthly_summary


# ======================================================
# TEMPORAL STATISTICS
# ======================================================

def perform_stl_decomposition(df: pd.DataFrame, date_col: str, value_col: str, period: int = 12) -> pd.DataFrame:
    """STL decomposition with policy interpretation."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    ts_df = df[[date_col, value_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col])
    ts_df = ts_df.sort_values(date_col).set_index(date_col)
    
    # Resample to monthly frequency
    ts_df = ts_df.resample('M').sum()
    
    if len(ts_df) < period:
        raise ValueError(f"Not enough data for STL decomposition. Need at least {period} months.")
    
    stl = STL(ts_df[value_col], period=period, robust=True)
    result = stl.fit()

    out = ts_df.copy()
    out["trend"] = result.trend
    out["seasonal"] = result.seasonal
    out["residual"] = result.resid
    
    # Policy additions
    out['trend_direction'] = 'stable'
    out.loc[out['trend'].diff() > out['trend'].std() * 0.5, 'trend_direction'] = 'increasing'
    out.loc[out['trend'].diff() < -out['trend'].std() * 0.5, 'trend_direction'] = 'decreasing'
    
    out['meets_monthly_target'] = out[value_col] >= POLICY_THRESHOLDS['monthly_target']
    out['concerning_drop'] = out[value_col].pct_change() < -0.3
    
    out.reset_index(inplace=True)
    out.rename(columns={date_col: 'month'}, inplace=True)
    out['month'] = out['month'].dt.strftime('%B')
    
    return out


def bayesian_change_point_detection(series: pd.Series, window: int = 3, threshold: float = 3.0) -> pd.DataFrame:
    """Bayesian-style change point detection with policy interpretation."""
    if series.empty:
        raise ValueError("Input series is empty.")
    
    monthly_series = series.resample('M').sum()
    df = pd.DataFrame({"value": monthly_series}).copy()
    
    df["posterior_mean"] = df["value"].rolling(window).mean()
    df["posterior_std"] = df["value"].rolling(window).std()
    df["z_score"] = (df["value"] - df["posterior_mean"]) / df["posterior_std"]
    df["change_point"] = df["z_score"].abs() > threshold
    
    # Policy additions
    df['change_type'] = 'normal'
    df.loc[(df['change_point']) & (df['z_score'] < -threshold), 'change_type'] = 'crisis_drop'
    df.loc[(df['change_point']) & (df['z_score'] > threshold), 'change_type'] = 'sudden_improvement'
    
    df['requires_investigation'] = df['change_type'].isin(['crisis_drop', 'sudden_improvement'])
    df['policy_note'] = ''
    df.loc[df['change_type'] == 'crisis_drop', 'policy_note'] = 'Investigate: What caused this drop?'
    df.loc[df['change_type'] == 'sudden_improvement', 'policy_note'] = 'Document success: What intervention worked?'
    
    df.reset_index(inplace=True)
    df.rename(columns={series.index.name or 'date': 'month'}, inplace=True)
    df['month'] = df['month'].dt.strftime('%B')
    
    return df


def build_district_month_matrix(df: pd.DataFrame, district_col: str, date_col: str, value_col: str) -> pd.DataFrame:
    """Creates a District × Month matrix."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["month"] = data[date_col].dt.strftime('%B')

    matrix = data.groupby([district_col, "month"])[value_col].sum().unstack(fill_value=0)
    return matrix


def compute_matrix_stats(matrix: pd.DataFrame) -> pd.DataFrame:
    """Computes row/column level statistics with policy interpretation."""
    stats = pd.DataFrame({
        "row_sum_total_updates": matrix.sum(axis=1),
        "row_mean_monthly_updates": matrix.mean(axis=1),
        "row_variance": matrix.var(axis=1),
        "row_min": matrix.min(axis=1),
        "row_max": matrix.max(axis=1),
    })
    
    # Policy additions
    stats['performance_category'] = 'below_target'
    stats.loc[stats['row_mean_monthly_updates'] >= POLICY_THRESHOLDS['target_threshold'], 'performance_category'] = 'on_track'
    stats.loc[stats['row_mean_monthly_updates'] >= POLICY_THRESHOLDS['excellence_threshold'], 'performance_category'] = 'excellent'
    stats.loc[stats['row_mean_monthly_updates'] < POLICY_THRESHOLDS['critical_threshold'], 'performance_category'] = 'critical'
    
    stats['needs_intervention'] = stats['performance_category'].isin(['critical', 'below_target'])
    stats['consistency_score'] = 1 / (1 + stats['row_variance'])
    stats['months_below_target'] = (matrix < POLICY_THRESHOLDS['monthly_target']).sum(axis=1)

    return stats


def compute_zscore_anomalies(matrix: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Detects abnormal district-months using Z-score with policy interpretation."""
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1)

    z_scores = matrix.sub(mean, axis=0).div(std, axis=0)

    anomalies = z_scores.stack().reset_index().rename(columns={0: "z_score"})
    anomalies["is_anomaly"] = anomalies["z_score"].abs() > z_thresh
    
    # Policy additions
    anomalies['anomaly_type'] = 'normal'
    anomalies.loc[(anomalies['is_anomaly']) & (anomalies['z_score'] < -z_thresh), 'anomaly_type'] = 'underperformance'
    anomalies.loc[(anomalies['is_anomaly']) & (anomalies['z_score'] > z_thresh), 'anomaly_type'] = 'exceptional_performance'
    
    anomalies['action_required'] = ''
    anomalies.loc[anomalies['anomaly_type'] == 'underperformance', 'action_required'] = 'Investigate root cause'
    anomalies.loc[anomalies['anomaly_type'] == 'exceptional_performance', 'action_required'] = 'Document best practices'

    return anomalies


# ======================================================
# POLICY INSIGHT GENERATION
# ======================================================

def generate_policy_insights(stats_dict: dict, state: str, district_agg: pd.DataFrame, 
                            monthly_summary: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
    """Generate actionable policy insights with district names and month-wise details."""
    insights = {
        'state': state,
        'executive_summary': [],
        'priority_actions': [],
        'critical_districts': [],
        'success_stories': [],
        'month_wise_anomalies': [],
        'risk_alerts': [],
        'resource_allocation': {}
    }
    
    # Extract univariate stats
    uni_stats = stats_dict.get('univariate_stats', {}).get('total_updates', {})
    
    # EXECUTIVE SUMMARY
    if uni_stats: