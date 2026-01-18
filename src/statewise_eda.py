import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config.api_config import RAW_DATA_DIR  # Assuming this points to Data/Raw/
from utils.state_mapper import normalize_state_name

# Set up seaborn for better plots
sns.set(style="whitegrid")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the DataFrame.
    - Creates 'total_updates' as sum of age group updates (int32).
    - Converts 'date' to datetime, drops invalid dates.
    - Extracts 'month_number' (int8) and 'month_name' (string).
    """
    # Sum age group updates into total_updates (optimize to int32)
    df['total_updates'] = (
        df['age_0_5'] + 
        df['age_5_17'] + 
        df['age_18_greater']
    ).astype('int32')
    
    # Convert date to datetime, coercing errors to NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows where date is NaT (invalid dates)
    df = df.dropna(subset=['date'])
    
    # Extract month number (int8 for efficiency, 1-12) - now safe since NaT are dropped
    df['month_number'] = df['date'].dt.month.astype('int8')
    
    # Extract month name
    df['month_name'] = df['date'].dt.month_name()
    
    return df

def perform_eda_and_plots(df: pd.DataFrame, state: str, output_dir: Path) -> str:
    """
    Performs basic EDA and generates visualizations.
    Saves plots to output_dir and returns a text summary.
    """
    
    # Aggregate data for plots
    monthly_total = df.groupby('month_name')['total_updates'].sum().reset_index()
    monthly_age_groups = df.groupby('month_name')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    district_total = df.groupby('district')['total_updates'].sum().reset_index().sort_values('total_updates', ascending=False).head(20)  # Top 20 districts
    
    # Plot 1: Month name vs. total_updates (Bar plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=monthly_total, x='month_name', y='total_updates', palette='viridis')
    plt.title(f'Total Updates by Month for {state}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot1_path = output_dir / f"{normalize_state_name(state)}_monthly_total.png"
    plt.savefig(plot1_path)
    plt.close()  # Free memory
    
    
    # Plot 2: Month name vs. updates by age groups (Stacked bar plot)
    plt.figure(figsize=(10, 6))
    monthly_age_groups.set_index('month_name').plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title(f'Updates by Age Groups and Month for {state}')
    plt.xticks(rotation=45)
    plt.ylabel('Updates')
    plt.tight_layout()
    plot2_path = output_dir / f"{normalize_state_name(state)}_monthly_age_groups.png"
    plt.savefig(plot2_path)
    plt.close()  # Free memory
    
    # Plot 3: District names vs. total_updates (Bar plot, top 20)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=district_total, x='district', y='total_updates', palette='coolwarm')
    plt.title(f'Total Updates by District for {state} (Top 20)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot3_path = output_dir / f"{normalize_state_name(state)}_district_total.png"
    plt.savefig(plot3_path)
    plt.close()  # Free memory

from config.api_config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EDA_PLOTS_DIR  # Add these to your config

def process_state_eda(state: str) -> str:
    """
    Full EDA pipeline for a single state: Load raw CSV, feature engineer, perform EDA/plots, save processed CSV.
    Saves to project directory (not notebook directory). Returns a text summary.
    Assumes enrolment dataset; adapt for others if needed.
    """
    normalized_state = normalize_state_name(state)
    raw_path = RAW_DATA_DIR / "enrolment" / f"{normalized_state}.csv"  # From config, e.g., Data/Raw/enrolment/
    processed_dir = PROCESSED_DATA_DIR / "enrolment"  # e.g., Data/Processed/enrolment/
    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = EDA_PLOTS_DIR / normalized_state  # e.g., Reports/EDA_Plots/{normalized_state}/
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw CSV
    df = pd.read_csv(raw_path)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Save processed CSV
    processed_path = processed_dir / f"{normalized_state}.csv"
    df.to_csv(processed_path, index=False)
    
    # Free memory
    del df
    