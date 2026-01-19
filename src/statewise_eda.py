import pandas as pd
import matplotlib.pyplot as plt
import gc
import seaborn as sns
from pathlib import Path
from config.api_config import RAW_DATA_DIR  # Assuming this points to Data/Raw/
from utils.state_mapper import normalize_state_name

# Set up seaborn for better plots
sns.set(style="whitegrid")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Sum age group updates into total_updates
    df['total_updates'] = (
        df['age_0_5'] + 
        df['age_5_17'] + 
        df['age_18_greater']
    ).astype('int32')
    
    # FIX: Added dayfirst=True to correctly parse DD-MM-YYYY and silence warning
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    # Drop rows where date is NaT
    df = df.dropna(subset=['date']).copy()
    
    # Extract month data
    df['month_number'] = df['date'].dt.month.astype('int8')
    df['month_name'] = df['date'].dt.month_name()
    
    return df

def perform_eda_and_plots(df: pd.DataFrame, state: str, output_dir: Path) -> str:
    """
    Performs basic EDA and generates visualizations.
    Saves plots to output_dir and returns a text summary.
    """
    
    # Aggregate data for plots
    monthly_total = df.groupby('month_name')['total_updates'].sum().reset_index()
    monthly_age_groups = df.groupby('month_name')[['age_0_5', 'age_5_17','age_18_greater']].sum().reset_index()
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
    normalized_state = normalize_state_name(state)
    raw_path = RAW_DATA_DIR / "enrolment" / f"{normalized_state}.csv"
    processed_dir = PROCESSED_DATA_DIR / "enrolment"
    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = EDA_PLOTS_DIR / normalized_state
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load raw CSV
        df = pd.read_csv(raw_path)
        
        if df.empty:
            return f" Warning: {state} dataset is empty."

        # Feature engineering
        df = feature_engineering(df)
        
        # Save processed CSV
        processed_path = processed_dir / f"{normalized_state}.csv"
        df.to_csv(processed_path, index=False)
        
        # Perform EDA and generate plots
        # Ensure perform_eda_and_plots returns a string summary!
        summary = perform_eda_and_plots(df, state, plots_dir)
        
        # Cleanup
        del df
        gc.collect()
        
        # FIX: Ensure a string is returned to avoid TypeError in the loop
        if not summary:
            summary = f"EDA completed for {state}. Plots saved to {plots_dir}."
        return summary

    except Exception as e:
        return f" Error processing {state}: {str(e)}"
    