# simple_climate_synchrony_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def create_simple_climate_synchrony_plots(basin_char_dir, synchrony_scales, coordinates, output_dir):
    """
    Create simple scatter plots of synchrony scale vs individual climate variables.
    
    Args:
        basin_char_dir: Directory with filtered basin characteristics
        synchrony_scales: Dictionary of station_id: synchrony_scale_km
        coordinates: Dictionary of station coordinates
        output_dir: Where to save plots
    """
    
    print("=== SIMPLE CLIMATE VS SYNCHRONY PLOTS ===")
    
    # Load climate data
    climate_path = os.path.join(basin_char_dir, "filtered_Climate.csv")
    if not os.path.exists(climate_path):
        print(f"ERROR: Climate data not found at {climate_path}")
        return None
    
    climate_df = pd.read_csv(climate_path)
    
    # Create synchrony dataframe
    synchrony_df = pd.DataFrame({
        'STAID': list(synchrony_scales.keys()),
        'synchrony_scale_km': list(synchrony_scales.values())
    })
    
    # Normalize STAID format for matching
    synchrony_df['STAID'] = synchrony_df['STAID'].astype(str).str.zfill(8)
    climate_df['STAID'] = climate_df['STAID'].astype(str).str.zfill(8)
    
    # Merge climate with synchrony
    merged_df = pd.merge(climate_df, synchrony_df, on='STAID', how='inner')
    
    # Filter out zero synchrony scales
    merged_df = merged_df[merged_df['synchrony_scale_km'] > 0]
    
    print(f"Merged data for {len(merged_df)} stations")
    
    # Define climate variables to plot
    climate_variables = [
        'PPTAVG_BASIN',       # Mean annual precipitation (cm)
        'T_AVG_BASIN',        # Mean annual temperature (°C)  
        'PET',                # Potential evapotranspiration (mm/yr)
        'SNOW_PCT_PRECIP',    # Snow percent of precipitation (%)
        'PRECIP_SEAS_IND',    # Precipitation seasonality index
        'WD_BASIN',           # Wet days per year
        'WDMAX_BASIN',        # Maximum monthly wet days
        'WDMIN_BASIN'         # Minimum monthly wet days
    ]
    
    # Filter to available variables
    available_vars = [var for var in climate_variables if var in merged_df.columns]
    print(f"Plotting {len(available_vars)} climate variables")
    
    # Create plots
    n_vars = len(available_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Store results for summary
    correlation_results = []
    
    for i, var in enumerate(available_vars):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Remove NaN values
        valid_data = merged_df[[var, 'synchrony_scale_km']].dropna()
        
        if len(valid_data) > 10:
            # Scatter plot
            ax.scatter(valid_data[var], valid_data['synchrony_scale_km'], 
                      alpha=0.6, s=40, color='steelblue', edgecolors='white', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(valid_data[var], valid_data['synchrony_scale_km'], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(valid_data[var])
            ax.plot(x_sorted, p(x_sorted), 'r-', linewidth=2, alpha=0.8)
            
            # Calculate correlation and p-value
            corr, p_val = stats.pearsonr(valid_data[var], valid_data['synchrony_scale_km'])
            
            # Determine significance
            sig_marker = "*" if p_val < 0.05 else ""
            sig_text = "significant" if p_val < 0.05 else "not significant"
            
            # Store results
            correlation_results.append({
                'variable': var,
                'correlation': corr,
                'p_value': p_val,
                'n_points': len(valid_data),
                'significant': p_val < 0.05
            })
            
            # Labels and formatting
            ax.set_xlabel(get_variable_label(var), fontsize=10)
            ax.set_ylabel('Synchrony Scale (km)', fontsize=10)
            ax.set_title(f'{get_variable_label(var)}\nr = {corr:.3f}{sig_marker} (p = {p_val:.3f})\n{sig_text}', 
                        fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add some stats text
            ax.text(0.05, 0.95, f'n = {len(valid_data)}', 
                   transform=ax.transAxes, fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        else:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {var}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_variable_label(var))
    
    # Remove empty subplots
    for i in range(len(available_vars), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].remove()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "simple_climate_synchrony_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create and save summary table
    results_df = pd.DataFrame(correlation_results)
    results_df = results_df.sort_values('correlation', key=abs, ascending=False)
    
    print("\n=== CORRELATION SUMMARY ===")
    print(f"{'Variable':<25} {'Correlation':<12} {'P-value':<10} {'N':<6} {'Significant'}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        sig_marker = "*" if row['significant'] else ""
        print(f"{get_variable_label(row['variable']):<25} {row['correlation']:8.3f}{sig_marker:<4} "
              f"{row['p_value']:8.4f} {row['n_points']:4.0f} {row['significant']}")
    
    # Save correlation results
    results_path = os.path.join(output_dir, "climate_synchrony_correlations.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"\nPlot saved to: {plot_path}")
    print(f"Results saved to: {results_path}")
    
    return results_df

def get_variable_label(var_name):
    """Get clean labels for climate variables."""
    labels = {
        'PPTAVG_BASIN': 'Annual Precipitation (cm)',
        'T_AVG_BASIN': 'Annual Temperature (°C)', 
        'PET': 'Potential Evapotranspiration (mm/yr)',
        'SNOW_PCT_PRECIP': 'Snow % of Precipitation',
        'PRECIP_SEAS_IND': 'Precipitation Seasonality',
        'WD_BASIN': 'Wet Days per Year',
        'WDMAX_BASIN': 'Max Monthly Wet Days',
        'WDMIN_BASIN': 'Min Monthly Wet Days'
    }
    return labels.get(var_name, var_name)

