# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 15:59:03 2025

@author: aayus
"""

# basin_characteristics_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')

class BasinSynchronyAnalyzer:
    """
    Comprehensive analysis of how basin characteristics control streamflow spatial coherence.
    """
    
    def __init__(self, basin_char_dir, synchrony_scales, coordinates, output_dir):
        """
        Initialize the analyzer.
        
        Args:
            basin_char_dir: Directory with filtered basin characteristics
            synchrony_scales: Dictionary of station_id: synchrony_scale_km
            coordinates: Dictionary of station coordinates  
            output_dir: Where to save results
        """
        self.basin_char_dir = basin_char_dir
        self.synchrony_scales = synchrony_scales
        self.coordinates = coordinates
        self.output_dir = output_dir
        self.basin_data = {}
        self.merged_data = None
        
        # Define variable categories and their key variables
        self.variable_categories = {
            'Climate': {
                'sheet': 'Climate',
                'variables': ['PPTAVG_BASIN', 'T_AVG_BASIN', 'PET', 'SNOW_PCT_PRECIP', 
                             'PRECIP_SEAS_IND', 'WD_BASIN', 'WDMAX_BASIN', 'WDMIN_BASIN'],
                'labels': {
                    'PPTAVG_BASIN': 'Annual Precipitation (cm)',
                    'T_AVG_BASIN': 'Annual Temperature (°C)',
                    'PET': 'Potential ET (mm/yr)',
                    'SNOW_PCT_PRECIP': 'Snow % of Precip',
                    'PRECIP_SEAS_IND': 'Precip Seasonality',
                    'WD_BASIN': 'Wet Days/Year',
                    'WDMAX_BASIN': 'Max Monthly Wet Days',
                    'WDMIN_BASIN': 'Min Monthly Wet Days'
                }
            },
            'Topography': {
                'sheet': 'Topo',
                'variables': ['ELEV_MEAN_M_BASIN', 'ELEV_MAX_M_BASIN', 'SLOPE_PCT', 
                             'ASPECT_NORTHNESS', 'ASPECT_EASTNESS'],
                'labels': {
                    'ELEV_MEAN_M_BASIN': 'Mean Elevation (m)',
                    'ELEV_MAX_M_BASIN': 'Max Elevation (m)',
                    'SLOPE_PCT': 'Mean Slope (%)',
                    'ASPECT_NORTHNESS': 'Aspect Northness',
                    'ASPECT_EASTNESS': 'Aspect Eastness'
                }
            },
            'Hydrology': {
                'sheet': 'Hydro',
                'variables': ['STREAMS_KM_SQ_KM', 'BFI_AVE', 'RUNAVE7100', 'TOPWET'],
                'labels': {
                    'STREAMS_KM_SQ_KM': 'Stream Density (km/km²)',
                    'BFI_AVE': 'Base Flow Index',
                    'RUNAVE7100': 'Avg Annual Runoff (mm)',
                    'TOPWET': 'Topographic Wetness Index'
                }
            },
            'Land_Cover': {
                'sheet': 'LC06_Basin',
                'variables': ['FORESTNLCD06', 'DEVNLCD06', 'PLANTNLCD06', 'WATERNLCD06'],
                'labels': {
                    'FORESTNLCD06': 'Forest Cover (%)',
                    'DEVNLCD06': 'Developed Land (%)',
                    'PLANTNLCD06': 'Agriculture (%)',
                    'WATERNLCD06': 'Open Water (%)'
                }
            },
            'Soils': {
                'sheet': 'Soils',
                'variables': ['HGA', 'HGB', 'HGC', 'HGD', 'PERMAVE', 'CLAYAVE', 'SILTAVE', 'SANDAVE'],
                'labels': {
                    'HGA': 'Soil Group A (%)',
                    'HGB': 'Soil Group B (%)', 
                    'HGC': 'Soil Group C (%)',
                    'HGD': 'Soil Group D (%)',
                    'PERMAVE': 'Permeability (in/hr)',
                    'CLAYAVE': 'Clay Content (%)',
                    'SILTAVE': 'Silt Content (%)',
                    'SANDAVE': 'Sand Content (%)'
                }
            },
            'Basin_Info': {
                'sheet': 'BasinID',
                'variables': ['DRAIN_SQKM', 'STATE'],
                'labels': {
                    'DRAIN_SQKM': 'Drainage Area (km²)',
                    'STATE': 'State'
                }
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_all_basin_data(self):
        """Load all basin characteristics data sheets."""
        print("=== LOADING BASIN CHARACTERISTICS DATA ===")
        
        for category, info in self.variable_categories.items():
            sheet_path = os.path.join(self.basin_char_dir, f"filtered_{info['sheet']}.csv")
            
            if os.path.exists(sheet_path):
                df = pd.read_csv(sheet_path)
                df['STAID'] = df['STAID'].astype(str).str.zfill(8)
                self.basin_data[category] = df
                print(f"Loaded {category}: {df.shape[0]} stations, {df.shape[1]} variables")
            else:
                print(f"WARNING: {sheet_path} not found")
        
        # Create master merged dataset
        self._create_merged_dataset()
        
    def _create_merged_dataset(self):
        """Create a single merged dataset with all variables and synchrony scales."""
        print("\n=== CREATING MERGED DATASET ===")
        
        # Start with synchrony data
        sync_df = pd.DataFrame({
            'STAID': list(self.synchrony_scales.keys()),
            'synchrony_scale_km': list(self.synchrony_scales.values())
        })
        sync_df['STAID'] = sync_df['STAID'].astype(str).str.zfill(8)
        sync_df = sync_df[sync_df['synchrony_scale_km'] > 0]  # Filter out zero scales
        
        merged = sync_df.copy()
        
        # Merge each category
        for category, df in self.basin_data.items():
            merged = pd.merge(merged, df, on='STAID', how='left')
            print(f"After merging {category}: {merged.shape[0]} stations")
        
        # Add coordinate data
        if self.coordinates:
            coord_df = pd.DataFrame([
                {'STAID': k, 'latitude': v['lat'], 'longitude': v['lon']} 
                for k, v in self.coordinates.items()
            ])
            coord_df['STAID'] = coord_df['STAID'].astype(str).str.zfill(8)
            merged = pd.merge(merged, coord_df, on='STAID', how='left')
        
        self.merged_data = merged
        print(f"\nFinal merged dataset: {merged.shape[0]} stations, {merged.shape[1]} variables")
        
    def calculate_all_correlations(self):
        """Calculate correlations between all basin characteristics and synchrony scales."""
        print("\n=== CALCULATING CORRELATIONS ===")
        
        if self.merged_data is None:
            self.load_all_basin_data()
        
        correlation_results = []
        
        for category, info in self.variable_categories.items():
            print(f"\nAnalyzing {category}...")
            
            for var in info['variables']:
                if var in self.merged_data.columns:
                    # Clean data
                    valid_data = self.merged_data[['STAID', var, 'synchrony_scale_km']].dropna()
                    
                    if len(valid_data) >= 10:  # Need sufficient data
                        # Calculate Pearson correlation
                        r_pearson, p_pearson = stats.pearsonr(
                            valid_data[var], valid_data['synchrony_scale_km']
                        )
                        
                        # Calculate Spearman correlation (rank-based, more robust)
                        r_spearman, p_spearman = stats.spearmanr(
                            valid_data[var], valid_data['synchrony_scale_km']
                        )
                        
                        correlation_results.append({
                            'category': category,
                            'variable': var,
                            'variable_label': info['labels'].get(var, var),
                            'pearson_r': r_pearson,
                            'pearson_p': p_pearson,
                            'spearman_r': r_spearman,
                            'spearman_p': p_spearman,
                            'n_points': len(valid_data),
                            'pearson_significant': p_pearson < 0.05,
                            'spearman_significant': p_spearman < 0.05
                        })
        
        self.correlation_results = pd.DataFrame(correlation_results)
        
        # Save results
        results_path = os.path.join(self.output_dir, "basin_synchrony_correlations.csv")
        self.correlation_results.to_csv(results_path, index=False)
        
        return self.correlation_results
    
    def create_correlation_summary_plot(self):
        """Create a comprehensive correlation summary visualization."""
        print("\n=== CREATING CORRELATION SUMMARY PLOT ===")
        
        if not hasattr(self, 'correlation_results'):
            self.calculate_all_correlations()
        
        # Filter to significant correlations only for main plot
        sig_results = self.correlation_results[
            (self.correlation_results['spearman_significant']) & 
            (self.correlation_results['n_points'] >= 20)
        ].copy()
        
        if len(sig_results) == 0:
            print("No significant correlations found")
            return
        
        # Sort by absolute correlation strength
        sig_results['abs_spearman_r'] = sig_results['spearman_r'].abs()
        sig_results = sig_results.sort_values('abs_spearman_r', ascending=True)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(sig_results) * 0.4)))
        
        # Color by category
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.variable_categories)))
        category_colors = dict(zip(self.variable_categories.keys(), colors))
        
        bars = ax.barh(range(len(sig_results)), sig_results['spearman_r'],
                      color=[category_colors[cat] for cat in sig_results['category']])
        
        # Customize plot
        ax.set_yticks(range(len(sig_results)))
        ax.set_yticklabels(sig_results['variable_label'], fontsize=10)
        ax.set_xlabel('Spearman Correlation with Synchrony Scale', fontsize=12)
        ax.set_title('Basin Characteristics vs Streamflow Synchrony Scale\n(Significant correlations only)', 
                    fontsize=14, pad=20)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add correlation values as text
        for i, (_, row) in enumerate(sig_results.iterrows()):
            ax.text(row['spearman_r'] + 0.01 * np.sign(row['spearman_r']), i, 
                   f'{row["spearman_r"]:.3f}', 
                   va='center', fontsize=8)
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors[cat]) 
                          for cat in self.variable_categories.keys()]
        ax.legend(legend_elements, self.variable_categories.keys(), 
                 loc='lower right', title='Variable Category')
        
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(self.output_dir, "correlation_summary_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Correlation summary saved to: {plot_path}")
        plt.show()
        
    def create_category_comparison_plots(self):
        """Create detailed plots for each variable category."""
        print("\n=== CREATING CATEGORY COMPARISON PLOTS ===")
        
        if not hasattr(self, 'correlation_results'):
            self.calculate_all_correlations()
        
        for category, info in self.variable_categories.items():
            if category not in self.basin_data:
                continue
                
            print(f"Creating plots for {category}...")
            
            # Get variables for this category
            cat_results = self.correlation_results[
                self.correlation_results['category'] == category
            ].copy()
            
            if len(cat_results) == 0:
                continue
            
            # Filter to variables with sufficient data
            cat_results = cat_results[cat_results['n_points'] >= 10]
            available_vars = cat_results['variable'].tolist()
            
            if len(available_vars) == 0:
                continue
            
            # Create subplots
            n_vars = len(available_vars)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, var in enumerate(available_vars):
                ax = axes[i]
                
                # Get data for this variable
                var_data = self.merged_data[[var, 'synchrony_scale_km']].dropna()
                
                if len(var_data) < 10:
                    ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes, 
                           ha='center', va='center')
                    ax.set_title(info['labels'].get(var, var))
                    continue
                
                # Scatter plot
                ax.scatter(var_data[var], var_data['synchrony_scale_km'], 
                          alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
                
                # Add trend line
                z = np.polyfit(var_data[var], var_data['synchrony_scale_km'], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(var_data[var])
                ax.plot(x_sorted, p(x_sorted), 'r-', linewidth=2, alpha=0.8)
                
                # Get correlation info
                var_result = cat_results[cat_results['variable'] == var].iloc[0]
                r = var_result['spearman_r']
                p_val = var_result['spearman_p']
                sig = "*" if p_val < 0.05 else ""
                
                # Labels
                ax.set_xlabel(info['labels'].get(var, var))
                ax.set_ylabel('Synchrony Scale (km)')
                ax.set_title(f'{info["labels"].get(var, var)}\nr = {r:.3f}{sig} (n = {len(var_data)})')
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(available_vars), len(axes)):
                axes[i].remove()
            
            plt.suptitle(f'{category} Variables vs Synchrony Scale', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save
            plot_path = os.path.join(self.output_dir, f"{category.lower()}_synchrony_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def perform_multiple_regression_analysis(self):
        """Perform multiple regression to identify key controls on synchrony."""
        print("\n=== MULTIPLE REGRESSION ANALYSIS ===")
        
        if self.merged_data is None:
            self.load_all_basin_data()
        
        # Select key variables from each category (avoid multicollinearity)
        key_variables = {
            'Climate': ['PPTAVG_BASIN', 'T_AVG_BASIN', 'SNOW_PCT_PRECIP'],
            'Topography': ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT'],  
            'Hydrology': ['BFI_AVE', 'STREAMS_KM_SQ_KM'],
            'Land_Cover': ['FORESTNLCD06', 'DEVNLCD06'],
            'Soils': ['PERMAVE', 'CLAYAVE'],
            'Basin_Info': ['DRAIN_SQKM']
        }
        
        # Flatten to single list
        all_predictors = []
        for cat_vars in key_variables.values():
            all_predictors.extend(cat_vars)
        
        # Keep only available variables
        available_predictors = [var for var in all_predictors if var in self.merged_data.columns]
        
        # Create analysis dataset
        analysis_vars = ['synchrony_scale_km'] + available_predictors
        analysis_data = self.merged_data[analysis_vars].dropna()
        
        if len(analysis_data) < 30:
            print(f"Insufficient data for regression: {len(analysis_data)} stations")
            return
        
        print(f"Performing regression with {len(analysis_data)} stations and {len(available_predictors)} predictors")
        
        # Prepare data
        X = analysis_data[available_predictors]
        y = analysis_data['synchrony_scale_km']
        
        # Standardize predictors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Linear regression
        lr_model = LinearRegression()
        lr_model.fit(X_scaled, y)
        
        # Random forest for variable importance
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)
        
        # Cross-validation scores
        cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5)
        
        # Results
        results_df = pd.DataFrame({
            'variable': available_predictors,
            'coefficient': lr_model.coef_,
            'abs_coefficient': np.abs(lr_model.coef_),
            'rf_importance': rf_model.feature_importances_
        })
        
        # Add variable labels
        all_labels = {}
        for info in self.variable_categories.values():
            all_labels.update(info['labels'])
        
        results_df['variable_label'] = results_df['variable'].map(lambda x: all_labels.get(x, x))
        
        # Sort by importance
        results_df = results_df.sort_values('rf_importance', ascending=False)
        
        # Print results
        print(f"\nRegression Results (R² = {lr_model.score(X_scaled, y):.3f}):")
        print(f"Cross-validation score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"\nTop predictors:")
        print(f"{'Variable':<30} {'Coefficient':<12} {'RF Importance':<15} {'Direction'}")
        print("-" * 75)
        
        for _, row in results_df.head(10).iterrows():
            direction = "Higher → Longer" if row['coefficient'] > 0 else "Higher → Shorter"
            print(f"{row['variable_label']:<30} {row['coefficient']:8.3f}     "
                  f"{row['rf_importance']:8.3f}       {direction}")
        
        # Save results
        results_path = os.path.join(self.output_dir, "multiple_regression_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Create importance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Random Forest importance
        top_vars = results_df.head(10)
        ax1.barh(range(len(top_vars)), top_vars['rf_importance'], color='forestgreen')
        ax1.set_yticks(range(len(top_vars)))
        ax1.set_yticklabels(top_vars['variable_label'])
        ax1.set_xlabel('Random Forest Importance')
        ax1.set_title('Variable Importance (Random Forest)')
        
        # Linear regression coefficients
        ax2.barh(range(len(top_vars)), top_vars['coefficient'], 
                color=['red' if x < 0 else 'blue' for x in top_vars['coefficient']])
        ax2.set_yticks(range(len(top_vars)))
        ax2.set_yticklabels(top_vars['variable_label'])
        ax2.set_xlabel('Standardized Coefficient')
        ax2.set_title('Linear Regression Coefficients')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "regression_importance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df
    
    def analyze_by_regions(self):
        """Analyze how controls vary by geographic region (by state)."""
        print("\n=== REGIONAL ANALYSIS ===")
        
        if self.merged_data is None:
            self.load_all_basin_data()
        
        if 'STATE' not in self.merged_data.columns:
            print("STATE variable not available for regional analysis")
            return
        
        # Get states with sufficient data
        state_counts = self.merged_data['STATE'].value_counts()
        states_with_data = state_counts[state_counts >= 10].index.tolist()
        
        if len(states_with_data) < 3:
            print("Insufficient data for regional analysis")
            return
        
        print(f"Analyzing {len(states_with_data)} states with ≥10 stations each")
        
        # Calculate synchrony statistics by state
        regional_stats = []
        
        for state in states_with_data:
            state_data = self.merged_data[self.merged_data['STATE'] == state]
            
            regional_stats.append({
                'state': state,
                'n_stations': len(state_data),
                'mean_synchrony': state_data['synchrony_scale_km'].mean(),
                'median_synchrony': state_data['synchrony_scale_km'].median(),
                'std_synchrony': state_data['synchrony_scale_km'].std(),
                'mean_precip': state_data['PPTAVG_BASIN'].mean() if 'PPTAVG_BASIN' in state_data.columns else np.nan,
                'mean_temp': state_data['T_AVG_BASIN'].mean() if 'T_AVG_BASIN' in state_data.columns else np.nan,
                'mean_elevation': state_data['ELEV_MEAN_M_BASIN'].mean() if 'ELEV_MEAN_M_BASIN' in state_data.columns else np.nan
            })
        
        regional_df = pd.DataFrame(regional_stats)
        regional_df = regional_df.sort_values('mean_synchrony', ascending=False)
        
        # Print results
        print(f"\nSynchrony by State:")
        print(f"{'State':<6} {'N':<4} {'Mean (km)':<10} {'Median (km)':<12} {'Std (km)':<10}")
        print("-" * 50)
        for _, row in regional_df.iterrows():
            print(f"{row['state']:<6} {row['n_stations']:<4} {row['mean_synchrony']:8.1f}   "
                  f"{row['median_synchrony']:10.1f}   {row['std_synchrony']:8.1f}")
        
        # Create regional comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Synchrony by state
        ax = axes[0, 0]
        state_data_for_plot = [self.merged_data[self.merged_data['STATE'] == state]['synchrony_scale_km'].values 
                              for state in states_with_data]
        ax.boxplot(state_data_for_plot, labels=states_with_data)
        ax.set_ylabel('Synchrony Scale (km)')
        ax.set_title('Synchrony Scale by State')
        ax.tick_params(axis='x', rotation=45)
        
        # Correlation with precipitation
        if 'PPTAVG_BASIN' in self.merged_data.columns:
            ax = axes[0, 1]
            regional_clean = regional_df.dropna(subset=['mean_synchrony', 'mean_precip'])
            if len(regional_clean) > 3:
                ax.scatter(regional_clean['mean_precip'], regional_clean['mean_synchrony'])
                for _, row in regional_clean.iterrows():
                    ax.annotate(row['state'], (row['mean_precip'], row['mean_synchrony']))
                ax.set_xlabel('Mean Annual Precipitation (cm)')
                ax.set_ylabel('Mean Synchrony Scale (km)')
                ax.set_title('Regional Climate vs Synchrony')
        
        # Sample size by state
        ax = axes[1, 0]
        ax.bar(regional_df['state'], regional_df['n_stations'])
        ax.set_ylabel('Number of Stations')
        ax.set_title('Sample Sizes by State')
        ax.tick_params(axis='x', rotation=45)
        
        # Synchrony variability
        ax = axes[1, 1]
        ax.scatter(regional_df['mean_synchrony'], regional_df['std_synchrony'])
        for _, row in regional_df.iterrows():
            ax.annotate(row['state'], (row['mean_synchrony'], row['std_synchrony']))
        ax.set_xlabel('Mean Synchrony Scale (km)')
        ax.set_ylabel('Std Dev Synchrony Scale (km)')
        ax.set_title('Mean vs Variability of Synchrony')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "regional_analysis_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save regional stats
        results_path = os.path.join(self.output_dir, "regional_synchrony_stats.csv")
        regional_df.to_csv(results_path, index=False)
        
        return regional_df
    
    def create_comprehensive_report(self):
        """Generate a comprehensive summary report."""
        print("\n=== GENERATING COMPREHENSIVE REPORT ===")
        
        # Run all analyses
        self.load_all_basin_data()
        corr_results = self.calculate_all_correlations()
        self.create_correlation_summary_plot()
        self.create_category_comparison_plots()
        reg_results = self.perform_multiple_regression_analysis()
        regional_results = self.analyze_by_regions()
        
        # Create summary text report
        report_lines = [
            "BASIN CHARACTERISTICS CONTROL ON STREAMFLOW SPATIAL COHERENCE",
            "=" * 65,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Stations: {len(self.merged_data)}",
            f"Mean Synchrony Scale: {self.merged_data['synchrony_scale_km'].mean():.1f} km",
            f"Synchrony Range: {self.merged_data['synchrony_scale_km'].min():.1f} - {self.merged_data['synchrony_scale_km'].max():.1f} km",
            "",
            "TOP CONTROLLING FACTORS (Spearman correlation):",
            "-" * 50
        ]
        
        # Add top correlations
        top_correlations = corr_results.nlargest(10, 'spearman_r', keep='all')
        for _, row in top_correlations.iterrows():
            direction = "longer" if row['spearman_r'] > 0 else "shorter"
            report_lines.append(
                f"{row['variable_label']:<35} r = {row['spearman_r']:6.3f} (→ {direction} scales)"
            )
        
        if hasattr(self, 'regression_results'):
            report_lines.extend([
                "",
                "MULTIPLE REGRESSION KEY FINDINGS:",
                "-" * 40
            ])
            
            for _, row in reg_results.head(5).iterrows():
                direction = "increases" if row['coefficient'] > 0 else "decreases"
                report_lines.append(
                    f"{row['variable_label']:<35} {direction} synchrony (RF importance: {row['rf_importance']:.3f})"
                )
        
        # Save report
        report_path = os.path.join(self.output_dir, "comprehensive_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nComprehensive analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Report saved to: {report_path}")
        
        return report_lines

def run_basin_characteristics_analysis(basin_char_dir, synchrony_scales, coordinates, output_dir):
    """
    Convenience function to run the complete basin characteristics analysis.
    
    Args:
        basin_char_dir: Path to filtered basin characteristics directory
        synchrony_scales: Dictionary of station synchrony scales  
        coordinates: Dictionary of station coordinates
        output_dir: Output directory for results
    
    Returns:
        BasinSynchronyAnalyzer instance with all results
    """
    print("=== COMPREHENSIVE BASIN CHARACTERISTICS ANALYSIS ===")
    
    # Initialize analyzer
    analyzer = BasinSynchronyAnalyzer(basin_char_dir, synchrony_scales, coordinates, output_dir)
    
    # Run comprehensive analysis
    analyzer.create_comprehensive_report()
    
    return analyzer

# Example usage in main.py:
if __name__ == "__main__":
    # This would be integrated into your main.py
    
    # Example parameters (replace with your actual data)
    basin_char_dir = r"C:\Research_Data\basin_stuff\filtered_basin_characteristics"
    output_dir = "./basin_characteristics_analysis_output"
    
    # Your synchrony_scales and coordinates would come from your existing analysis
    # synchrony_scales = {...}  # From your correlation synchrony analysis
    # coordinates = {...}       # From your coordinate loading
    
    # Run analysis
    # analyzer = run_basin_characteristics_analysis(
    #     basin_char_dir, synchrony_scales, coordinates, output_dir
    # )