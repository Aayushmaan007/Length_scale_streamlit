# Updated main.py with correlation synchrony analysis and baseline integration
import yaml
import os
import numpy as np
import pandas as pd
import pickle
import data_loader
import analysis
import visualization
import mapping 
import simple_climate_synchrony_plots as scp 
import basin_characteristics_analysis as bca

def save_intermediate_results(correlation_matrix, synchrony_scales, output_dir):
    """Save correlation matrix and synchrony scales for future runs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save correlation matrix
    correlation_path = os.path.join(output_dir, "correlation_matrix.pkl")
    with open(correlation_path, 'wb') as f:
        pickle.dump(correlation_matrix, f)
    print(f"Saved correlation matrix to: {correlation_path}")
    
    # Save synchrony scales
    synchrony_path = os.path.join(output_dir, "synchrony_scales.pkl")
    with open(synchrony_path, 'wb') as f:
        pickle.dump(synchrony_scales, f)
    print(f"Saved synchrony scales to: {synchrony_path}")

def load_intermediate_results(output_dir):
    """Load previously calculated correlation matrix and synchrony scales."""
    correlation_path = os.path.join(output_dir, "correlation_matrix.pkl")
    synchrony_path = os.path.join(output_dir, "synchrony_scales.pkl")
    
    correlation_matrix = None
    synchrony_scales = None
    
    # Load correlation matrix
    if os.path.exists(correlation_path):
        with open(correlation_path, 'rb') as f:
            correlation_matrix = pickle.load(f)
        print(f"Loaded correlation matrix from cache: {correlation_matrix.shape}")
    
    # Load synchrony scales
    if os.path.exists(synchrony_path):
        with open(synchrony_path, 'rb') as f:
            synchrony_scales = pickle.load(f)
        print(f"Loaded synchrony scales from cache: {len(synchrony_scales)} stations")
    
    return correlation_matrix, synchrony_scales

def run_analysis_pipeline(config_path='config.yaml'):
    """Main function to run the entire analysis workflow."""

    # --- Control Switches ---
    RUN_PDF_PLOTS = False
    RUN_CORRELATION_MAP = False
    RUN_AVG_CORRELATION_HEATMAP = False
    RUN_CORRELATION_SYNCHRONY = False 
    RUN_SYNCHRONY_MAP = True
    RUN_ENSO_SYNCHRONY_COMPARISON = True
    RUN_CORRELATION_DISTANCE_ANALYSIS = True
    RUN_CLIMATE_SYNCHRONY_ANALYSIS = True
    
    # --- Caching Control ---
    USE_CACHED_RESULTS = True  # Set to False to force recalculation
    SAVE_RESULTS = True        # Set to False to skip saving cached results     
    # ------------------------

    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = config['base_path']
    paths = config['paths']
    filenames = config['filenames']
    output_dir = config['output_dir']

    # 2. Load all data using the data_loader module
    print("--- Loading Data ---")
    coordinates = data_loader.load_coordinates(os.path.join(base_path, paths['coordinates']))
    
    # Try to load cached results first
    correlation_matrix = None
    if USE_CACHED_RESULTS:
        print("--- Attempting to Load Cached Results ---")
        correlation_matrix, _ = load_intermediate_results(output_dir)
    
    # Load observed data and calculate correlation if not cached
    if correlation_matrix is None:
        print("--- Loading Observed Data and Calculating Correlation ---")
        observed_data = data_loader.load_observed_data(
            os.path.join(base_path, paths['observed']), 
            filenames['observed_pattern']
        )
        
        # 3. Perform analysis using the analysis module
        print("\n--- Calculating Inter-Station Spearman Correlation ---")
        correlation_matrix = analysis.calculate_inter_station_correlation(observed_data)
        
        if SAVE_RESULTS:
            # Save just the correlation matrix for now
            correlation_path = os.path.join(output_dir, "correlation_matrix.pkl")
            os.makedirs(output_dir, exist_ok=True)
            with open(correlation_path, 'wb') as f:
                pickle.dump(correlation_matrix, f)
            print(f"Saved correlation matrix to cache")
    else:
        print("Using cached correlation matrix - loading observed data for other analyses")
        observed_data = data_loader.load_observed_data(
            os.path.join(base_path, paths['observed']), 
            filenames['observed_pattern']
        )
    
    # 4. Generate PDF Plot Visualizations
    if RUN_PDF_PLOTS:
        print("\n--- Generating PDF Plot Visualizations ---")
        elnino_data = data_loader.load_simulation_data(
            os.path.join(base_path, paths['elnino']), 
            filenames['elnino_pattern'], 
            '+1elnino_flows_'
        )
        lanina_data = data_loader.load_simulation_data(
            os.path.join(base_path, paths['lanina']), 
            filenames['lanina_pattern'], 
            '-1lanina_flows_'
        )
        baseline_data = data_loader.load_baseline_data(
            os.path.join(base_path, paths['baseline']), 
            filenames['baseline_pattern']
        )
        station_stats = analysis.calculate_all_station_stats(observed_data, elnino_data, lanina_data, baseline_data)
        print(f"Successfully calculated stats for {len(station_stats)} stations.")
        
        if not station_stats:
            print("No stations with sufficient data to plot.")
        else:
            first_station_id = list(station_stats.keys())[0]
            visualization.create_pdf_plot(
                station_id=first_station_id,
                obs_data=observed_data[first_station_id],
                elnino_realizations=elnino_data[first_station_id],
                lanina_realizations=lanina_data[first_station_id],
                save_path=os.path.join(output_dir, f"pdf_station_{first_station_id}.png"),
                log_scale=False
            )
    else:
        print("\n--- Skipping PDF Plot Visualizations ---")

    # 5. Generate Correlation Map (Base Station)
    if RUN_CORRELATION_MAP:
        print("\n--- Generating Correlation Map ---")
        if not correlation_matrix.empty and coordinates:
            base_station_for_map = correlation_matrix.columns[0]
            print(f"Creating map with base station: {base_station_for_map}")
            
            mapping.create_correlation_map(
                base_station_id=base_station_for_map,
                correlation_matrix=correlation_matrix,
                coordinates=coordinates,
                save_path=os.path.join(output_dir, "station_correlation_map.png")
            )
        else:
            print("Skipping map generation due to missing correlation or coordinate data.")
    else:
        print("\n--- Skipping Correlation Map ---")

    # 6. Generate Average Correlation Heatmap
    if RUN_AVG_CORRELATION_HEATMAP:
        print("\n--- Generating Average Correlation Heatmap ---")
        if not correlation_matrix.empty and coordinates:
            average_correlations = analysis.calculate_average_correlation(correlation_matrix)
            print("Average correlations calculated:")
            print(average_correlations.head())

            mapping.create_average_correlation_map(
                average_correlations=average_correlations,
                coordinates=coordinates,
                save_path=os.path.join(output_dir, "average_correlation_heatmap.png")
            )
        else:
            print("Skipping heatmap generation due to missing correlation or coordinate data.")
    else:
        print("\n--- Skipping Average Correlation Heatmap ---")

    # 7. NEW: Calculate Correlation Synchrony Scales
    synchrony_scales = None
    
    # Try to load cached synchrony scales first
    if USE_CACHED_RESULTS:
        _, synchrony_scales = load_intermediate_results(output_dir)
    
    if RUN_CORRELATION_SYNCHRONY or RUN_CLIMATE_SYNCHRONY_ANALYSIS:
        if synchrony_scales is None:
            print("\n--- Calculating Correlation Synchrony Scales ---")
            if not correlation_matrix.empty and coordinates:
                synchrony_scales = analysis.calculate_correlation_synchrony_scale(
                    correlation_matrix, 
                    coordinates,
                    correlation_threshold=0.7,  # Stations must have r >= 0.7
                    fraction_threshold=0.5,     # At least 50% of stations in radius must meet threshold
                    max_radius_km=4000,
                    radius_step_km=25
                )
                
                # Calculate summary statistics
                stats = analysis.calculate_mean_correlation_synchrony_scale(synchrony_scales)
                
                print(f"Correlation Synchrony Scale Results:")
                print(f"  Mean: {stats['mean']:.1f} km")
                print(f"  Median: {stats['median']:.1f} km") 
                print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f} km")
                print(f"  Std Dev: {stats['std']:.1f} km")
                print(f"  Number of stations: {stats['count']}")
                
                # Print some individual examples
                print(f"\nExample synchrony scales for first 5 stations:")
                for i, (station_id, scale) in enumerate(list(synchrony_scales.items())[:5]):
                    print(f"  Station {station_id}: {scale} km")
                
                # Save synchrony scales
                if SAVE_RESULTS:
                    synchrony_path = os.path.join(output_dir, "synchrony_scales.pkl")
                    with open(synchrony_path, 'wb') as f:
                        pickle.dump(synchrony_scales, f)
                    print(f"Saved synchrony scales to cache")
                        
            else:
                print("Skipping synchrony analysis due to missing correlation or coordinate data.")
        else:
            print("\n--- Using Cached Synchrony Scales ---")
            stats = analysis.calculate_mean_correlation_synchrony_scale(synchrony_scales)
            print(f"Cached Synchrony Scale Stats: Mean = {stats['mean']:.1f} km, N = {stats['count']}")
    else:
        print("\n--- Skipping Correlation Synchrony Analysis ---")

    # 8. NEW: Create Synchrony Scale Map
    if RUN_SYNCHRONY_MAP:
        print("\n--- Generating Correlation Synchrony Map ---")
        if synchrony_scales is not None and coordinates:
            # Convert to pandas Series for compatibility with existing mapping functions
            synchrony_series = pd.Series(synchrony_scales)
            
            mapping.create_synchrony_scale_map(
                synchrony_scales=synchrony_series,
                coordinates=coordinates,
                save_path=os.path.join(output_dir, "correlation_synchrony_map.png")
            )
        else:
            print("Skipping synchrony map due to missing data.")
    else:
        print("\n--- Skipping Synchrony Scale Map ---")
    
    # 9. NEW: Simple Climate Controls on Synchrony Analysis
    if RUN_CLIMATE_SYNCHRONY_ANALYSIS:
        print("\n--- Simple Climate vs Synchrony Plots ---")
        
        if synchrony_scales is None:
            print("ERROR: Synchrony scales not available. Need to calculate them first.")
            print("Set RUN_CORRELATION_SYNCHRONY = True or ensure cached results exist.")
        else:
            # Set paths for basin characteristics and simple climate analysis output
            basin_char_dir = r"C:\Research_Data\basin_stuff\filtered_basin_characteristics"
            climate_output_dir = os.path.join(output_dir, "simple_climate_plots")
            
            # Run the simple climate-synchrony analysis
            climate_corr_results = scp.create_simple_climate_synchrony_plots(
                basin_char_dir=basin_char_dir,
                synchrony_scales=synchrony_scales,
                coordinates=coordinates,
                output_dir=climate_output_dir
            )
            
            # Print summary of findings
            if climate_corr_results is not None:
                print("\nTop 3 climate drivers of synchrony:")
                for i, (_, row) in enumerate(climate_corr_results.head(3).iterrows()):
                    direction = "longer" if row['correlation'] > 0 else "shorter"
                    sig_text = " (significant)" if row['significant'] else ""
                    print(f"  {i+1}. {scp.get_variable_label(row['variable'])}: r = {row['correlation']:.3f}{sig_text}")
                    print(f"     Higher values → {direction} synchrony scales")
            
            print(f"\nSimple climate plots saved to: {climate_output_dir}")
    else:
        print("\n--- Skipping Simple Climate Analysis ---")
        
    # 10. ENSO + Baseline Synchrony Comparison
    if RUN_ENSO_SYNCHRONY_COMPARISON:
        print("\n--- ALL CONDITIONS SYNCHRONY SCALE COMPARISON ---")
        
        # STEP 1: Load all simulation data (including baseline)
        print("Loading simulation data...")
        elnino_data = data_loader.load_simulation_data(
            os.path.join(base_path, paths['elnino']), 
            filenames['elnino_pattern'], 
            '+1elnino_flows_'
        )
        lanina_data = data_loader.load_simulation_data(
            os.path.join(base_path, paths['lanina']), 
            filenames['lanina_pattern'], 
            '-1lanina_flows_'
        )
        
        # NEW: Load baseline data
        baseline_data = data_loader.load_baseline_data(
            os.path.join(base_path, paths['baseline']), 
            filenames['baseline_pattern']
        )
        
        print(f"Loaded El Niño data for {len(elnino_data)} stations")
        print(f"Loaded La Niña data for {len(lanina_data)} stations")
        print(f"Loaded Baseline data for {len(baseline_data)} stations")
        
        print("\n=== STATION ID MATCHING DIAGNOSIS ===")
        print(f"Observed stations: {len(observed_data)}")
        print(f"El Niño stations: {len(elnino_data)}")
        print(f"La Niña stations: {len(lanina_data)}")
        print(f"Baseline stations: {len(baseline_data)}")

        # Get sample station IDs from each dataset
        obs_sample = list(observed_data.keys())[:5]
        elnino_sample = list(elnino_data.keys())[:5]
        lanina_sample = list(lanina_data.keys())[:5]
        baseline_sample = list(baseline_data.keys())[:5]

        print(f"\nSample Observed IDs: {obs_sample}")
        print(f"Sample El Niño IDs: {elnino_sample}")
        print(f"Sample La Niña IDs: {lanina_sample}")
        print(f"Sample Baseline IDs: {baseline_sample}")

        # Check for common stations across all datasets
        common_obs_elnino = set(observed_data.keys()) & set(elnino_data.keys())
        common_all_three = common_obs_elnino & set(lanina_data.keys())
        common_all_four = common_all_three & set(baseline_data.keys())

        print(f"\nMatching stations:")
        print(f"  Observed ∩ El Niño: {len(common_obs_elnino)}")
        print(f"  All three (obs, el niño, la niña): {len(common_all_three)}")
        print(f"  All four datasets: {len(common_all_four)}")

        # Check if IDs are just formatted differently
        print(f"\nChecking for formatting differences...")
        print("Do El Niño IDs contain extra characters?")
        for elnino_id in list(elnino_data.keys())[:3]:
            matches = [obs_id for obs_id in obs_sample if obs_id in elnino_id or elnino_id in obs_id]
            print(f"  El Niño '{elnino_id}' potential matches: {matches}")
        
        print("Do Baseline IDs contain extra characters?")
        for baseline_id in list(baseline_data.keys())[:3]:
            matches = [obs_id for obs_id in obs_sample if obs_id in baseline_id or baseline_id in obs_id]
            print(f"  Baseline '{baseline_id}' potential matches: {matches}")
        
        # STEP 2: Run analysis with baseline included
        synchrony_results = analysis.compare_enso_synchrony_scales(
            observed_data, elnino_data, lanina_data, coordinates,
            correlation_threshold=0.8,
            fraction_threshold=0.5,
            max_radius_km=4000,
            radius_step_km=50,
            baseline_data=baseline_data  # NEW: Pass baseline data
        )
        
        # STEP 3: Create visualizations (existing functions now handle 4 conditions automatically)
        print("\n--- Generating All Conditions Visualizations ---")
        
        # Maps (now shows up to 4 panels depending on available data)
        mapping.create_enso_synchrony_comparison_maps(
            synchrony_results, coordinates,
            save_path=os.path.join(output_dir, "all_conditions_synchrony_maps.png")
        )
        
        # Distributions (now shows up to 4 conditions depending on available data)
        mapping.create_enso_synchrony_distributions(
            synchrony_results,
            save_path=os.path.join(output_dir, "all_conditions_synchrony_distributions.png")
        )
        
        # STEP 4: Print summary for all available conditions
        print("\n=== ALL CONDITIONS SYNCHRONY SUMMARY ===")
        print(f"{'Condition':<12} {'Mean (km)':<10} {'Median (km)':<12} {'Std (km)':<10} {'Stations':<8}")
        print("-" * 65)
        
        for condition in ['observed', 'baseline', 'el_nino', 'la_nina']:
            if condition in synchrony_results:
                stats = synchrony_results[condition]['stats']
                print(f"{condition:<12} {stats['mean']:8.1f}   {stats['median']:10.1f}   "
                      f"{stats['std']:8.1f}   {stats['count']:6}")
        
        # STEP 5: Statistical comparisons
        print("\n=== CONDITION COMPARISONS ===")
        available_conditions = [cond for cond in ['observed', 'baseline', 'el_nino', 'la_nina'] 
                               if cond in synchrony_results]
        
        for i, cond1 in enumerate(available_conditions):
            for cond2 in available_conditions[i+1:]:
                mean1 = synchrony_results[cond1]['stats']['mean']
                mean2 = synchrony_results[cond2]['stats']['mean']
                diff = mean1 - mean2
                percent_diff = (diff / mean2) * 100 if mean2 != 0 else 0
                print(f"{cond1.title()} vs {cond2.title()}: {diff:+.1f} km ({percent_diff:+.1f}%)")
                
        # STEP 6: Correlation-Distance Analysis
        if RUN_CORRELATION_DISTANCE_ANALYSIS:
            print("\n--- Analyzing Correlation-Distance Relationships ---")
            correlation_distance_data = analysis.calculate_correlation_distance_relationships(
                synchrony_results, coordinates,
                distance_bins=np.arange(0, 3500, 150),  # 150 km bins
                sample_pairs=800  # Sample size per bin
            )
            
            # Create decay plots
            mapping.create_correlation_distance_plots(
                correlation_distance_data,
                save_path=os.path.join(output_dir, "all_conditions_correlation_distance_decay.png")
            )
        else:
            print("\n--- Skipping Correlation-Distance Analysis ---")
    
    else:
        print("\n--- Skipping All Conditions Synchrony Comparison ---")

if __name__ == "__main__":
    run_analysis_pipeline()