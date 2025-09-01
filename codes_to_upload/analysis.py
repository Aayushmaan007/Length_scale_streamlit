# analysis.py
import pandas as pd
import numpy as np
from scipy import stats
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error, mean_absolute_error

def _compute_metrics(observed, simulated_mean):
    """Helper to compute a standard set of performance metrics."""
    # Ensure no NaNs for calculation
    valid_mask = (~observed.isna()) & (~simulated_mean.isna())
    obs_clean = observed[valid_mask]
    sim_clean = simulated_mean[valid_mask]
    
    if len(obs_clean) < 2: # Need at least 2 points for correlation
        return {}

    return {
        'rmse': np.sqrt(mean_squared_error(obs_clean, sim_clean)),
        'mae': mean_absolute_error(obs_clean, sim_clean),
        'correlation': stats.pearsonr(obs_clean, sim_clean)[0],
        'percent_bias': ((sim_clean.sum() - obs_clean.sum()) / obs_clean.sum()) * 100 if obs_clean.sum() != 0 else 0,
    }

def calculate_all_station_stats(observed_data, elnino_data, lanina_data, baseline_data=None):
    """Calculates comparison statistics for all available stations."""
    all_stats = {}
    
    if baseline_data is not None:
        common_stations = (set(observed_data.keys()) & set(elnino_data.keys()) & 
                          set(lanina_data.keys()) & set(baseline_data.keys()))
    else:
        common_stations = set(observed_data.keys()) & set(elnino_data.keys()) & set(lanina_data.keys())

    for station_id in common_stations:
        obs = observed_data[station_id]
        elnino_sims = elnino_data[station_id]
        lanina_sims = lanina_data[station_id]
        
        # Align dates
        common_index = obs.index.intersection(elnino_sims.index).intersection(lanina_sims.index)
        
        if baseline_data is not None:
            baseline_sims = baseline_data[station_id]
            common_index = common_index.intersection(baseline_sims.index)
        
        if len(common_index) < 10:
            continue
            
        obs_aligned = obs.loc[common_index]
        elnino_aligned = elnino_sims.loc[common_index]
        lanina_aligned = lanina_sims.loc[common_index]
        
        # Calculate ensemble means
        elnino_mean = elnino_aligned.mean(axis=1)
        lanina_mean = lanina_aligned.mean(axis=1)

        # Compute metrics
        elnino_metrics = _compute_metrics(obs_aligned, elnino_mean)
        lanina_metrics = _compute_metrics(obs_aligned, lanina_mean)

        all_stats[station_id] = {
            'elnino': elnino_metrics,
            'lanina': lanina_metrics,
            'n_obs': len(obs_aligned.dropna())
        }
        
        # Add baseline metrics if provided
        if baseline_data is not None:
            baseline_aligned = baseline_sims.loc[common_index]
            baseline_mean = baseline_aligned.mean(axis=1)
            baseline_metrics = _compute_metrics(obs_aligned, baseline_mean)
            all_stats[station_id]['baseline'] = baseline_metrics
            
    return all_stats

def calculate_inter_station_correlation(observed_data):
    """
    Calculates the Spearman correlation matrix between all stations' streamflow.

    Args:
        observed_data (dict): A dictionary where keys are station IDs and 
                              values are pandas Series of observed streamflow.

    Returns:
        pandas.DataFrame: A DataFrame containing the Spearman correlation matrix.
    """
    # Combine all station data into a single DataFrame, aligning by date
    # The keys of the dictionary become the column names
    combined_df = pd.DataFrame(observed_data)

    # Drop any rows that have missing values to ensure a fair comparison
    combined_df.dropna(inplace=True)

    if combined_df.shape[0] < 2:
        print("Warning: Not enough overlapping data between stations to calculate correlation.")
        return pd.DataFrame()

    # Calculate the Spearman correlation matrix
    # 'spearman' is robust to non-linear relationships and outliers
    correlation_matrix = combined_df.corr(method='spearman')
    
    return correlation_matrix

def calculate_average_correlation(correlation_matrix):
    """
    Calculates the average correlation for each station against all others.

    Args:
        correlation_matrix (pd.DataFrame): The Spearman correlation matrix.

    Returns:
        pd.Series: A Series with station IDs as index and their average 
                   correlation as values.
    """
    # Create a copy to avoid modifying the original matrix
    matrix = correlation_matrix.copy()
    
    # Set the diagonal (self-correlation) to NaN so it's ignored in the mean calculation
    np.fill_diagonal(matrix.values, np.nan)
    
    # Calculate the mean for each column (station), ignoring the NaN values
    average_correlations = matrix.mean(axis=0)
    
    return average_correlations

def calculate_correlation_synchrony_scale(correlation_matrix, coordinates, 
                                        correlation_threshold=0.5, 
                                        fraction_threshold=0.5,
                                        max_radius_km=1000, 
                                        radius_step_km=10):
    """
    Calculate the correlation synchrony scale for each station.
    
    Similar to Berghuijs et al. (2019) flood synchrony scale, but using 
    correlation values instead of flood timing.
    
    Args:
        correlation_matrix (pd.DataFrame): Spearman correlation matrix between stations
        coordinates (dict): Dictionary with station IDs as keys and {'lat': lat, 'lon': lon} as values
        correlation_threshold (float): Minimum correlation to be considered "synchronous" (default 0.5)
        fraction_threshold (float): Minimum fraction of stations that must meet correlation threshold (default 0.5)
        max_radius_km (int): Maximum radius to search in km (default 1000)
        radius_step_km (int): Step size for radius increments in km (default 10)
    
    Returns:
        dict: Dictionary with station IDs as keys and their correlation synchrony scale (km) as values
    """
    
    def calculate_distance_km(coord1, coord2):
        """Calculate distance between two coordinates in kilometers."""
        return geodesic((coord1['lat'], coord1['lon']), 
                       (coord2['lat'], coord2['lon'])).kilometers
    
    def get_fraction_above_threshold(reference_station, radius_km):
        """
        Calculate fraction of stations within radius that have correlation 
        above threshold with reference station.
        """
        if reference_station not in coordinates:
            return 0.0
            
        ref_coord = coordinates[reference_station]
        stations_in_radius = []
        
        # Find all stations within the radius
        for station_id, coord in coordinates.items():
            if station_id == reference_station:
                continue  # Skip self
            if station_id not in correlation_matrix.index:
                continue  # Skip if not in correlation matrix
                
            distance = calculate_distance_km(ref_coord, coord)
            if distance <= radius_km:
                stations_in_radius.append(station_id)
        
        if len(stations_in_radius) == 0:
            return 0.0
        
        # Count stations with correlation above threshold
        correlations = correlation_matrix.loc[reference_station, stations_in_radius]
        above_threshold = (correlations.abs() >= correlation_threshold).sum()
        
        return above_threshold / len(stations_in_radius)
    
    synchrony_scales = {}
    
    # Calculate synchrony scale for each station
    for station_id in correlation_matrix.index:
        if station_id not in coordinates:
            continue
            
        max_radius = 0
        
        # Search through increasing radii
        for radius in range(radius_step_km, max_radius_km + radius_step_km, radius_step_km):
            fraction = get_fraction_above_threshold(station_id, radius)
            
            if fraction >= fraction_threshold:
                max_radius = radius
            else:
                # Stop when fraction drops below threshold
                break
        
        synchrony_scales[station_id] = max_radius
    
    return synchrony_scales

def calculate_mean_correlation_synchrony_scale(synchrony_scales):
    """
    Calculate statistics for correlation synchrony scales.
    
    Args:
        synchrony_scales (dict): Output from calculate_correlation_synchrony_scale
        
    Returns:
        dict: Statistics including mean, median, min, max
    """
    scales = list(synchrony_scales.values())
    scales = [s for s in scales if s > 0]  # Filter out zero values
    
    if len(scales) == 0:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0, 'count': 0}
    
    return {
        'mean': np.mean(scales),
        'median': np.median(scales),
        'min': np.min(scales),
        'max': np.max(scales),
        'std': np.std(scales),
        'count': len(scales)
    }

def calculate_enso_correlation_matrices(elnino_data, lanina_data, baseline_data=None, method='pooled'):
    """
    Calculate correlation matrices using pooled or ensemble mean approach.
    
    Args:
        elnino_data (dict): El Niño simulation data with multiple realizations
        lanina_data (dict): La Niña simulation data with multiple realizations
        baseline_data (dict, optional): Baseline simulation data with multiple realizations
        method (str): 'pooled' concatenates all realizations, 'ensemble_mean' averages them
    
    Returns:
        tuple: (elnino_correlation_matrix, lanina_correlation_matrix) or
               (elnino_correlation_matrix, lanina_correlation_matrix, baseline_correlation_matrix)
    """
    
    def process_simulation_data(sim_data, method):
        """Process simulation data using specified method."""
        processed_data = {}
        
        for station_id, realizations in sim_data.items():
            if method == 'pooled':
                # Concatenate all realizations into one long time series
                # realizations is a DataFrame with columns r1, r2, ..., r10
                all_values = []
                
                for col in realizations.columns:  # r1, r2, ..., r10
                    realization_data = realizations[col].dropna()  # Remove NaN
                    all_values.extend(realization_data.values)
                
                # Convert to pandas Series with sequential index
                processed_data[station_id] = pd.Series(all_values)
                
            elif method == 'ensemble_mean':
                # Calculate ensemble mean across realizations (original method)
                processed_data[station_id] = realizations.mean(axis=1)
        
        return processed_data
    
    # Process simulation datasets
    elnino_processed = process_simulation_data(elnino_data, method)
    lanina_processed = process_simulation_data(lanina_data, method)
    
    # Calculate correlation matrices using existing function
    elnino_corr_matrix = calculate_inter_station_correlation(elnino_processed)
    lanina_corr_matrix = calculate_inter_station_correlation(lanina_processed)
    
    print(f"El Niño correlation matrix: {elnino_corr_matrix.shape}")
    print(f"La Niña correlation matrix: {lanina_corr_matrix.shape}")
    
    if method == 'pooled':
        # Show sample sizes for pooled approach
        sample_station = list(elnino_processed.keys())[0]
        elnino_sample_size = len(elnino_processed[sample_station])
        lanina_sample_size = len(lanina_processed[sample_station])
        print(f"Sample sizes (pooled): El Niño = {elnino_sample_size}, La Niña = {lanina_sample_size}")
    
    if baseline_data is not None:
        baseline_processed = process_simulation_data(baseline_data, method)
        baseline_corr_matrix = calculate_inter_station_correlation(baseline_processed)
        print(f"Baseline correlation matrix: {baseline_corr_matrix.shape}")
        
        if method == 'pooled':
            baseline_sample_size = len(baseline_processed[sample_station])
            print(f"Baseline sample size (pooled): {baseline_sample_size}")
        
        return elnino_corr_matrix, lanina_corr_matrix, baseline_corr_matrix
    
    return elnino_corr_matrix, lanina_corr_matrix

def compare_data_sizes(elnino_data, lanina_data, baseline_data=None):
    """
    Compare data sizes between ensemble mean and pooled approaches.
    Useful for understanding what pooling does to your sample sizes.
    """
    print("=== DATA SIZE COMPARISON ===")
    
    # Get sample station
    sample_station = list(elnino_data.keys())[0]
    sample_elnino = elnino_data[sample_station]
    
    print(f"Sample station: {sample_station}")
    print(f"Original data shape: {sample_elnino.shape} (time_steps x realizations)")
    
    # Ensemble mean approach
    ensemble_mean_size = len(sample_elnino.mean(axis=1).dropna())
    print(f"\nEnsemble mean approach:")
    print(f"  Sample size per station: {ensemble_mean_size} time steps")
    
    # Pooled approach
    pooled_values = []
    for col in sample_elnino.columns:
        realization_data = sample_elnino[col].dropna()
        pooled_values.extend(realization_data.values)
    
    pooled_size = len(pooled_values)
    print(f"\nPooled approach:")
    print(f"  Sample size per station: {pooled_size} values")
    print(f"  Increase factor: {pooled_size / ensemble_mean_size:.1f}x larger")
    
    # Show for all datasets if available
    datasets = {'El Niño': elnino_data, 'La Niña': lanina_data}
    if baseline_data is not None:
        datasets['Baseline'] = baseline_data
    
    print(f"\nSample sizes across all datasets:")
    for name, data in datasets.items():
        sample_data = data[sample_station]
        pooled_values = []
        for col in sample_data.columns:
            realization_data = sample_data[col].dropna()
            pooled_values.extend(realization_data.values)
        print(f"  {name}: {len(pooled_values)} values")

def compare_enso_synchrony_scales(observed_data, elnino_data, lanina_data, coordinates,
                                correlation_threshold=0.4, fraction_threshold=0.5,
                                max_radius_km=3000, radius_step_km=50, baseline_data=None,
                                simulation_method='pooled'):
    """
    Calculate and compare synchrony scales with chosen simulation method.
    
    Args:
        observed_data (dict): Observed streamflow data
        elnino_data (dict): El Niño simulation data
        lanina_data (dict): La Niña simulation data
        coordinates (dict): Station coordinates
        correlation_threshold (float): Minimum correlation for synchrony
        fraction_threshold (float): Minimum fraction of stations needed
        max_radius_km (int): Maximum search radius
        radius_step_km (int): Radius increment
        baseline_data (dict, optional): Baseline simulation data
        simulation_method (str): 'pooled' or 'ensemble_mean'
    
    Returns:
        dict: Dictionary with synchrony scales and statistics for each condition
    """
    
    method_name = "POOLED METHOD" if simulation_method == 'pooled' else "ENSEMBLE MEAN METHOD"
    condition_name = f"ENSO SYNCHRONY SCALE COMPARISON ({method_name})"
    if baseline_data is not None:
        condition_name = f"ALL CONDITIONS SYNCHRONY SCALE COMPARISON ({method_name})"
    
    print(f"=== {condition_name} ===")
    
    # Calculate correlation matrices
    print(f"\n1. Calculating correlation matrices using {simulation_method} method...")
    observed_corr_matrix = calculate_inter_station_correlation(observed_data)
    
    if baseline_data is not None:
        elnino_corr_matrix, lanina_corr_matrix, baseline_corr_matrix = calculate_enso_correlation_matrices(
            elnino_data, lanina_data, baseline_data, method=simulation_method
        )
    else:
        elnino_corr_matrix, lanina_corr_matrix = calculate_enso_correlation_matrices(
            elnino_data, lanina_data, method=simulation_method
        )
    
    # Calculate synchrony scales for each condition
    conditions = {
        'observed': observed_corr_matrix,
        'el_nino': elnino_corr_matrix, 
        'la_nina': lanina_corr_matrix
    }
    
    # Add baseline if provided
    if baseline_data is not None:
        conditions['baseline'] = baseline_corr_matrix
    
    synchrony_results = {}
    
    for condition, corr_matrix in conditions.items():
        print(f"\n2. Calculating {condition} synchrony scales...")
        
        synchrony_scales = calculate_correlation_synchrony_scale(
            corr_matrix, coordinates,
            correlation_threshold=correlation_threshold,
            fraction_threshold=fraction_threshold,
            max_radius_km=max_radius_km,
            radius_step_km=radius_step_km
        )
        
        # Calculate statistics
        stats = calculate_mean_correlation_synchrony_scale(synchrony_scales)
        
        synchrony_results[condition] = {
            'scales': synchrony_scales,
            'stats': stats,
            'correlation_matrix': corr_matrix
        }
        
        print(f"  {condition.upper()} Results:")
        print(f"    Mean: {stats['mean']:.1f} km")
        print(f"    Median: {stats['median']:.1f} km")
        print(f"    Std Dev: {stats['std']:.1f} km") 
        print(f"    Valid stations: {stats['count']}")
    
    return synchrony_results

def calculate_correlation_distance_relationships(synchrony_results, coordinates, 
                                               distance_bins=np.arange(0, 3500, 100),
                                               sample_pairs=1000):
    """
    Calculate how correlation decays with distance for each condition.
    Works with both ensemble mean and pooled correlation matrices.
    
    Args:
        synchrony_results (dict): Results from compare_enso_synchrony_scales()
        coordinates (dict): Station coordinates
        distance_bins (array): Distance bins in km
        sample_pairs (int): Number of random station pairs to sample per distance bin
        
    Returns:
        dict: Distance-correlation data for each condition
    """
    from geopy.distance import geodesic
    import random
    
    correlation_data = {}
    
    # Check all possible conditions
    possible_conditions = ['observed', 'el_nino', 'la_nina', 'baseline']
    
    for condition in possible_conditions:
        if condition not in synchrony_results:
            continue
            
        corr_matrix = synchrony_results[condition]['correlation_matrix']
        stations = list(corr_matrix.index)
        
        # Filter stations that have coordinates
        valid_stations = [s for s in stations if s in coordinates]
        
        print(f"Processing {condition}: {len(valid_stations)} stations")
        
        # Store distance-correlation pairs
        distances = []
        correlations = []
        
        # Sample station pairs and calculate distances/correlations
        total_samples = sample_pairs * len(distance_bins)
        print(f"  Sampling {total_samples} station pairs...")
        
        for _ in range(total_samples):
            # Random pair
            if len(valid_stations) < 2:
                continue
            station1, station2 = random.sample(valid_stations, 2)
            
            # Calculate distance
            coord1 = coordinates[station1]
            coord2 = coordinates[station2]
            dist = geodesic((coord1['lat'], coord1['lon']), 
                           (coord2['lat'], coord2['lon'])).kilometers
            
            # Get correlation (use absolute value to focus on strength)
            corr = abs(corr_matrix.loc[station1, station2])
            
            distances.append(dist)
            correlations.append(corr)
        
        # Bin the data
        distance_centers = distance_bins[:-1] + np.diff(distance_bins) / 2
        binned_correlations = []
        binned_std = []
        
        for i in range(len(distance_bins) - 1):
            # Find correlations in this distance bin
            mask = (np.array(distances) >= distance_bins[i]) & (np.array(distances) < distance_bins[i + 1])
            bin_corrs = np.array(correlations)[mask]
            
            if len(bin_corrs) > 5:  # Need minimum data points
                binned_correlations.append(np.mean(bin_corrs))
                binned_std.append(np.std(bin_corrs))
            else:
                binned_correlations.append(np.nan)
                binned_std.append(np.nan)
        
        correlation_data[condition] = {
            'distances': distance_centers,
            'correlations': np.array(binned_correlations),
            'std': np.array(binned_std),
            'raw_distances': distances,
            'raw_correlations': correlations
        }
        
        print(f"  Completed distance analysis for {condition}")
    
    return correlation_data