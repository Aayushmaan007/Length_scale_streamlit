# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 20:57:18 2025

@author: aayus
"""

# data_loader.py
import os
import glob
import pandas as pd

def normalize_station_id(station_id):
    """
    Normalize station ID to 8-digit format with leading zeros.
    """
    # Remove any non-numeric characters and ensure 8 digits with leading zeros
    numeric_id = ''.join(filter(str.isdigit, str(station_id)))
    return numeric_id.zfill(8)

def load_coordinates(coord_path):
    """Loads station coordinates from a text file."""
    coordinates = {}
    try:
        with open(coord_path, 'r') as f:
            for line in f:
                if ':' in line:
                    station_id, coords_str = line.strip().split(': ')
                    lat, lon = map(float, coords_str.strip('()').split(', '))
                    coordinates[station_id] = {'lat': lat, 'lon': lon}
        print(f"Loaded coordinates for {len(coordinates)} stations.")
    except Exception as e:
        print(f"Error loading coordinates: {e}")
    return coordinates

def _create_date_from_components(df):
    """Helper to create a DatetimeIndex."""
    required_cols = ['YYYY', 'MM', 'DD']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one or more date columns: {required_cols}")
    df['Date'] = pd.to_datetime(df[required_cols])
    df.set_index('Date', inplace=True)
    return df

def load_observed_data(path, pattern):
    """Loads all observed streamflow data files."""
    observed_data = {}
    file_paths = glob.glob(os.path.join(path, pattern))
    for file_path in file_paths:
        station_id = os.path.basename(file_path).split('_')[2]
        try:
            df = pd.read_csv(file_path, sep='\t', parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            observed_data[station_id] = df['Flow'] # Return a Series for simplicity
        except Exception as e:
            print(f"Error loading observed data for {station_id}: {e}")
    return observed_data

def load_simulation_data(path, pattern, prefix_to_remove):
    """Generic function to load simulation data (El Niño or La Niña)."""
    simulation_data = {}
    file_paths = glob.glob(os.path.join(path, pattern))
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        station_id = filename.replace(prefix_to_remove, '').replace('.txt', '')
        
        station_id = normalize_station_id(station_id)
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            if 'Date' not in df.columns:
                df = _create_date_from_components(df)
            else:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            realization_cols = [f'r{i}' for i in range(1, 11) if f'r{i}' in df.columns]
            if realization_cols:
                simulation_data[station_id] = df[realization_cols].dropna(how='all')
        except Exception as e:
            print(f"Error loading simulation for {station_id}: {e}")
    return simulation_data

def load_baseline_data(path, pattern):
    """
    Load baseline streamflow simulation data.
    
    Expected format:
    YYYY	MM	DD	timestamp	Qobs	r1	r2	r3	r4	r5	r6	r7	r8	r9	r10
    
    Args:
        path: Directory containing baseline simulation files
        pattern: File pattern (e.g., "simulated_*.txt")
    
    Returns:
        dict: Dictionary with station IDs as keys and DataFrames with realizations as values
    """
    baseline_data = {}
    file_paths = glob.glob(os.path.join(path, pattern))
    
    print(f"Loading baseline data from {len(file_paths)} files...")
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        # Extract station ID from filename (e.g., "simulated_01013500.txt" -> "01013500")
        if filename.startswith('simulated_'):
            station_id = filename.replace('simulated_', '').replace('.txt', '')
        else:
            # Fallback: try to extract numeric part
            station_id = ''.join(filter(str.isdigit, filename))
        
        station_id = normalize_station_id(station_id)
        
        try:
            # Read the baseline simulation file
            df = pd.read_csv(file_path, sep='\t')
            
            # Create date index
            if 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
            else:
                # Use YYYY, MM, DD columns
                df['Date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']])
            
            df.set_index('Date', inplace=True)
            
            # Get realization columns (r1, r2, ..., r10)
            realization_cols = [f'r{i}' for i in range(1, 11) if f'r{i}' in df.columns]
            
            if realization_cols:
                baseline_data[station_id] = df[realization_cols].dropna(how='all')
                print(f"Loaded baseline data for station {station_id}: {len(realization_cols)} realizations")
            else:
                print(f"Warning: No realization columns found for {station_id}")
                
        except Exception as e:
            print(f"Error loading baseline data for {station_id}: {e}")
            continue
    
    print(f"Successfully loaded baseline data for {len(baseline_data)} stations")
    return baseline_data

