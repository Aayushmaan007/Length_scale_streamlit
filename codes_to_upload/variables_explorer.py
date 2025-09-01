# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 08:47:03 2025

@author: aayus
"""

# filter_gages_basin_characteristics.py

import pandas as pd
import os
import glob
from pathlib import Path

def explore_gages_excel_structure(excel_path):
    """
    Explore the structure of the GAGES-II Excel file to understand
    its sheets, columns, and data format.
    """
    print("=== EXPLORING GAGES-II EXCEL FILE STRUCTURE ===")
    
    # Read all sheet names
    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names
    
    print(f"\nFound {len(sheet_names)} sheets in the Excel file:")
    for i, sheet in enumerate(sheet_names):
        print(f"  {i+1}. {sheet}")
    
    # Examine each sheet
    sheet_info = {}
    for sheet_name in sheet_names:
        print(f"\n--- SHEET: {sheet_name} ---")
        
        try:
            # Read first few rows to understand structure
            df_sample = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=5)
            print(f"Shape: {df_sample.shape}")
            print(f"Columns: {list(df_sample.columns)[:10]}...")  # First 10 columns
            
            # Check if STAID column exists (station ID)
            if 'STAID' in df_sample.columns:
                print(f"STAID column found! Sample values: {df_sample['STAID'].head().tolist()}")
            
            sheet_info[sheet_name] = {
                'shape': df_sample.shape,
                'columns': list(df_sample.columns),
                'has_staid': 'STAID' in df_sample.columns,
                'sample_data': df_sample.head(2)
            }
            
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
            sheet_info[sheet_name] = {'error': str(e)}
    
    return sheet_info

def get_streamflow_station_ids(streamflow_data_path, observed_pattern):
    """
    Extract station IDs from the streamflow data files.
    """
    print("=== EXTRACTING STATION IDs FROM STREAMFLOW DATA ===")
    
    file_paths = glob.glob(os.path.join(streamflow_data_path, observed_pattern))
    station_ids = []
    
    print(f"Found {len(file_paths)} streamflow files")
    
    for file_path in file_paths:
        # Extract station ID from filename (assuming pattern: filtered_site_*_daily_discharge.txt)
        filename = os.path.basename(file_path)
        try:
            # Split by '_' and get the station ID part
            parts = filename.split('_')
            if len(parts) >= 3:
                station_id = parts[2]  # Should be the station ID
                station_ids.append(station_id)
        except Exception as e:
            print(f"Error extracting station ID from {filename}: {e}")
    
    # Remove duplicates and sort
    station_ids = sorted(list(set(station_ids)))
    
    print(f"Extracted {len(station_ids)} unique station IDs")
    print(f"Sample station IDs: {station_ids[:10]}")
    
    return station_ids

def normalize_station_id(station_id):
    """
    Normalize station ID to ensure consistent formatting for matching.
    """
    # Remove any non-numeric characters and ensure 8 digits with leading zeros
    numeric_id = ''.join(filter(str.isdigit, str(station_id)))
    return numeric_id.zfill(8)

def filter_gages_data_by_stations(excel_path, station_ids, output_dir):
    """
    Filter GAGES-II basin characteristics data to include only the specified station IDs.
    """
    print("=== FILTERING GAGES-II DATA BY STATION IDs ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize station IDs for matching
    normalized_streamflow_ids = set([normalize_station_id(sid) for sid in station_ids])
    print(f"Normalized {len(normalized_streamflow_ids)} station IDs for matching")
    
    excel_file = pd.ExcelFile(excel_path)
    filtered_sheets = {}
    
    for sheet_name in excel_file.sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        
        try:
            # Read the full sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"  Original shape: {df.shape}")
            
            # Check if this sheet has station IDs
            if 'STAID' not in df.columns:
                print(f"  No STAID column found, skipping...")
                continue
            
            # Normalize station IDs in the sheet for matching
            df['STAID_NORMALIZED'] = df['STAID'].apply(normalize_station_id)
            
            # Filter to only include our stations
            filtered_df = df[df['STAID_NORMALIZED'].isin(normalized_streamflow_ids)].copy()
            
            # Remove the temporary normalized column
            filtered_df = filtered_df.drop('STAID_NORMALIZED', axis=1)
            
            print(f"  Filtered shape: {filtered_df.shape}")
            print(f"  Retained {filtered_df.shape[0]} out of {df.shape[0]} stations")
            
            if filtered_df.shape[0] > 0:
                filtered_sheets[sheet_name] = filtered_df
                
                # Save individual sheet as CSV
                output_path = os.path.join(output_dir, f"filtered_{sheet_name}.csv")
                filtered_df.to_csv(output_path, index=False)
                print(f"  Saved to: {output_path}")
            else:
                print(f"  No matching stations found in this sheet")
                
        except Exception as e:
            print(f"  Error processing sheet {sheet_name}: {e}")
    
    # Save combined Excel file with all filtered sheets
    if filtered_sheets:
        combined_excel_path = os.path.join(output_dir, "filtered_gages_basin_characteristics.xlsx")
        with pd.ExcelWriter(combined_excel_path, engine='openpyxl') as writer:
            for sheet_name, df in filtered_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nSaved combined Excel file to: {combined_excel_path}")
        
        # Print summary
        print(f"\n=== FILTERING SUMMARY ===")
        print(f"Total streamflow station IDs: {len(station_ids)}")
        print(f"Sheets processed: {len(filtered_sheets)}")
        
        for sheet_name, df in filtered_sheets.items():
            print(f"  {sheet_name}: {df.shape[0]} stations, {df.shape[1]} variables")
    
    return filtered_sheets

def diagnose_station_id_matching(excel_path, station_ids):
    """
    Diagnose potential station ID matching issues.
    """
    print("=== DIAGNOSING STATION ID MATCHING ===")
    
    # Get sample of station IDs from Excel file
    excel_file = pd.ExcelFile(excel_path)
    
    for sheet_name in excel_file.sheet_names:
        try:
            df_sample = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=10)
            if 'STAID' in df_sample.columns:
                excel_station_ids = df_sample['STAID'].tolist()
                print(f"\nSheet '{sheet_name}' station ID samples:")
                print(f"  Excel: {excel_station_ids}")
                print(f"  Streamflow: {station_ids[:5]}")
                
                # Check formatting differences
                print(f"  Excel ID lengths: {[len(str(sid)) for sid in excel_station_ids[:5]]}")
                print(f"  Streamflow ID lengths: {[len(str(sid)) for sid in station_ids[:5]]}")
                
                # Test normalization
                normalized_excel = [normalize_station_id(sid) for sid in excel_station_ids[:5]]
                normalized_stream = [normalize_station_id(sid) for sid in station_ids[:5]]
                print(f"  Normalized Excel: {normalized_excel}")
                print(f"  Normalized Stream: {normalized_stream}")
                
                break
        except:
            continue

def main():
    """Main function to filter GAGES-II basin characteristics data."""
    
    # Configuration
    config = {
        'gages_excel_path': r"C:\Research_Data\basin_stuff\gagesII_sept30_2011_conterm.xlsx",
        'streamflow_data_path': r"C:\Research_Data\observed_streamflow\data1960",
        'observed_pattern': "filtered_site_*_daily_discharge.txt",
        'output_dir': r"C:\Research_Data\basin_stuff\filtered_basin_characteristics"
    }
    
    # Check if files exist
    if not os.path.exists(config['gages_excel_path']):
        print(f"ERROR: GAGES Excel file not found at: {config['gages_excel_path']}")
        return
    
    if not os.path.exists(config['streamflow_data_path']):
        print(f"ERROR: Streamflow data directory not found at: {config['streamflow_data_path']}")
        return
    
    # Step 1: Explore Excel file structure
    print("Step 1: Exploring Excel file structure...")
    sheet_info = explore_gages_excel_structure(config['gages_excel_path'])
    
    # Step 2: Extract station IDs from streamflow data
    print("\nStep 2: Extracting station IDs from streamflow data...")
    station_ids = get_streamflow_station_ids(
        config['streamflow_data_path'], 
        config['observed_pattern']
    )
    
    if not station_ids:
        print("ERROR: No station IDs found in streamflow data")
        return
    
    # Step 3: Diagnose potential matching issues
    print("\nStep 3: Diagnosing station ID matching...")
    diagnose_station_id_matching(config['gages_excel_path'], station_ids)
    
    # Step 4: Filter the GAGES data
    print("\nStep 4: Filtering GAGES-II data...")
    filtered_data = filter_gages_data_by_stations(
        config['gages_excel_path'],
        station_ids,
        config['output_dir']
    )
    
    print("\n=== FILTERING COMPLETE ===")
    print(f"Filtered data saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()