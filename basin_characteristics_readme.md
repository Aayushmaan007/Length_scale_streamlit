

# Basin Characteristics Data Usage Guide

## Overview
This guide explains how to use the filtered GAGES-II basin characteristics data that was created for your streamflow synchrony analysis. The data has been filtered from the original 9,067 stations to match your 450 streamflow stations.

## Data Location and Structure

### File Locations
```
C:\Research_Data\basin_stuff\filtered_basin_characteristics\
├── filtered_gages_basin_characteristics.xlsx  # Combined Excel file
├── filtered_BasinID.csv                       # Station identifiers and basic info
├── filtered_Climate.csv                       # Climate variables
├── filtered_Topo.csv                         # Topographic variables
├── filtered_Hydro.csv                        # Hydrologic variables
├── filtered_Soils.csv                        # Soil characteristics
├── filtered_LC06_Basin.csv                   # Land cover (2006)
├── filtered_Geology.csv                      # Geological characteristics
├── filtered_HydroMod_Dams.csv               # Dam modifications
├── filtered_Pop_Infrastr.csv                # Population and infrastructure
└── ... (additional sheets as individual CSV files)
```

### Data Coverage
- **Original Dataset**: 9,067 GAGES-II stations
- **Filtered Dataset**: 450 stations matching your streamflow analysis
- **Time Period**: Various (see individual variable descriptions)
- **Geographic Coverage**: Continental United States

## Key Data Sheets and Variables

### 1. BasinID (filtered_BasinID.csv)
**Purpose**: Station identifiers and basic watershed information
**Key Variables**:
- `STAID`: Station ID (8-digit format)
- `STANAME`: Station name
- `DRAIN_SQKM`: Watershed drainage area (km²)
- `LAT_GAGE`, `LNG_GAGE`: Station coordinates
- `STATE`: State abbreviation
- `HCDN-2009`: Whether station is in Hydro-Climatic Data Network

### 2. Climate (filtered_Climate.csv)
**Purpose**: Long-term climate characteristics
**Key Variables**:
- `PPTAVG_BASIN`: Mean annual precipitation (cm)
- `T_AVG_BASIN`: Mean annual temperature (°C)
- `PET`: Potential evapotranspiration (mm/yr)
- `SNOW_PCT_PRECIP`: Snow percentage of precipitation (%)
- `PRECIP_SEAS_IND`: Precipitation seasonality index (0-1)
- `WD_BASIN`: Wet days per year
- `WDMAX_BASIN`, `WDMIN_BASIN`: Max/min monthly wet days

### 3. Topography (filtered_Topo.csv)
**Purpose**: Terrain characteristics
**Key Variables**:
- `ELEV_MEAN_M_BASIN`: Mean watershed elevation (m)
- `ELEV_MAX_M_BASIN`, `ELEV_MIN_M_BASIN`: Max/min elevation (m)
- `SLOPE_PCT`: Mean watershed slope (%)
- `ASPECT_NORTHNESS`, `ASPECT_EASTNESS`: Aspect characteristics

### 4. Hydrology (filtered_Hydro.csv)
**Purpose**: Natural hydrologic characteristics
**Key Variables**:
- `STREAMS_KM_SQ_KM`: Stream density (km/km²)
- `BFI_AVE`: Base flow index (%)
- `RUNAVE7100`: Average annual runoff (mm/yr)
- `TOPWET`: Topographic wetness index

### 5. Land Cover (filtered_LC06_Basin.csv)
**Purpose**: 2006 land cover percentages
**Key Variables**:
- `FORESTNLCD06`: Forest percentage (%)
- `DEVNLCD06`: Developed/urban percentage (%)
- `PLANTNLCD06`: Agriculture percentage (%)
- `WATERNLCD06`: Open water percentage (%)

### 6. Soils (filtered_Soils.csv)
**Purpose**: Soil characteristics
**Key Variables**:
- `HGA`, `HGB`, `HGC`, `HGD`: Hydrologic soil groups A-D (%)
- `PERMAVE`: Average permeability (inches/hour)
- `CLAYAVE`, `SILTAVE`, `SANDAVE`: Soil texture percentages

## How to Load and Use the Data

### Python Examples

#### 1. Basic Data Loading
```python
import pandas as pd
import os

# Set base directory
basin_char_dir = r"C:\Research_Data\basin_stuff\filtered_basin_characteristics"

# Load specific data sheets
climate_df = pd.read_csv(os.path.join(basin_char_dir, "filtered_Climate.csv"))
topo_df = pd.read_csv(os.path.join(basin_char_dir, "filtered_Topo.csv"))
basin_id_df = pd.read_csv(os.path.join(basin_char_dir, "filtered_BasinID.csv"))

# Load all sheets from Excel file (alternative approach)
excel_file = os.path.join(basin_char_dir, "filtered_gages_basin_characteristics.xlsx")
all_sheets = pd.read_excel(excel_file, sheet_name=None)  # Returns dict of DataFrames
```

#### 2. Merge with Your Synchrony Results
```python
# Assuming you have synchrony_scales from your analysis
def merge_with_synchrony(basin_df, synchrony_scales):
    """Merge basin characteristics with synchrony scales."""
    
    # Create synchrony dataframe
    synchrony_df = pd.DataFrame({
        'STAID': list(synchrony_scales.keys()),
        'synchrony_scale_km': list(synchrony_scales.values())
    })
    
    # Normalize STAID format
    synchrony_df['STAID'] = synchrony_df['STAID'].astype(str).str.zfill(8)
    basin_df['STAID'] = basin_df['STAID'].astype(str).str.zfill(8)
    
    # Merge
    merged_df = pd.merge(basin_df, synchrony_df, on='STAID', how='inner')
    
    # Filter out zero synchrony scales
    merged_df = merged_df[merged_df['synchrony_scale_km'] > 0]
    
    return merged_df

# Example usage
climate_with_sync = merge_with_synchrony(climate_df, synchrony_scales)
```

#### 3. Multi-Sheet Analysis
```python
def analyze_multiple_characteristics(synchrony_scales, basin_char_dir, variable_dict):
    """
    Analyze relationships between multiple basin characteristics and synchrony.
    
    variable_dict: {'sheet_name': ['var1', 'var2', ...]}
    """
    results = {}
    
    for sheet_name, variables in variable_dict.items():
        # Load data
        df = pd.read_csv(os.path.join(basin_char_dir, f"filtered_{sheet_name}.csv"))
        
        # Merge with synchrony
        merged_df = merge_with_synchrony(df, synchrony_scales)
        
        # Calculate correlations
        correlations = {}
        for var in variables:
            if var in merged_df.columns:
                valid_data = merged_df[[var, 'synchrony_scale_km']].dropna()
                if len(valid_data) > 10:
                    from scipy import stats
                    corr, p_val = stats.pearsonr(valid_data[var], valid_data['synchrony_scale_km'])
                    correlations[var] = {'correlation': corr, 'p_value': p_val}
        
        results[sheet_name] = correlations
    
    return results

# Example usage
variables_to_analyze = {
    'Climate': ['PPTAVG_BASIN', 'T_AVG_BASIN', 'PET'],
    'Topo': ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT'],
    'Hydro': ['BFI_AVE', 'STREAMS_KM_SQ_KM'],
    'LC06_Basin': ['FORESTNLCD06', 'DEVNLCD06']
}

multi_results = analyze_multiple_characteristics(synchrony_scales, basin_char_dir, variables_to_analyze)
```

## Common Analysis Patterns

### 1. Correlation Analysis
```python
def basin_synchrony_correlations(sheet_name, variables, synchrony_scales):
    """Calculate correlations between basin characteristics and synchrony."""
    
    # Load data
    df = pd.read_csv(f"filtered_{sheet_name}.csv")
    merged_df = merge_with_synchrony(df, synchrony_scales)
    
    results = []
    for var in variables:
        valid_data = merged_df[[var, 'synchrony_scale_km']].dropna()
        if len(valid_data) > 10:
            corr, p_val = stats.pearsonr(valid_data[var], valid_data['synchrony_scale_km'])
            results.append({
                'variable': var,
                'correlation': corr,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'n_points': len(valid_data)
            })
    
    return pd.DataFrame(results).sort_values('correlation', key=abs, ascending=False)
```

### 2. Categorical Analysis
```python
def analyze_by_categories(synchrony_scales, category_var, category_sheet='BasinID'):
    """Analyze synchrony by categorical variables (e.g., state, HUC region)."""
    
    # Load categorical data
    cat_df = pd.read_csv(f"filtered_{category_sheet}.csv")
    merged_df = merge_with_synchrony(cat_df, synchrony_scales)
    
    # Group by category
    category_stats = merged_df.groupby(category_var)['synchrony_scale_km'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    return category_stats

# Example: Analyze by state
state_analysis = analyze_by_categories(synchrony_scales, 'STATE')
```

### 3. Multiple Regression Analysis
```python
def multiple_regression_analysis(synchrony_scales, predictor_vars_dict):
    """
    Perform multiple regression with basin characteristics as predictors.
    
    predictor_vars_dict: {'sheet_name': ['var1', 'var2', ...]}
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Collect all predictors
    all_predictors = pd.DataFrame()
    
    for sheet, vars_list in predictor_vars_dict.items():
        df = pd.read_csv(f"filtered_{sheet}.csv")
        df['STAID'] = df['STAID'].astype(str).str.zfill(8)
        
        if all_predictors.empty:
            all_predictors = df[['STAID'] + vars_list]
        else:
            all_predictors = pd.merge(all_predictors, df[['STAID'] + vars_list], on='STAID')
    
    # Merge with synchrony
    merged_df = merge_with_synchrony(all_predictors, synchrony_scales)
    
    # Prepare data
    predictor_cols = [col for col in merged_df.columns if col not in ['STAID', 'synchrony_scale_km']]
    X = merged_df[predictor_cols].dropna()
    y = merged_df.loc[X.index, 'synchrony_scale_km']
    
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Results
    results = pd.DataFrame({
        'variable': predictor_cols,
        'coefficient': model.coef_,
        'abs_coefficient': abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    results['r_squared'] = model.score(X_scaled, y)
    
    return results, model, scaler
```

## Integration with Existing Analysis

### Update Your main.py
```python
# Add this function to your main.py or create a separate module
def load_basin_characteristics(basin_char_dir, sheet_names=None):
    """
    Load multiple basin characteristic sheets.
    
    Args:
        basin_char_dir: Path to filtered basin characteristics
        sheet_names: List of sheet names to load (None = load all)
    
    Returns:
        Dictionary of DataFrames
    """
    if sheet_names is None:
        # Load all available sheets
        import glob
        csv_files = glob.glob(os.path.join(basin_char_dir, "filtered_*.csv"))
        sheet_names = [os.path.basename(f).replace('filtered_', '').replace('.csv', '') 
                      for f in csv_files]
    
    basin_data = {}
    for sheet in sheet_names:
        file_path = os.path.join(basin_char_dir, f"filtered_{sheet}.csv")
        if os.path.exists(file_path):
            basin_data[sheet] = pd.read_csv(file_path)
            # Normalize STAID
            basin_data[sheet]['STAID'] = basin_data[sheet]['STAID'].astype(str).str.zfill(8)
    
    return basin_data

# Usage in your main analysis
if RUN_BASIN_CHARACTERISTICS_ANALYSIS:
    basin_char_dir = r"C:\Research_Data\basin_stuff\filtered_basin_characteristics"
    basin_data = load_basin_characteristics(basin_char_dir)
    
    # Now you can use basin_data['Climate'], basin_data['Topo'], etc.
```

## Data Quality and Limitations

### Missing Values
- Different variables may have missing values for some stations
- Always check for and handle NaN values before analysis
- Use `.dropna()` or `.fillna()` as appropriate

### Temporal Coverage
- Climate variables: Typically 1971-2000 or 1961-1990
- Land cover: 2006 data (NLCD 2006)
- Infrastructure: Various years (see variable descriptions)
- Your streamflow data: 1960+ (based on your filtering)

### Spatial Coverage
- Limited to continental United States
- No Alaska, Hawaii, or Puerto Rico stations in your filtered dataset
- Coordinate system: NAD83 decimal degrees

## Troubleshooting

### Common Issues
1. **STAID Formatting**: Always normalize to 8-digit format with leading zeros
2. **Missing Files**: Check file paths and ensure filtering script ran successfully
3. **Memory Issues**: For large analyses, process one sheet at a time
4. **Merge Problems**: Verify STAID formats match between datasets

### Verification Steps
```python
# Verify data integrity
def verify_basin_data(basin_char_dir, expected_stations=450):
    """Verify the filtered basin characteristics data."""
    
    import glob
    
    csv_files = glob.glob(os.path.join(basin_char_dir, "filtered_*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        sheet_name = os.path.basename(file_path).replace('filtered_', '').replace('.csv', '')
        
        if 'STAID' in df.columns:
            n_stations = len(df)
            print(f"{sheet_name}: {n_stations} stations, {len(df.columns)-1} variables")
            
            if n_stations != expected_stations:
                print(f"  WARNING: Expected {expected_stations} stations, found {n_stations}")
        else:
            print(f"{sheet_name}: No STAID column (likely reference sheet)")

# Run verification
verify_basin_data(basin_char_dir)
```

## Example Complete Analysis

See the `simple_climate_synchrony_plots.py` module for a complete example of how to:
1. Load filtered basin characteristics
2. Merge with synchrony scales
3. Calculate correlations
4. Create visualizations
5. Save results

This pattern can be adapted for any basin characteristic analysis.

## Next Steps

1. **Explore Individual Variables**: Start with climate, then expand to topography, land cover, etc.
2. **Multi-Variable Analysis**: Use multiple regression or machine learning approaches
3. **Regional Patterns**: Analyze how basin characteristics vary by geographic region
4. **Temporal Analysis**: If you have time-varying synchrony measures, explore temporal relationships
5. **Comparative Studies**: Compare your 450 stations to the broader GAGES-II dataset

## Contact and Support

For questions about specific variables, refer to the original GAGES-II documentation or the variable description CSV you have. The filtered data maintains all original variable definitions and units from the source dataset.