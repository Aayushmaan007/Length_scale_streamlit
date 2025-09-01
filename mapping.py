# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 22:08:47 2025

@author: aayus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from geopy.distance import geodesic
from matplotlib.patches import Circle

def _get_color_for_correlation(correlation_value):
    """
    Returns a color based on the Spearman correlation value.
    - Strong positive (>= 0.7): Dark Green
    - Moderate positive (>= 0.4): Light Green
    - Weak (-0.4 < r < 0.4): Gray
    - Moderate negative (<= -0.4): Light Red
    - Strong negative (<= -0.7): Dark Red
    """
    if correlation_value >= 0.7:
        return 'darkgreen'
    elif correlation_value >= 0.4:
        return 'limegreen'
    elif correlation_value > -0.4:
        return 'gray'
    elif correlation_value > -0.7:
        return 'salmon'
    else:
        return 'darkred'

def create_correlation_map(base_station_id, correlation_matrix, coordinates, save_path=None):
    """
    Creates a static Cartopy map to visualize streamflow correlations.
    """
    # ... (rest of the function remains the same)
    if base_station_id not in coordinates:
        print(f"Error: Base station '{base_station_id}' not found in coordinates file.")
        return

    # --- Matplotlib and Cartopy Setup ---
    # Define the map projection
    projection = ccrs.Mercator()
    # Use PlateCarree for plotting lat/lon data
    data_transform = ccrs.PlateCarree()
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': projection})

    # --- Calculate Map Extent ---
    lons = [coords['lon'] for coords in coordinates.values()]
    lats = [coords['lat'] for coords in coordinates.values()]
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1], crs=data_transform)

    # --- Add Geographic Features ---
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.LAKES, color='lightblue')
    ax.add_feature(cfeature.RIVERS)
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # --- Plot Data ---
    base_coords = [coordinates[base_station_id]['lon'], coordinates[base_station_id]['lat']]
    correlations = correlation_matrix[base_station_id]

    # Plot lines and station markers
    for station_id, coords in coordinates.items():
        if station_id in correlations:
            corr_value = correlations[station_id]
            color = _get_color_for_correlation(corr_value)
            
            # Plot a line from the base station to the current station
            if station_id != base_station_id:
                ax.plot([base_coords[0], coords['lon']], [base_coords[1], coords['lat']],
                        color=color, linewidth=2, transform=data_transform,
                        alpha=0.8)

            # Plot the station marker
            ax.scatter(coords['lon'], coords['lat'],
                       s=100, c=color, edgecolors='black',
                       transform=data_transform, zorder=5)

    # Plot a special marker for the base station
    ax.scatter(base_coords[0], base_coords[1],
               s=250, c='blue', marker='*', edgecolors='white',
               transform=data_transform, zorder=10, label=f'Base Station ({base_station_id})')

    # --- Create a Custom Legend ---
    legend_elements = [
        Line2D([0], [0], color='darkgreen', lw=4, label='Strong Positive (r ≥ 0.7)'),
        Line2D([0], [0], color='limegreen', lw=4, label='Moderate Positive (0.4 ≤ r < 0.7)'),
        Line2D([0], [0], color='gray', lw=4, label='Weak (-0.4 < r < 0.4)'),
        Line2D([0], [0], color='salmon', lw=4, label='Moderate Negative (-0.7 < r ≤ -0.4)'),
        Line2D([0], [0], color='darkred', lw=4, label='Strong Negative (r ≤ -0.7)'),
        plt.scatter([], [], s=250, c='blue', marker='*', edgecolors='white', label=f'Base Station ({base_station_id})')
    ]
    ax.legend(handles=legend_elements, loc='lower left', title="Spearman Correlation")
    
    ax.set_title(f"Streamflow Correlation with Station {base_station_id}", fontsize=16)

    # Save the map to a PNG file
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation map saved to: {save_path}")
    
    plt.show()


def create_interactive_lengthscale_visualization(
    base_station_id, correlation_matrix, coordinates, viz_data, current_radius
):
    """
    Creates an interactive, two-part plot for the Streamlit app.
    """
    fig = plt.figure(figsize=(20, 9))
    
    # --- Part 1: Map Plot with Expanding Radius ---
    ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.Mercator())
    data_transform = ccrs.PlateCarree()

    lons = [c['lon'] for c in coordinates.values()]
    lats = [c['lat'] for c in coordinates.values()]
    ax_map.set_extent([min(lons) - 1, max(lons) + 1, min(lats) - 1, max(lats) + 1], crs=data_transform)
    
    ax_map.add_feature(cfeature.LAND, color='#f0f0f0')
    ax_map.add_feature(cfeature.OCEAN, color='#d1e0e0')
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.gridlines(draw_labels=True, alpha=0.5)

    base_coord = coordinates[base_station_id]
    
    # Draw the expanding circle
    ax_map.tissot(rad_km=current_radius, lons=[base_coord['lon']], lats=[base_coord['lat']],
                  n_samples=100, facecolor='blue', alpha=0.2, edgecolor='blue')

    # Plot all stations, highlighting those inside the radius
    for station_id, coord in coordinates.items():
        if station_id not in correlation_matrix.index:
            continue
        
        dist_km = geodesic((base_coord['lat'], base_coord['lon']), (coord['lat'], coord['lon'])).kilometers
        corr_val = correlation_matrix.loc[base_station_id, station_id]
        is_correlated = abs(corr_val) >= viz_data['correlation_threshold']

        color = 'gray'
        edgecolor = 'black'
        size = 30
        alpha = 0.5
        
        if dist_km <= current_radius:
            color = 'red' if is_correlated else 'orange'
            size = 90
            alpha = 1.0

        ax_map.scatter(coord['lon'], coord['lat'], s=size, c=color, edgecolors=edgecolor,
                       transform=data_transform, zorder=5, alpha=alpha)

    ax_map.scatter(base_coord['lon'], base_coord['lat'], s=250, c='yellow', marker='*',
                   edgecolors='black', transform=data_transform, zorder=10)
    
    ax_map.set_title(f"Searching from Station {base_station_id}\nCurrent Radius: {current_radius} km")

    # --- Part 2: Fraction vs. Radius Plot ---
    ax_line = fig.add_subplot(1, 2, 2)
    
    ax_line.plot(viz_data['radii'], viz_data['fractions'], 'o-', color='gray', label='Fraction of correlated stations')
    ax_line.axhline(y=viz_data['fraction_threshold'], color='r', linestyle='--', label=f'Fraction Threshold')
    ax_line.axvline(x=viz_data['synchrony_scale'], color='g', linestyle='--', label=f'Final Synchrony Scale ({viz_data["synchrony_scale"]} km)')
    
    # Highlight the current point on the graph
    current_fraction_index = min(range(len(viz_data['radii'])), key=lambda i: abs(viz_data['radii'][i] - current_radius))
    ax_line.plot(viz_data['radii'][current_fraction_index], viz_data['fractions'][current_fraction_index], 'o',
                 markersize=15, color='blue', alpha=0.7, label='Current Radius')

    ax_line.set_xlabel("Search Radius (km)")
    ax_line.set_ylabel(f"Fraction of Stations with |r| ≥ {viz_data['correlation_threshold']}")
    ax_line.set_title("How Synchrony Scale is Calculated")
    ax_line.legend()
    ax_line.grid(True, alpha=0.3)
    ax_line.set_ylim(0, 1.05)
    ax_line.set_xlim(0, viz_data['radii'][-1])

    plt.tight_layout()
    return fig

# ... (rest of the mapping functions remain the same)
def create_average_correlation_map(average_correlations, coordinates, save_path=None):
    """
    Creates a Cartopy map visualizing the average correlation of each station.
    This is styled similarly to Figure 2 in Berghuijs et al. (2019).
    """
    projection = ccrs.Mercator()
    data_transform = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': projection})

    lons = [coords['lon'] for coords in coordinates.values()]
    lats = [coords['lat'] for coords in coordinates.values()]
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1], crs=data_transform)

    ax.add_feature(cfeature.LAND, color='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, color='#d1e0e0')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Prepare data for plotting
    plot_lons = [coordinates[st_id]['lon'] for st_id in average_correlations.index]
    plot_lats = [coordinates[st_id]['lat'] for st_id in average_correlations.index]
    plot_colors = average_correlations.values

    # Create the scatter plot with a colormap
    scatter = ax.scatter(plot_lons, plot_lats,
                         s=50, c=plot_colors,
                         cmap='YlOrRd', # A nice yellow-to-red colormap
                         edgecolors='black', linewidth=0.5,
                         transform=data_transform, zorder=5,
                         vmin=0, vmax=max(plot_colors)) # Set color limits

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.6)
    cbar.set_label('Average Spearman Correlation', fontsize=12)

    # --- Create an Inset Histogram (like in the paper) ---
    # Define position: [left, bottom, width, height] in figure coordinates
    inset_ax = fig.add_axes([0.18, 0.65, 0.25, 0.2])
    inset_ax.hist(average_correlations, bins=20, color='gray', edgecolor='white')
    inset_ax.set_title('Distribution of Values', fontsize=10)
    inset_ax.set_xlabel('Avg. Correlation', fontsize=8)
    inset_ax.set_ylabel('Frequency', fontsize=8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    inset_ax.set_facecolor('white')
    inset_ax.patch.set_alpha(0.8) # Make it slightly transparent

    ax.set_title("Spatial Distribution of Average Station Correlation", fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average correlation map saved to: {save_path}")
    
    plt.show()

def create_synchrony_scale_map(synchrony_scales, coordinates, save_path=None):
    """
    Creates a map visualizing correlation synchrony scales for each station.
    Similar to Figure 2 in Berghuijs et al. (2019) but for correlation synchrony.
    
    Args:
        synchrony_scales (pd.Series): Correlation synchrony scales for each station
        coordinates (dict): Dictionary with station IDs as keys and their lat/lon as values
        save_path (str, optional): Path to save the map
    """
    projection = ccrs.Mercator()
    data_transform = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': projection})

    # Get extent from coordinates
    lons = [coords['lon'] for coords in coordinates.values()]
    lats = [coords['lat'] for coords in coordinates.values()]
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1], crs=data_transform)

    # Add geographic features
    ax.add_feature(cfeature.LAND, color='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, color='#d1e0e0')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':', alpha=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Filter for stations that have both coordinates and synchrony scales
    valid_stations = [st_id for st_id in synchrony_scales.index 
                     if st_id in coordinates and synchrony_scales[st_id] > 0]
    
    if not valid_stations:
        print("No valid stations with both coordinates and synchrony scales > 0")
        return
    
    # Prepare data for plotting
    plot_lons = [coordinates[st_id]['lon'] for st_id in valid_stations]
    plot_lats = [coordinates[st_id]['lat'] for st_id in valid_stations]
    plot_values = [synchrony_scales[st_id] for st_id in valid_stations]

    # Create the scatter plot with a colormap
    scatter = ax.scatter(plot_lons, plot_lats,
                         s=80, c=plot_values,
                         cmap='plasma',  # Purple-to-yellow colormap, good for distances
                         edgecolors='black', linewidth=0.5,
                         transform=data_transform, zorder=5,
                         vmin=0, vmax=max(plot_values))

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.6)
    cbar.set_label('Correlation Synchrony Scale (km)', fontsize=12)

    # Add statistics text box
    stats_text = (f'n = {len(valid_stations)}\n'
                 f'Mean: {np.mean(plot_values):.0f} km\n'
                 f'Median: {np.median(plot_values):.0f} km\n'
                 f'Range: {min(plot_values):.0f}-{max(plot_values):.0f} km')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    # Add inset histogram
    inset_ax = fig.add_axes([0.65, 0.65, 0.25, 0.2])
    inset_ax.hist(plot_values, bins=15, color='skyblue', edgecolor='white', alpha=0.7)
    inset_ax.set_title('Distribution', fontsize=10)
    inset_ax.set_xlabel('Scale (km)', fontsize=8)
    inset_ax.set_ylabel('Count', fontsize=8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    inset_ax.set_facecolor('white')
    inset_ax.patch.set_alpha(0.9)

    ax.set_title("Correlation Synchrony Scales\n(Radius where ≥50% of stations have r≥0.5)", 
                 fontsize=16, pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation synchrony map saved to: {save_path}")
    
    plt.show()

def create_synchrony_comparison_map(synchrony_scales1, synchrony_scales2, coordinates, 
                                  labels=None, save_path=None):
    """
    Creates a side-by-side comparison of two different synchrony scale calculations.
    Useful for comparing different thresholds or methods.
    
    Args:
        synchrony_scales1, synchrony_scales2 (pd.Series): Two sets of synchrony scales
        coordinates (dict): Station coordinates
        labels (list): Labels for the two maps [default: ['Method 1', 'Method 2']]
        save_path (str): Path to save the comparison map
    """
    if labels is None:
        labels = ['Method 1', 'Method 2']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                   subplot_kw={'projection': ccrs.Mercator()})
    
    for ax, scales, label in zip([ax1, ax2], [synchrony_scales1, synchrony_scales2], labels):
        # Set up map
        lons = [coords['lon'] for coords in coordinates.values()]
        lats = [coords['lat'] for coords in coordinates.values()]
        ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1], 
                      crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, color='#f0f0f0')
        ax.add_feature(cfeature.OCEAN, color='#d1e0e0')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Plot data
        valid_stations = [st_id for st_id in scales.index 
                         if st_id in coordinates and scales[st_id] > 0]
        
        if valid_stations:
            plot_lons = [coordinates[st_id]['lon'] for st_id in valid_stations]
            plot_lats = [coordinates[st_id]['lat'] for st_id in valid_stations]
            plot_values = [scales[st_id] for st_id in valid_stations]
            
            scatter = ax.scatter(plot_lons, plot_lats, s=60, c=plot_values,
                               cmap='plasma', edgecolors='black', linewidth=0.3,
                               transform=ccrs.PlateCarree(), zorder=5,
                               vmin=0, vmax=max(max(synchrony_scales1), max(synchrony_scales2)))
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
            cbar.set_label('Synchrony Scale (km)', fontsize=10)
        
        ax.set_title(label, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison map saved to: {save_path}")
    
    plt.show()

def create_enso_synchrony_comparison_maps(synchrony_results, coordinates, save_path=None):
    """
    Create comparison map showing synchrony scales under different conditions.
    Automatically handles 3 or 4 conditions based on input data.
    
    Args:
        synchrony_results (dict): Results from analysis.compare_enso_synchrony_scales()
        coordinates (dict): Station coordinates
        save_path (str): Path to save the figure
    """
    # Determine available conditions and layout
    conditions = ['observed', 'baseline', 'el_nino', 'la_nina']
    titles = ['Observed', 'Baseline', 'El Niño', 'La Niña']
    
    available_conditions = [(cond, title) for cond, title in zip(conditions, titles) 
                           if cond in synchrony_results]
    
    n_conditions = len(available_conditions)
    
    if n_conditions <= 2:
        fig, axes = plt.subplots(1, n_conditions, figsize=(8*n_conditions, 6),
                                subplot_kw={'projection': ccrs.Mercator()})
    elif n_conditions == 3:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                                subplot_kw={'projection': ccrs.Mercator()})
    else:  # 4 conditions
        fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                                subplot_kw={'projection': ccrs.Mercator()})
        axes = axes.flatten()
    
    if n_conditions == 1:
        axes = [axes]
    
    # Get global min/max for consistent color scaling
    all_scales = []
    for condition, _ in available_conditions:
        scales = [s for s in synchrony_results[condition]['scales'].values() if s > 0]
        all_scales.extend(scales)
    
    vmin, vmax = 0, max(all_scales) if all_scales else 3000
    
    for ax, (condition, title) in zip(axes, available_conditions):
        # Set up map
        lons = [coords['lon'] for coords in coordinates.values()]
        lats = [coords['lat'] for coords in coordinates.values()]
        ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1], 
                     crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, color='#f0f0f0')
        ax.add_feature(cfeature.OCEAN, color='#d1e0e0')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':', alpha=0.5)
        
        # Plot synchrony scales
        synchrony_scales = synchrony_results[condition]['scales']
        valid_stations = [st_id for st_id in synchrony_scales.keys() 
                         if st_id in coordinates and synchrony_scales[st_id] > 0]
        
        if valid_stations:
            plot_lons = [coordinates[st_id]['lon'] for st_id in valid_stations]
            plot_lats = [coordinates[st_id]['lat'] for st_id in valid_stations]
            plot_values = [synchrony_scales[st_id] for st_id in valid_stations]
            
            scatter = ax.scatter(plot_lons, plot_lats, s=60, c=plot_values,
                               cmap='plasma', edgecolors='black', linewidth=0.3,
                               transform=ccrs.PlateCarree(), zorder=5,
                               vmin=vmin, vmax=vmax)
        
        # Add statistics text
        stats = synchrony_results[condition]['stats']
        stats_text = (f"n = {stats['count']}\n"
                      f"Mean: {stats['mean']:.0f} km\n"
                      f"Median: {stats['median']:.0f} km")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
        
        ax.set_title(title, fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if n_conditions > 2 else 0.1)

    # Add shared colorbar
    cbar = fig.colorbar(scatter, ax=axes if n_conditions > 1 else axes[0], 
                       orientation='horizontal', shrink=0.6, pad=0.08, aspect=30)
    cbar.set_label('Correlation Synchrony Scale (km)', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Synchrony comparison maps saved to: {save_path}")
    
    plt.show()

def create_enso_synchrony_distributions(synchrony_results, save_path=None):
    """
    Create histogram comparison of synchrony scale distributions under different conditions.
    Automatically handles 3 or 4 conditions based on input data.
    
    Args:
        synchrony_results (dict): Results from analysis.compare_enso_synchrony_scales()
        save_path (str): Path to save the figure
    """
    # Determine available conditions
    conditions = ['observed', 'baseline', 'el_nino', 'la_nina']
    titles = ['Observed', 'Baseline', 'El Niño', 'La Niña']  
    colors = ['black', 'green', 'red', 'blue']
    
    available_data = [(cond, title, color) for cond, title, color in zip(conditions, titles, colors)
                      if cond in synchrony_results]
    
    n_conditions = len(available_data)
    
    # Adjust subplot layout based on number of conditions
    if n_conditions <= 3:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Individual histograms
    for i, (condition, title, color) in enumerate(available_data):
        scales = [s for s in synchrony_results[condition]['scales'].values() if s > 0]
        
        if n_conditions <= 3:
            ax = axes[i//2, i%2]
        else:
            row = 0 if i < 2 else 1
            col = i % 2 if i < 2 else (i-2) % 2
            ax = axes[row, col]
        
        ax.hist(scales, bins=20, color=color, alpha=0.7, edgecolor='white')
        ax.set_title(title)
        ax.set_xlabel('Synchrony Scale (km)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if scales:
            mean_val = np.mean(scales)
            ax.axvline(mean_val, color='darkred', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.0f} km')
            ax.legend()
    
    # Combined comparison in remaining subplot
    if n_conditions <= 3:
        ax = axes[1, 1]
    else:
        ax = axes[1, 1]
    
    for condition, title, color in available_data:
        scales = [s for s in synchrony_results[condition]['scales'].values() if s > 0]
        if scales:
            ax.hist(scales, bins=20, alpha=0.5, label=title, color=color, density=True)
    
    ax.set_title('Distribution Comparison')
    ax.set_xlabel('Synchrony Scale (km)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot comparison if we have 4 conditions
    if n_conditions == 4:
        ax = axes[0, 2]
        box_data = []
        box_labels = []
        
        for condition, title, color in available_data:
            scales = [s for s in synchrony_results[condition]['scales'].values() if s > 0]
            if scales:
                box_data.append(scales)
                box_labels.append(title)
        
        ax.boxplot(box_data, labels=box_labels)
        ax.set_title('Box Plot Comparison')
        ax.set_ylabel('Synchrony Scale (km)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution comparison saved to: {save_path}")
    
    plt.show()

def create_correlation_distance_plots(correlation_data, save_path=None):
    """
    Create correlation vs distance decay plots for all available conditions.
    Automatically handles 3 or 4 conditions based on input data.
    
    Args:
        correlation_data (dict): Output from calculate_correlation_distance_relationships()
        save_path (str): Path to save the figure
    """
    # Determine available conditions
    conditions = ['observed', 'baseline', 'el_nino', 'la_nina']
    titles = ['Observed', 'Baseline', 'El Niño', 'La Niña']
    colors = ['black', 'green', 'red', 'blue']
    
    available_data = [(cond, title, color) for cond, title, color in zip(conditions, titles, colors)
                      if cond in correlation_data]
    
    n_conditions = len(available_data)
    
    if n_conditions <= 3:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Individual plots
    for i, (condition, title, color) in enumerate(available_data):
        if n_conditions <= 3:
            ax = axes[i//2, i%2]
        else:
            row = 0 if i < 2 else 1
            col = i % 2 if i < 2 else (i-2)
            ax = axes[row, col]
        
        data = correlation_data[condition]
        
        # Remove NaN values
        valid_mask = ~np.isnan(data['correlations'])
        distances = data['distances'][valid_mask]
        correlations = data['correlations'][valid_mask]
        std_vals = data['std'][valid_mask]
        
        # Plot mean correlation with error bars
        ax.errorbar(distances, correlations, yerr=std_vals, 
                   color=color, linewidth=2, capsize=3, alpha=0.8,
                   label=f'{title} (binned mean)')
        
        # Fit exponential decay curve
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(exp_decay, distances, correlations, 
                               p0=[0.5, 0.001, 0.1], maxfev=1000)
            
            x_fit = np.linspace(0, max(distances), 100)
            y_fit = exp_decay(x_fit, *popt)
            ax.plot(x_fit, y_fit, '--', color=color, alpha=0.6,
                   label=f'Fit: {popt[0]:.2f}*exp(-{popt[1]:.4f}*d)+{popt[2]:.2f}')
            
        except:
            pass
        
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Absolute Spearman Correlation')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    
    # Combined comparison
    if n_conditions <= 3:
        ax = axes[1, 1]
    else:
        ax = plt.subplot2grid((2, 3), (1, 2), rowspan=1)
    
    for condition, title, color in available_data:
        data = correlation_data[condition]
        valid_mask = ~np.isnan(data['correlations'])
        distances = data['distances'][valid_mask]
        correlations = data['correlations'][valid_mask]
        
        ax.plot(distances, correlations, 'o-', color=color, 
               linewidth=2, markersize=4, alpha=0.8, label=title)
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Absolute Spearman Correlation')
    ax.set_title('Correlation Decay Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation-distance plots saved to: {save_path}")
    
    plt.show()

def create_correlation_distance_scatter(correlation_data, condition='observed', save_path=None):
    """
    Create scatter plot of raw correlation vs distance data for one condition.
    Useful for seeing the full data distribution.
    """
    if condition not in correlation_data:
        print(f"No data for condition: {condition}")
        return
    
    data = correlation_data[condition]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data['raw_distances'], data['raw_correlations'], 
               alpha=0.1, s=1, color='blue')
    
    # Add binned means
    valid_mask = ~np.isnan(data['correlations'])
    plt.plot(data['distances'][valid_mask], data['correlations'][valid_mask], 
            'ro-', linewidth=2, markersize=6, label='Binned means')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Absolute Spearman Correlation')
    plt.title(f'Correlation vs Distance: {condition.title()}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    plt.show()