# visualization.py
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import os

def _plot_kde_on_ax(ax, data, label, color, linewidth=2, alpha=0.8):
    """Helper function to calculate and plot a KDE curve on a given axis."""
    if len(data) > 1:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        ax.plot(x_range, kde(x_range), color=color, linewidth=linewidth, label=label, alpha=alpha)

def create_pdf_plot(station_id, obs_data, elnino_realizations, lanina_realizations, save_path=None, log_scale=False):
    """Creates and saves a single PDF comparison plot for one station."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    obs = obs_data.dropna().values
    elnino = elnino_realizations.values.flatten()
    elnino = elnino[~np.isnan(elnino)]
    lanina = lanina_realizations.values.flatten()
    lanina = lanina[~np.isnan(lanina)]

    if log_scale:
        # A simple way to handle log scale for data with zeros
        obs = np.log10(obs + 0.01)
        elnino = np.log10(elnino + 0.01)
        lanina = np.log10(lanina + 0.01)
        flow_label = 'Log₁₀(Streamflow)'
        title_suffix = ' (Log Scale)'
    else:
        flow_label = 'Streamflow'
        title_suffix = ''
        
    # Plotting
    _plot_kde_on_ax(ax, obs, 'Observed', 'black', linewidth=3)
    _plot_kde_on_ax(ax, elnino, 'El Niño (All Realizations)', 'red')
    _plot_kde_on_ax(ax, lanina, 'La Niña (All Realizations)', 'blue')
    
    ax.set_xlabel(flow_label)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Station {station_id}: PDF Comparison{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    
    
def create_combined_pdf_plot(all_obs_data, all_elnino_data, all_lanina_data, num_stations, save_path=None, log_scale=False):
    """Creates and saves a single PDF summary plot for all stations combined."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # The data is already aggregated and cleaned NumPy arrays, so we can use them directly.
    obs = all_obs_data
    elnino = all_elnino_data
    lanina = all_lanina_data

    if log_scale:
        obs = np.log10(obs[obs > 0] + 0.01) # Filter out non-positive values before log
        elnino = np.log10(elnino[elnino > 0] + 0.01)
        lanina = np.log10(lanina[lanina > 0] + 0.01)
        flow_label = 'Log₁₀(Streamflow)'
        title_suffix = ' (Log Scale)'
    else:
        flow_label = 'Streamflow'
        title_suffix = ''
        
    # Plotting using the same helper function! This is the power of modular code.
    _plot_kde_on_ax(ax, obs, 'Observed', 'black', linewidth=3)
    _plot_kde_on_ax(ax, elnino, 'El Niño (All Realizations)', 'red')
    _plot_kde_on_ax(ax, lanina, 'La Niña (All Realizations)', 'blue')
    
    ax.set_xlabel(flow_label)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'All Stations Combined ({num_stations} Stations): PDF Comparison{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Combined plot saved to {save_path}")
    
    plt.show()
    
    
    
