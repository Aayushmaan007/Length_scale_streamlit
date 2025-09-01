# app.py
import streamlit as st
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your existing modules
import data_loader
import analysis
import visualization
import mapping

# --- Page Configuration ---
st.set_page_config(
    page_title="Streamflow Synchrony Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- App Title ---
st.title("Streamflow Spatial Coherence and Synchrony Analysis")
st.markdown("An interactive web app to explore streamflow data and synchrony scales.")

# --- Caching Functions ---
# Use st.cache_data to prevent reloading data on every interaction
@st.cache_data
def load_all_data(config):
    """Loads all the necessary data for the app."""
    base_path = config['base_path']
    paths = config['paths']
    filenames = config['filenames']
    
    coordinates = data_loader.load_coordinates(os.path.join(base_path, paths['coordinates']))
    observed_data = data_loader.load_observed_data(
        os.path.join(base_path, paths['observed']), 
        filenames['observed_pattern']
    )
    return coordinates, observed_data

# --- NEW ---
# Cache the correlation matrix calculation, as it's computationally expensive
@st.cache_data
def compute_correlation_matrix(_observed_data): # Use _ to indicate the input triggers the cache
    """Calculates and caches the Spearman correlation matrix."""
    st.info("Calculating inter-station correlation matrix. This may take a moment...")
    correlation_matrix = analysis.calculate_inter_station_correlation(_observed_data)
    st.success("Correlation matrix calculated and cached!")
    return correlation_matrix

# --- Main App Logic ---
# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("ERROR: `config.yaml` not found. Please make sure it's in the same directory.")
    st.stop()

# Load data using the cached function
coordinates, observed_data = load_all_data(config)
station_ids = sorted(list(observed_data.keys()))

st.success(f"Successfully loaded data for {len(station_ids)} stations.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Analysis Controls")

# Dropdown to select a station
selected_station = st.sidebar.selectbox(
    "Select a Base Station for Analysis:",
    options=station_ids,
    index=station_ids.index('01487000') if '01487000' in station_ids else 0 # Default to a known station
)

st.sidebar.markdown("---")

# --- NEW ---
# Button to trigger the correlation analysis
show_correlation_map = st.sidebar.button("Show Correlation Map", type="primary")


# --- Main Panel for Displaying Results ---

st.header(f"Analysis for Station: `{selected_station}`")

# --- NEW: Display Correlation Map when the button is clicked ---
if show_correlation_map:
    # First, get the correlation matrix (it will be computed once and then retrieved from cache)
    correlation_matrix = compute_correlation_matrix(observed_data)

    if not correlation_matrix.empty:
        st.subheader(f"Correlation Map Relative to Station {selected_station}")
        
        # Generate the map using your existing mapping function
        # We need to capture the figure to display it in Streamlit
        fig_map, ax_map = plt.subplots(figsize=(12, 10), subplot_kw={'projection': mapping.ccrs.Mercator()})
        
        # We need to adapt the mapping function slightly to accept an axis object,
        # or we can recreate its logic here. For now, let's call the original and display.
        # NOTE: Your mapping function calls plt.show(), which we don't want in Streamlit.
        # We will need to modify it later for a cleaner integration.
        # For now, we suppress the warning.
        #st.set_option('deprecation.showPyplotGlobalUse', False)
        
        with st.spinner("Generating map..."):
            mapping.create_correlation_map(
                base_station_id=selected_station,
                correlation_matrix=correlation_matrix,
                coordinates=coordinates,
                # We pass save_path=None to prevent saving and just show it
                save_path=None
            )
            # Use st.pyplot() without arguments to grab the "global" plot created by your function
            st.pyplot()

    else:
        st.warning("Could not compute or retrieve correlation matrix.")
else:
    # This is the default view when the button hasn't been clicked
    st.subheader("Observed Streamflow Time Series")
    fig, ax = plt.subplots(figsize=(10, 4))
    observed_data[selected_station].plot(ax=ax, color='steelblue')
    ax.set_title(f"Observed Daily Flow for Station {selected_station}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Flow (cfs)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("Click the 'Show Correlation Map' button in the sidebar to run the spatial analysis.")