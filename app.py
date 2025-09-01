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
st.title("Interactive Synchrony Scale Explorer")
st.markdown("Use the controls to visualize how the streamflow synchrony scale is calculated.")

# --- Caching Functions ---
@st.cache_data
def load_all_data(config):
    # ... (function remains the same)
    base_path = config['base_path']
    paths = config['paths']
    filenames = config['filenames']
    
    coordinates = data_loader.load_coordinates(os.path.join(base_path, paths['coordinates']))
    observed_data = data_loader.load_observed_data(
        os.path.join(base_path, paths['observed']), 
        filenames['observed_pattern']
    )
    return coordinates, observed_data

@st.cache_data
def compute_correlation_matrix(_observed_data):
    # ... (function remains the same)
    st.info("Calculating inter-station correlation matrix. This may take a moment...")
    correlation_matrix = analysis.calculate_inter_station_correlation(_observed_data)
    st.success("Correlation matrix calculated and cached!")
    return correlation_matrix

@st.cache_data
def get_viz_data(station_id, _correlation_matrix, _coordinates):
    """Caches the data needed for the visualization."""
    return analysis.calculate_synchrony_scale_for_station_for_visualization(
        base_station_id=station_id,
        correlation_matrix=_correlation_matrix,
        coordinates=_coordinates,
        correlation_threshold=0.7,
        fraction_threshold=0.5,
        max_radius_km=4000,
        radius_step_km=25
    )

# --- Main App Logic ---
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("ERROR: `config.yaml` not found.")
    st.stop()

coordinates, observed_data = load_all_data(config)
station_ids = sorted(list(observed_data.keys()))
correlation_matrix = compute_correlation_matrix(observed_data)

# --- Sidebar Controls ---
st.sidebar.header("Station Selection")
selected_station = st.sidebar.selectbox(
    "Select a Base Station:",
    options=station_ids,
    index=station_ids.index('01487000') if '01487000' in station_ids else 0
)

# --- Main Panel ---
if not correlation_matrix.empty:
    st.header(f"Visualizing for Station: `{selected_station}`")

    # Get the pre-calculated data for the plot
    viz_data = get_viz_data(selected_station, correlation_matrix, coordinates)
    
    # --- Interactive Slider ---
    st.markdown("### Animate the Search Radius")
    selected_radius = st.slider(
        "Drag the slider to change the search radius (km) and see the map update.",
        min_value=0,
        max_value=4000,
        value=viz_data['synchrony_scale'], # Start the slider at the final calculated value
        step=25 # Should match the radius_step_km
    )

    # --- Create and Display the Plot ---
    with st.spinner("Generating interactive plot..."):
        fig = mapping.create_interactive_lengthscale_visualization(
            selected_station,
            correlation_matrix,
            coordinates,
            viz_data,
            selected_radius
        )
        st.pyplot(fig, use_container_width=True)

else:
    st.warning("Could not compute or retrieve the correlation matrix.")