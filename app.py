# app.py (Final Version for Deployment)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
import io

# Import your existing modules
import analysis
import mapping
import data_loader # We still need this for coordinates

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
def load_data_from_urls():
    """
    Downloads the pre-computed correlation matrix and the coordinates file
    from their public GitHub URLs.
    """
    # URL for the pre-computed correlation matrix from your GitHub Release
    matrix_url = "https://github.com/Aayushmaan007/Length_scale_streamlit/releases/download/v1.0/correlation_matrix.pkl"
    
    # URL for the coordinates file directly from your main branch
    # Note: This is the "raw" content link, not the regular GitHub page link.
    coords_url = "https://raw.githubusercontent.com/Aayushmaan007/Length_scale_streamlit/master/datasets/coordinates/station_coordinates.txt"

    st.info(f"Downloading pre-computed correlation matrix...")
    try:
        # Download and load the correlation matrix
        response_matrix = requests.get(matrix_url)
        response_matrix.raise_for_status() # Raises an error for bad responses (404, 500, etc.)
        correlation_matrix = pickle.load(io.BytesIO(response_matrix.content))
        st.success("Correlation matrix loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load correlation matrix: {e}")
        return None, None

    st.info(f"Downloading station coordinates...")
    try:
        # Download and load the coordinates
        response_coords = requests.get(coords_url)
        response_coords.raise_for_status()
        # We need to simulate reading a file for your existing data_loader function
        coordinates_text = response_coords.text
        coordinates = {}
        for line in coordinates_text.strip().split('\n'):
            if ':' in line:
                station_id, coords_str = line.strip().split(': ')
                lat, lon = map(float, coords_str.strip('()').split(', '))
                coordinates[station_id] = {'lat': lat, 'lon': lon}
        st.success("Station coordinates loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load coordinates: {e}")
        return correlation_matrix, None

    return correlation_matrix, coordinates

# --- Main App Logic ---

# Load the data using the new download function
correlation_matrix, coordinates = load_data_from_urls()

# --- Sidebar Controls ---
st.sidebar.header("Station Selection")

if correlation_matrix is not None and coordinates is not None:
    # Get station IDs from the columns of the loaded matrix
    station_ids = sorted(list(correlation_matrix.columns))
    
    selected_station = st.sidebar.selectbox(
        "Select a Base Station:",
        options=station_ids,
        index=station_ids.index('01487000') if '01487000' in station_ids else 0
    )

    # --- Main Panel ---
    st.header(f"Visualizing for Station: `{selected_station}`")

    # Get the pre-calculated data for the plot using the analysis function
    # This function is fast because the heavy lifting is already done.
    viz_data = analysis.calculate_synchrony_scale_for_station_for_visualization(
        base_station_id=selected_station,
        correlation_matrix=correlation_matrix,
        coordinates=coordinates,
        correlation_threshold=0.7,
        fraction_threshold=0.5,
        max_radius_km=4000,
        radius_step_km=25
    )
    
    # --- Interactive Slider ---
    st.markdown("### Animate the Search Radius")
    selected_radius = st.slider(
        "Drag the slider to change the search radius (km) and see the map update.",
        min_value=0,
        max_value=4000,
        value=viz_data['synchrony_scale'],
        step=25
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
    st.error("Data could not be loaded. The application cannot start. Please check the URLs in the script and your GitHub repository settings.")