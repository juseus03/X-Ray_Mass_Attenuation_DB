import streamlit as st
from XrayTransmission import Simulation
import plotly.graph_objects as go
import numpy as np
from plotly_style import setup_custom_template, apply_style

setup_custom_template()

st.title("X-ray Spectra Transmission Simulator")


# Initialize simulation
@st.cache_resource
def load_simulation():
    return Simulation()


sim = load_simulation()

# Initialize filters list in session state
if "filters" not in st.session_state:
    st.session_state.filters = []

# Initialize filter ID counter for stable keys
if "filter_counter" not in st.session_state:
    st.session_state.filter_counter = 0

# Sidebar controls
st.sidebar.header("Parameters")

# Spectrum energy selector
spectrum_energy = st.sidebar.slider(
    "Spectrum Energy (kV)", min_value=9, max_value=100, value=20, step=1
)
spectrum_energy = str(spectrum_energy)

# Filters section
st.sidebar.header("Filters")

# Add filter button
if st.sidebar.button("Add Filter"):
    # Add a new empty filter with unique ID
    st.session_state.filters.append(
        {
            "id": st.session_state.filter_counter,
            "type": "Element",
            "material": "Aluminum",
            "thickness": 0.1,
        }
    )
    st.session_state.filter_counter += 1

# Display existing filters
filters_to_remove = []
filters_changed = False

for i, filter_data in enumerate(st.session_state.filters):
    filter_id = filter_data.get(
        "id", i
    )  # Use ID if exists, fallback to index for old filters

    with st.sidebar.expander(f"Filter {i+1}", expanded=True):
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Remove", key=f"remove_{filter_id}"):
                filters_to_remove.append(i)

        # Filter type selector
        old_type = filter_data["type"]
        filter_type = st.radio(
            "Type",
            options=["Element", "Compound"],
            key=f"type_{filter_id}",
            index=0 if filter_data["type"] == "Element" else 1,
        )
        if filter_type != old_type:
            filters_changed = True
        st.session_state.filters[i]["type"] = filter_type

        # Material selector based on filter type
        if filter_type == "Element":
            materials = sim.info_elements["Element"].to_list()
            symbols = sim.info_elements["Symbol"].to_list()

            joint_list = np.array(
                [f"{m} ({symbols[i]})" for i, m in enumerate(materials)]
            )

            default_index = (
                12
                if filter_data["material"] not in materials
                else materials.index(filter_data["material"])
            )
            material = st.selectbox(
                "Element", joint_list, index=default_index, key=f"material_{filter_id}"
            )
            idx = np.argwhere(joint_list == material)[0][0]
            material = materials[idx]
            is_element = True
        else:
            materials = sim.info_compounds["Material"].to_list()
            default_index = (
                0
                if filter_data["material"] not in materials
                else materials.index(filter_data["material"])
            )
            material = st.selectbox(
                "Compound", materials, index=default_index, key=f"material_{filter_id}"
            )
            is_element = False

        # Check if material changed
        if material != filter_data["material"]:
            filters_changed = True
        st.session_state.filters[i]["material"] = material

        # Thickness input
        old_thickness = filter_data["thickness"]
        thickness = st.number_input(
            "Thickness (cm)",
            min_value=0.01,
            value=filter_data["thickness"],
            step=0.01,
            key=f"thickness_{filter_id}",
            format="%.3f",
        )
        if abs(thickness - old_thickness) > 1e-6:
            filters_changed = True
        st.session_state.filters[i]["thickness"] = thickness
        st.session_state.filters[i]["is_element"] = is_element

# Remove filters marked for deletion
if filters_to_remove:
    for i in reversed(filters_to_remove):
        st.session_state.filters.pop(i)
        sim.remove_filter(i)
    st.rerun()

# Trigger rerun if filters changed to ensure simulation updates
if filters_changed:
    st.rerun()

# Reinitialize spectrum and apply all filters to simulation
sim.set_base_spectrum(spectrum_energy)
for i, filter_data in enumerate(st.session_state.filters):
    sim.add_filter(
        filter_data["material"], filter_data["thickness"], filter_data["is_element"], i
    )

# Plotting
st.sidebar.header("Plot Parameters")

is_log_y = st.sidebar.checkbox("Log Y", value=True)

# Generate plot
fig = go.Figure()

# Plot open beam
spectrum = sim.get_base_spectrum()
energy = spectrum["Energy[keV]"].to_numpy().reshape(-1)
intensity = spectrum[spectrum_energy].to_numpy().reshape(-1)
fig.add_trace(
    go.Scatter(
        x=energy, y=intensity, line=dict(color="black"), name="Open Beam", mode="lines"
    )
)

# Plot filtered spectrum if filters exist
if len(st.session_state.filters) > 0:
    spectrum2 = sim.get_current_spectrum()
    energy2 = spectrum2["Energy[keV]"].to_numpy().reshape(-1)
    intensity2 = spectrum2[spectrum_energy].to_numpy().reshape(-1)

    # Create label from all filters
    filter_labels = [
        f"{f['material']} ({f['thickness']:.2f} cm)" for f in st.session_state.filters
    ]
    label = "<br>+ ".join(filter_labels)

    fig.add_trace(
        go.Scatter(
            x=energy2,
            y=intensity2,
            name=label,
            line=dict(dash="dash", color="red"),
        )
    )

fig.update_layout(
    xaxis_title="Energy [keV]",
    yaxis_title="Intensity",
    yaxis_type="log" if is_log_y else "linear",
    yaxis_range=[-10, -5] if is_log_y else None,
    height=600,
    showlegend=True,
    yaxis=dict(exponentformat="e", dtick=1 if is_log_y else 0),
)

fig = apply_style(fig)

st.plotly_chart(fig, use_container_width=True)
