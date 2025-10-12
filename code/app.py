import streamlit as st
from XrayTransmission import Simulation
import plotly.graph_objects as go
from plotly_style import setup_custom_template, apply_style

setup_custom_template()

st.title("X-ray Spectra Transmission Simulator")


# Initialize simulation
@st.cache_resource
def load_simulation():
    return Simulation()


sim = load_simulation()


if "zoom_state" not in st.session_state:
    st.session_state.zoom_state = {}


# Sidebar controls
st.sidebar.header("Parameters")

# Spectrum energy selector
spectrum_energy = st.sidebar.slider(
    "Spectrum Energy (kV)", min_value=9, max_value=100, value=20, step=1
)
spectrum_energy = str(spectrum_energy)

# Filter type selector
filter_type = st.sidebar.radio("Filter Type", options=["None", "Element", "Compound"])

# Material selector based on filter type
if filter_type == "Element":
    materials = sim.info_elements["Element"].to_list()
    material = st.sidebar.selectbox("Select Element", materials, index=12)
    is_element = True
elif filter_type == "Compound":
    materials = sim.info_compounds["Material"].to_list()
    material = st.sidebar.selectbox("Select Compound", materials)
    is_element = False
else:
    material = None
    is_element = None

# Thickness slider
if filter_type != "None":
    thickness = st.sidebar.number_input(
        "Thickness (cm)", min_value=0.01, value=0.1, step=0.01
    )

# Plotting
st.sidebar.header("Plot Parameters")

is_log_y = st.sidebar.checkbox("Log Y", value=True)

# Generate plot
fig = go.Figure()

# Plot open beam
spectrum = sim.get_spectrum(spectrum_energy)
energy = spectrum["Energy[keV]"].to_numpy().reshape(-1)
intensity = spectrum[spectrum_energy].to_numpy().reshape(-1)
fig.add_trace(
    go.Scatter(
        x=energy, y=intensity, line=dict(color="black"), name="Open Beam", mode="lines"
    )
)

# Plot filtered spectrum if selected
if filter_type != "None":
    energy2, intensity2 = sim.calculate_transmited_spectrum(
        spectrum_energy, material, thickness, is_element
    )
    fig.add_trace(
        go.Scatter(
            x=energy2,
            y=intensity2,
            name=f"{material} ({thickness:.2f} cm)",
            line=dict(dash="dash", color="black"),
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
