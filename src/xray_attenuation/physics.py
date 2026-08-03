import numpy as np


def get_transmission(mu: float, thickness: float) -> float:
    """Calculates the linear transmission coefficient

    Args:
        mu (float): linear attenuation coefficient [1/cm]
        thickness (float): material thickness

    Returns:
        float: linear transmission coefficient
    """
    return np.exp(-1 * mu * thickness)


def calculate_filtered_spectrum(
    spectrum: np.ndarray, mu: np.ndarray, thickness: float
) -> np.ndarray:
    """Calculates the filtered spectrum based on the input spectrum and filter thickness

    Args:
        spectrum (np.ndarray): Input spectrum with energy and intensity
        thickness (float): Filter thickness

    Returns:
        np.ndarray: Filtered spectrum
    """
    # Assuming the first column is energy and the second column is intensity
    energy = spectrum[:, 0]
    intensity = spectrum[:, 1]

    # Calculate the transmission for each energy
    new_intensity = intensity * get_transmission(mu, thickness)
    return np.column_stack((energy, new_intensity))
