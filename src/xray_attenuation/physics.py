import numpy as np
from numpy.typing import NDArray


def get_transmission(
    mu: float | NDArray[np.float64], thickness: float
) -> np.float64 | NDArray[np.float64]:
    """Calculates the linear transmission coefficient (Beer-Lambert)

    Works element-wise, so mu may be a single coefficient or a whole attenuation curve.

    Args:
        mu (float | NDArray[np.float64]): linear attenuation coefficient(s) [1/cm]
        thickness (float): material thickness [cm]

    Returns:
        np.float64 | NDArray[np.float64]: linear transmission coefficient. A scalar mu
            gives a scalar back, an array gives an array of the same shape
    """
    return np.exp(-mu * thickness)


def calculate_filtered_spectrum(
    spectrum: np.ndarray, mu: np.ndarray, thickness: float
) -> np.ndarray:
    """Calculates the filtered spectrum based on the input spectrum and filter thickness

    Args:
        spectrum (np.ndarray): Input spectrum intensities
        mu (np.ndarray): Linear attenuation coefficients [1/cm], one per spectrum bin
        thickness (float): Filter thickness [cm]

    Returns:
        np.ndarray: Filtered spectrum
    """
    return spectrum * get_transmission(mu, thickness)
