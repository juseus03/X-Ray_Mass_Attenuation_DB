from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def get_transmission(mu: float, thickness: float) -> np.float64: ...


@overload
def get_transmission(
    mu: NDArray[np.float64], thickness: float
) -> NDArray[np.float64]: ...


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
    spectrum: NDArray[np.float64], mu: NDArray[np.float64], thickness: float
) -> NDArray[np.float64]:
    """Calculates the filtered spectrum based on the input spectrum and filter thickness

    Args:
        spectrum (NDArray[np.float64]): Input spectrum intensities
        mu (NDArray[np.float64]): Linear attenuation coefficients [1/cm], one per
            spectrum bin
        thickness (float): Filter thickness [cm]

    Returns:
        NDArray[np.float64]: Filtered spectrum
    """
    return spectrum * get_transmission(mu, thickness)


def calculate_total_filtered_fraction(
    spectrum_orig: NDArray[np.float64], spectrum_filter: NDArray[np.float64]
) -> float:
    """Gives the total filtered fraction of photons for an spectrum

    Args:
        spectrum_orig (NDArray[np.float64]): Non-filtered spectrum
        spectrum_filter (NDArray[np.float64]): Filtered spectrum

    Returns:
        float: fraction
    """

    i0 = np.sum(spectrum_orig)
    i1 = np.sum(spectrum_filter)
    f = np.round(i1 / i0, 5)

    if f > 1:
        raise ValueError()

    return float(f)


def get_mean_energy_spectrum(
    spectrum: NDArray[np.float64], spectrum_bins: NDArray[np.float64]
) -> float:
    """Calculates the weighted mean value of the spectrum

    Args:
        spectrum (NDArray[np.float64]): Intensity
        spectrum_bins (NDArray[np.float64]): Energy bins

    Returns:
        float: Mean energy
    """

    if np.sum(spectrum) == 0:
        return 0.0

    m = np.sum(spectrum * spectrum_bins) / np.sum(spectrum)

    return float(m)


def get_hvl(
    spectrum: NDArray[np.float64], mu: NDArray[np.float64], t_max: float = 1e4
) -> float:
    """Calculates the HVL of a material, determined by its linear attenuation
    coefficient, for the given spectrum

    Args:
        spectrum (NDArray[np.float64]): Intensity
        mu (NDArray[np.float64]): Linear attenuation coefficient [1/cm]
        t_max (float, optional): Largest thickness to search [cm]. Defaults to 1e4

    Returns:
        float: HVL in mm. 0.0 for an empty spectrum

    Raises:
        ValueError: if the beam is not halved anywhere below t_max,
        which happens when mu is zero or negative over the whole spectrum
    """

    i0 = np.sum(spectrum)

    if i0 <= 0:
        return 0.0

    if np.all(mu <= 0):
        raise ValueError

    low, high = 0.0, t_max  # cm
    frac_err = 1e6
    eps = 1e-12

    while abs(frac_err) >= eps:
        mid = 0.5 * (low + high)
        i1 = np.sum(calculate_filtered_spectrum(spectrum, mu, mid))
        frac_err = (i1 / i0) - 0.5

        if frac_err > 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high) * 10  # cm -> mm


def get_effective_energy(
    spectrum: NDArray[np.float64],
    spectrum_bins: NDArray[np.float64],
    mu: NDArray[np.float64],
    hvl: float | None = None,
) -> float:
    """Calculates the effective energy of the spectrum based on the
    HVL in a material determined by mu

    Args:
        spectrum (NDArray[np.float64]): Intensity
        spectrum_bins (NDArray[np.float64]): Energy bins
        mu (NDArray[np.float64]): Material linear attenuation. Standard is Aluminum
        hvl (float | None, optional): HVL in Material in mm. Defaults to None.

    Returns:
        float: Effective energy
    """

    if hvl is None:
        hvl = get_hvl(spectrum, mu) * 1e-1  # cm

    mu_eff = np.log(2) / (hvl * 1e-1)
    idx_high = int(np.argwhere(mu < mu_eff)[0][0])
    if idx_high == len(mu):
        return float(spectrum_bins[idx_high - 1])
    return float((spectrum_bins[idx_high] + spectrum_bins[idx_high + 1]) * 0.5)
