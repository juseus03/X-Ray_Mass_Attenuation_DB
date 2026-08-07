import numpy as np
import polars as pl
import pytest

from xray_attenuation.data import Data
from xray_attenuation.physics import (
    calculate_filtered_spectrum,
    calculate_total_filtered_fraction,
    get_effective_energy,
    get_hvl,
    get_mean_energy_spectrum,
    get_transmission,
)


def test_get_transmission():
    assert get_transmission(0, 0.1) == 1
    assert get_transmission(1.0, 0.5) == pytest.approx(0.6065, abs=1e-4)
    assert get_transmission(1.0, 1e6) == pytest.approx(0, abs=1e-4)

    assert get_transmission(0.720613425, 0.1) == pytest.approx(
        0.93047, abs=1e-4
    )  # 1mm Al @ 60 keV
    assert get_transmission(3.758245771, 0.5) == pytest.approx(
        0.1527, abs=1e-4
    )  # 5mm CdTe @ 150 keV


def test_calculate_filtered_spectrum():
    spectrum = np.array([100, 200, 300])
    mu = np.array([0.1, 0.2, 0.3])
    thickness = 1.0

    filtered_spectrum = calculate_filtered_spectrum(spectrum, mu, thickness)

    expected_intensity = [
        100 * get_transmission(0.1, thickness),
        200 * get_transmission(0.2, thickness),
        300 * get_transmission(0.3, thickness),
    ]

    for i, fsp in enumerate(filtered_spectrum):
        assert fsp == pytest.approx(expected_intensity[i], abs=1e-4)


def test_calculate_total_filtered_fraction():
    s0 = np.array([1, 2, 3])

    f1 = calculate_total_filtered_fraction(s0, s0 * 0.5)
    f2 = calculate_total_filtered_fraction(s0, s0 * 0.1)
    f3 = calculate_total_filtered_fraction(s0, s0 * 0.3)
    f4 = calculate_total_filtered_fraction(s0, s0 * 0.8)
    f5 = calculate_total_filtered_fraction(s0, s0)

    assert f1 == pytest.approx(0.5, abs=1e-5)
    assert f2 == pytest.approx(0.1, abs=1e-5)
    assert f3 == pytest.approx(0.3, abs=1e-5)
    assert f4 == pytest.approx(0.8, abs=1e-5)
    assert f5 == pytest.approx(1, abs=1e-5)

    with pytest.raises(ValueError):
        calculate_total_filtered_fraction(s0, s0 * 1.1)


def test_get_mean_energy_spectrum():
    s1 = np.array([1, 1, 1, 1])
    s2 = np.array([1, 1, 1, 1000])
    s3 = np.array([1, 3, 2, 1])
    s4 = np.array([0, 0, 0, 0])

    b = np.array([1, 2, 3, 4])

    assert get_mean_energy_spectrum(s1, b) == pytest.approx(2.5, abs=1e-4)
    assert get_mean_energy_spectrum(s2, b) == pytest.approx(3.994, abs=1e-4)
    assert get_mean_energy_spectrum(s3, b) == pytest.approx(2.42857, abs=1e-4)
    assert get_mean_energy_spectrum(s4, b) == 0


MATERIAL = "Aluminum"
TUBE_VOLTAGES = ["20", "40", "60", "70", "80", "100"]


@pytest.fixture(scope="module")
def spectra_and_mu() -> pl.DataFrame:
    """Every tungsten spectrum joined to Aluminum's mu on a shared energy grid"""
    data = Data()
    mu_curve = data.get_linear_attenuation_curve(MATERIAL)

    return data.df_spectra.join(
        mu_curve, left_on="Energy[keV]", right_on="Energy", validate="1:1"
    )


def test_get_hvl_monochromatic():
    """Single energy test hvl ~= ln(2)/mu"""
    assert get_hvl(np.array([1.0]), np.array([np.log(2)])) == pytest.approx(10.0)
    assert get_hvl(np.array([1.0]), np.array([1.0])) == pytest.approx(6.93147, abs=1e-5)
    assert get_hvl(np.array([1.0]), np.array([10.0])) == pytest.approx(
        0.693147, abs=1e-6
    )


def test_get_hvl_is_half_value(spectra_and_mu):
    """Filtering by the returned HVL must correspond to half of the intensity"""
    mu = spectra_and_mu[MATERIAL].to_numpy().flatten()

    for kv in TUBE_VOLTAGES:
        spectrum = spectra_and_mu[kv].to_numpy().flatten()

        hvl_mm = get_hvl(spectrum, mu)
        filtered = calculate_filtered_spectrum(spectrum, mu, hvl_mm / 10)  # mm -> cm

        assert calculate_total_filtered_fraction(spectrum, filtered) == pytest.approx(
            0.5, abs=1e-4
        )


def test_get_hvl_monotonic_in_tube_voltage(spectra_and_mu):
    """A higher tube voltage gives a harder beam and so a larger HVL"""
    mu = spectra_and_mu[MATERIAL].to_numpy().flatten()

    hvls = [
        get_hvl(spectra_and_mu[kv].to_numpy().flatten(), mu) for kv in TUBE_VOLTAGES
    ]

    assert all(hvl > 0 for hvl in hvls)
    assert hvls == sorted(hvls)


def test_get_hvl_reference_value(spectra_and_mu):
    """Test for data change"""
    mu = spectra_and_mu[MATERIAL].to_numpy().flatten()
    spectrum = spectra_and_mu["40"].to_numpy().flatten()

    assert get_hvl(spectrum, mu) == pytest.approx(0.0786, abs=1e-3)


def test_get_hvl_edge_cases():
    mu = np.array([1.0, 2.0, 3.0])

    assert get_hvl(np.zeros(3), mu) == 0.0

    # This should never happen if the data does not change
    with pytest.raises(ValueError):
        get_hvl(np.array([1.0, 1.0, 1.0]), np.zeros(3))


def test_get_effective_energy(spectra_and_mu):
    """Test calculation of effective energy.
    Manual HVL and given HVL (independent from spectrum)

    Args:
        spectra_and_mu (_type_): _description_
    """
    spectrum_bins = spectra_and_mu["Energy[keV]"].to_numpy().flatten()
    spectrum = spectra_and_mu["40"].to_numpy().flatten()
    mu = spectra_and_mu[MATERIAL].to_numpy().flatten()
    hvl = 1.75
    eeff = get_effective_energy(spectrum, spectrum_bins, mu, hvl)

    assert eeff == pytest.approx(27.25, abs=1e-3)
    eeff = get_effective_energy(spectrum, spectrum_bins, mu)
    assert eeff == pytest.approx(9.35, abs=1e-3)
