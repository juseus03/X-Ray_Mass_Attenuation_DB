import pytest
import numpy as np
from xray_attenuation.physics import get_transmission, calculate_filtered_spectrum


def test_get_transmission():
    assert get_transmission(0, 0.1) == 1
    assert get_transmission(1.0, 0.5) == pytest.approx(0.6065, abs=1e-4)


def test_calculate_filtered_spectrum():
    spectrum = np.array([[10, 100], [20, 200], [30, 300]])
    mu = np.array([0.1, 0.2, 0.3])
    thickness = 1.0

    filtered_spectrum = calculate_filtered_spectrum(spectrum, mu, thickness)

    expected_intensity = [
        100 * get_transmission(0.1, thickness),
        200 * get_transmission(0.2, thickness),
        300 * get_transmission(0.3, thickness),
    ]

    for i, fsp in enumerate(filtered_spectrum):
        assert fsp[0] == spectrum[i][0]
        assert fsp[1] == pytest.approx(expected_intensity[i], abs=1e-4)
