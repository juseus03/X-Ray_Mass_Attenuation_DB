from cli import get_transmission
import pytest


def test_get_transmission():
    assert get_transmission(0, 0.1) == 1
    assert get_transmission(1.0, 0.5) == pytest.approx(0.6065, abs=1e-4)
