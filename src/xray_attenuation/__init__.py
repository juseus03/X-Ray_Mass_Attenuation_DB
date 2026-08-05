"""X-ray transmission and mass attenuation for NIST elements and compounds."""

from xray_attenuation.cli import CLI, Filter
from xray_attenuation.data import Data, MaterialNotFoundError

__all__ = ["Data", "MaterialNotFoundError", "CLI", "Filter"]
__version__ = "0.1.0"
