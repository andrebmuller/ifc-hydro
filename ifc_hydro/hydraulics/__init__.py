"""
Hydraulics module for hydraulic calculations.

Includes multi-diameter support with diameter-specific equivalent length
coefficients for fittings and valves.
"""

from .design_flow import DesignFlow
from .pressure_drop import PressureDrop
from .pressure import Pressure
from .input_tables import (
    SUPPORTED_DIAMETERS,
    fitting_coefficients,
    valve_coefficients,
    get_nominal_diameter,
    get_internal_diameter,
    get_fitting_coefficient,
    get_valve_coefficient,
)

__all__ = [
    "DesignFlow",
    "PressureDrop",
    "Pressure",
    # Multi-diameter support
    "SUPPORTED_DIAMETERS",
    "fitting_coefficients",
    "valve_coefficients",
    "get_nominal_diameter",
    "get_internal_diameter",
    "get_fitting_coefficient",
    "get_valve_coefficient",
]
