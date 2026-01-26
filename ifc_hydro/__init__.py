"""
ifc-hydro - Hydraulic system analysis for IFC models.

This package provides hydraulic system analysis capabilities for IFC (Industry Foundation Classes) models.
It includes functionality for topology creation, property calculation, and hydraulic calculations
for building water supply and drainage systems.

Version: 3.1.0
"""

from .core.base import Base
from .core.graph import Graph
from .core.vector import Vector
from .topology.topology import Topology
from .properties.pipe import Pipe
from .properties.fitting import Fitting
from .properties.valve import Valve
from .hydraulics.design_flow import DesignFlow
from .hydraulics.pressure_drop import PressureDrop
from .hydraulics.pressure import Pressure
from .hydraulics.input_tables import (
    SUPPORTED_DIAMETERS,
    get_nominal_diameter,
    get_internal_diameter,
    get_fitting_coefficient,
    get_valve_coefficient,
)

__version__ = "3.1.0"
__all__ = [
    "Base",
    "Graph",
    "Vector",
    "Topology",
    "Pipe",
    "Fitting",
    "Valve",
    "DesignFlow",
    "PressureDrop",
    "Pressure",
    # Multi-diameter support
    "SUPPORTED_DIAMETERS",
    "get_nominal_diameter",
    "get_internal_diameter",
    "get_fitting_coefficient",
    "get_valve_coefficient",
]
