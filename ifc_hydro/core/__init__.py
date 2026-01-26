"""
Core module containing base classes and data structures.
"""

from .base import Base
from .graph import Graph
from .vector import Vector
from ..hydraulics.input_tables import (
    SUPPORTED_DIAMETERS,
    NOMINAL_TO_INTERNAL_DIAMETER,
    get_nominal_diameter,
    get_internal_diameter,
)

__all__ = [
    "Base",
    "Graph",
    "Vector",
    # Diameter constants (re-exported from hydraulics.input_tables)
    "SUPPORTED_DIAMETERS",
    "NOMINAL_TO_INTERNAL_DIAMETER",
    "get_nominal_diameter",
    "get_internal_diameter",
]
