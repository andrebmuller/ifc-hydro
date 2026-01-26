"""
Constants and lookup tables for ifc-hydro.

This module contains centralized constants used across the package,
including diameter mappings to avoid circular import issues.
"""

# Supported nominal diameters (in mm)
SUPPORTED_DIAMETERS = [15, 20, 25, 32, 40, 50, 60, 75, 85, 100, 110]

# Nominal diameter to real internal diameter mapping (in meters)
# Based on PVC pipe specifications
NOMINAL_TO_INTERNAL_DIAMETER = {
    15: 0.0170,
    20: 0.0216,
    25: 0.0278,
    32: 0.0352,
    40: 0.0440,
    50: 0.0534,
    60: 0.0600,
    65: 0.0666,
    75: 0.0756,
    85: 0.0850,
    100: 0.0978,
    110: 0.1080,
}


def get_nominal_diameter(measured_diameter_m: float) -> int:
    """
    Map a measured diameter (in meters) to the nearest nominal diameter (in mm).

    Args:
        measured_diameter_m: Measured diameter in meters

    Returns:
        int: Nominal diameter in mm (one of SUPPORTED_DIAMETERS)
    """
    measured_mm = measured_diameter_m * 1000  # Convert to mm

    # Find the closest nominal diameter
    closest = min(SUPPORTED_DIAMETERS, key=lambda d: abs(d - measured_mm))
    return closest


def get_internal_diameter(nominal_diameter_mm: int) -> float:
    """
    Get the real internal diameter for a given nominal diameter.

    Args:
        nominal_diameter_mm: Nominal diameter in mm

    Returns:
        float: Internal diameter in meters, or 0.0 if not found
    """
    return NOMINAL_TO_INTERNAL_DIAMETER.get(nominal_diameter_mm, 0.0)
