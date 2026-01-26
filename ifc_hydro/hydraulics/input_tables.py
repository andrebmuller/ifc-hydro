"""
Input tables for hydraulic calculations.

This module contains reference tables used in hydraulic calculations,
including pressure drop coefficients and design flow rates.

Multi-diameter support: Coefficient tables are organized by nominal diameter
(in mm) to provide accurate pressure drop calculations for different pipe sizes.
"""

from ..core.constants import (
    SUPPORTED_DIAMETERS,
    NOMINAL_TO_INTERNAL_DIAMETER as nominal_to_internal_diameter,
    get_nominal_diameter,
    get_internal_diameter,
)

# Equivalent length factors for fittings by nominal diameter (in mm)
# Structure: {diameter: {fitting_type: coefficient}}
# For BEND: {diameter: {angle: coefficient}} where angle is 45 or 90
# For JUNCTION: {diameter: (junction_0, junction_90)}
fitting_coefficients = {
    15: {
        'BEND': {45: 0.40, 90: 1.00},
        'EXIT': 0.20,
        'JUNCTION': (0.70, 1.00),
    },
    20: {
        'BEND': {45: 0.50, 90: 1.10},
        'EXIT': 0.30,
        'JUNCTION': (0.80, 2.30),
    },
    25: {
        'BEND': {45: 0.70, 90: 1.20},
        'EXIT': 0.40,
        'JUNCTION': (0.90, 2.40),
    },
    32: {
        'BEND': {45: 0.90, 90: 1.50},
        'EXIT': 0.50,
        'JUNCTION': (1.20, 3.10),
    },
    40: {
        'BEND': {45: 1.00, 90: 2.00},
        'EXIT': 0.60,
        'JUNCTION': (1.50, 4.60),
    },
    50: {
        'BEND': {45: 1.30, 90: 3.20},
        'EXIT': 0.80,
        'JUNCTION': (2.20, 7.30),
    },
    60: {
        'BEND': {45: 1.50, 90: 3.40},
        'EXIT': 1.00,
        'JUNCTION': (2.30, 7.60),
    },
    75: {
        'BEND': {45: 1.70, 90: 3.70},
        'EXIT': 1.20,
        'JUNCTION': (2.40, 7.80),
    },
    85: {
        'BEND': {45: 1.80, 90: 3.90},
        'EXIT': 1.50,
        'JUNCTION': (2.50, 8.00),
    },
    100: {
        'BEND': {45: 1.90, 90: 4.10},
        'EXIT': 2.00,
        'JUNCTION': (2.60, 8.20),
    },
    110: {
        'BEND': {45: 2.00, 90: 4.30},
        'EXIT': 2.50,
        'JUNCTION': (2.70, 8.40),
    },
}

# Equivalent length factors for valves by nominal diameter (in mm)
# Structure: {diameter: {valve_type: coefficient}}
valve_coefficients = {
    15: {'ISOLATING': 0.10, 'REGULATING': 4.90},
    20: {'ISOLATING': 0.10, 'REGULATING': 6.70},
    25: {'ISOLATING': 0.20, 'REGULATING': 8.20},
    32: {'ISOLATING': 0.20, 'REGULATING': 11.30},
    40: {'ISOLATING': 0.30, 'REGULATING': 13.40},
    50: {'ISOLATING': 0.40, 'REGULATING': 17.40},
    60: {'ISOLATING': 0.40, 'REGULATING': 21.00},
    75: {'ISOLATING': 0.50, 'REGULATING': 26.00},
    85: {'ISOLATING': 0.60, 'REGULATING': 30.00},
    100: {'ISOLATING': 0.70, 'REGULATING': 34.00},
    110: {'ISOLATING': 0.90, 'REGULATING': 43.00},
}

# Legacy flat table (kept for backwards compatibility, uses 25mm reference)
# DEPRECATED: Use fitting_coefficients and valve_coefficients instead
local_pressure_drop_table = {
    'JUNCTION': (0.9, 2.4),  # (straight flow, flow with direction change)
    'BEND': {45: 0.7, 90: 1.2},  # angle-based: 45° bends = 0.7, 90° bends = 1.2
    'EXIT': 0.4,
    'ISOLATING': 0.2,
    'REGULATING': 8.2
}

# Design flow rates by sanitary terminal type (L/s)
design_flow_table = {
    'BATH': 0.3,
    'BIDET': 0.1,
    'CISTERN': 0.15,
    'SANITARYFOUNTAIN': 0.25,
    'SHOWER': 0.2,
    'SINK': 0.25,
    'TOILETPAN': 1.70,
    'URINAL': 0.5,
    'WASHHANDBASIN': 0.15,
    'WCSEAT': 1.70,          # Deprecated, use TOILETPAN instead
    'USERDEFINED': 0.3
}


def get_fitting_coefficient(fitting_type: str, nominal_diameter_mm: int, angle: float = None) -> float:
    """
    Get the equivalent length coefficient for a fitting based on diameter.

    Args:
        fitting_type: Type of fitting ('BEND', 'EXIT', 'JUNCTION')
        nominal_diameter_mm: Nominal diameter in mm
        angle: Direction change angle in degrees (required for BEND and JUNCTION)

    Returns:
        float: Equivalent length coefficient
    """
    # Get diameter-specific coefficients, fallback to 25mm if not found
    diameter_key = nominal_diameter_mm if nominal_diameter_mm in fitting_coefficients else 25
    coefficients = fitting_coefficients[diameter_key]

    if fitting_type == 'BEND':
        bend_coeffs = coefficients.get('BEND', {45: 0.7, 90: 1.2})
        # Use threshold of 67.5° (midpoint between 45° and 90°) to select coefficient
        if angle is not None and angle < 67.5:
            return bend_coeffs.get(45, 0.7)
        else:
            return bend_coeffs.get(90, 1.2)

    elif fitting_type == 'JUNCTION':
        junction_coeffs = coefficients.get('JUNCTION', (0.9, 2.4))
        # Select coefficient based on angle: index 0 for 0.0°, index 1 for others
        if angle == 0.0:
            return junction_coeffs[0]
        else:
            return junction_coeffs[1]

    elif fitting_type == 'EXIT':
        return coefficients.get('EXIT', 0.4)

    # Fallback for unknown fitting types
    return local_pressure_drop_table.get(fitting_type, 0.0)


def get_valve_coefficient(valve_type: str, nominal_diameter_mm: int) -> float:
    """
    Get the equivalent length coefficient for a valve based on diameter.

    Args:
        valve_type: Type of valve ('ISOLATING', 'REGULATING')
        nominal_diameter_mm: Nominal diameter in mm

    Returns:
        float: Equivalent length coefficient
    """
    # Get diameter-specific coefficients, fallback to 25mm if not found
    diameter_key = nominal_diameter_mm if nominal_diameter_mm in valve_coefficients else 25
    coefficients = valve_coefficients[diameter_key]

    return coefficients.get(valve_type, local_pressure_drop_table.get(valve_type, 0.0))
