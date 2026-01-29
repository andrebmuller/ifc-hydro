"""
Input tables for hydraulic calculations.

This module contains reference tables used in hydraulic calculations,
including pressure drop coefficients and design flow rates.
"""

# Equivalent length factors for different connection types
# JUNCTION uses tuple: (0.9 for 0° angle, 2.4 for other angles)
local_pressure_drop_table = {
    'JUNCTION': (0.9, 2.4),  # (straight flow, flow with direction change)
    'BEND': 1.2,
    'EXIT': 1.2,
    'ISOLATING': 0.2,
    'REGULATING': 11.4
}

# Design flow rates by sanitary terminal type (L/s)
design_flow_table = {
    'SHOWER': 0.2,
    'WASHHANDBASIN': 0.15,
    'WCSEAT': 0.15
}

# Tank height adjustment (meters)
# Accounts for water level above the pipe connection point at the tank bottom.
# This value represents the static head contribution from the water column
# inside the tank, measured from the outlet pipe connection to the water surface.
tank_height_adjustment = 0.5
