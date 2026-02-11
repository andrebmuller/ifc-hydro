"""
Input tables for hydraulic calculations.

This module contains reference tables used in hydraulic calculations,
including pressure drop coefficients and design flow rates.
"""

# Equivalent length factors for different connection types
# JUNCTION and BEND use dict: {angle_in_degrees: coefficient}
local_pressure_drop_table = {
    'JUNCTION': {0: 0.9, 180: 0.9, 90: 2.4},
    'BEND': {90: 1.2, 45: 0.7},
    'EXIT': 1.2,
    'ISOLATING': 0.2,
    'REGULATING': 11.4
}

# Design flow rates by sanitary terminal type (L/s)
design_flow_table = {
    'BATH': 0.3,
    'BIDET': 0.1,
    'CISTERN': 0.25,
    'SANITARYFOUNTAIN': 0.2,
    'SHOWER': 0.2,
    'SINK': 0.25,
    'TOILETPAN': 0.15,
    'URINAL': 0.5,
    'WASHHANDBASIN': 0.15,
    'WCSEAT': 0.15,
    'USERDEFINED': 0.3,
    'NOTDEFINED': 0.2
}

# Tank height adjustment (meters)
# Accounts for water level above the pipe connection point at the tank bottom
# This value represents the static head contribution from the water column
# Inside the tank, measured from the outlet pipe connection to the water surface
tank_height_adjustment = 0.5
