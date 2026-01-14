"""
Simple example demonstrating ifc-hydro library usage.

This script shows how to use ifc-hydro as a library to perform
hydraulic calculations on IFC models.
"""

import ifcopenshell as ifc
from ifc_hydro import Base, TopologyCreator, PropCalculator, HydroCalculator


# Configure logging
Base.configure_log(Base, log_dir="", log_name="example-run")

# Load IFC model
model = ifc.open('projeto-demonstracao.ifc')

# Create topology from the model
topology = TopologyCreator(model)
all_paths = topology.all_paths_finder()

# Initialize calculators
prop_calc = PropCalculator()
hydro_calc = HydroCalculator()

# Calculate available pressure at a terminal (Wash and Basin - ID: 6986)
terminal = model.by_id(6986)
available_pressure = hydro_calc.available_pressure(terminal, all_paths)

print(f"\n{'='*60}")
print(f"Available pressure at terminal {terminal.id()}: {round(available_pressure, 3)} m")
print(f"{'='*60}\n")
