# ifc-hydro

A Python library for hydraulic system analysis of IFC (Industry Foundation Classes) building models. ifc-hydro provides tools for analyzing cold water supply systems in building information models, including topology creation, property extraction, and hydraulic calculations.

## Features

- **Topology Analysis**: Create graph representations of hydraulic system connections from IFC models
- **Property Extraction**: Extract geometric and type properties from pipes, fittings, and valves
- **Hydraulic Calculations**: Perform flow analysis, pressure drop calculations, and available pressure determination
- **Angle-Aware Analysis**: Direction change detection in JUNCTION fittings for accurate pressure drop calculations
- **Comprehensive Error Handling**: Validates IFC elements and provides meaningful error messages
- **IFC Integration**: Direct integration with IFC models using IfcOpenShell
- **Centralized Reference Data**: Lookup tables for hydraulic coefficients and design flow rates
- **Logging**: Built-in logging system for debugging and analysis tracking

## Installation

### From source

```bash
git clone https://github.com/andrebmuller/ifc-hydro.git
cd ifc-hydro
pip install -e .
```

### Prerequisites

- Python 3.7 or higher
- IfcOpenShell library (installed automatically with setup)

## Usage

### Basic Example

```python
import ifcopenshell as ifc
from ifc_hydro import Base, Topology, Pressure
import sys

# Configure logging
Base.configure_log(Base, log_dir="logs", log_name="my-analysis")

# Load IFC model with error handling
try:
    model = ifc.open('your_model.ifc')
    Base.append_log(Base, f"Successfully loaded IFC model")
except Exception as e:
    print(f"ERROR: Failed to open IFC file: {e}")
    sys.exit(1)

# Create topology from the model with error handling
try:
    topology = Topology(model)
    all_paths = topology.all_paths_finder()
except ValueError as e:
    print(f"ERROR: Topology creation failed: {e}")
    sys.exit(1)

# Initialize pressure calculator
pressure_calc = Pressure()

# Calculate available pressure for all terminals
terminals = model.by_type("IfcSanitaryTerminal")
for terminal in terminals:
    try:
        available_pressure = pressure_calc.available(terminal, all_paths)
        print(f"Terminal {terminal.id()}: {available_pressure:.2f} m")
    except Exception as e:
        print(f"Warning: Failed to calculate pressure for terminal {terminal.id()}: {e}")
        continue
```

For complete working examples with interactive prompts, see:
- `ifc_hydro/examples/demo/demo.py` - Interactive demonstration script
- `ifc_hydro/examples/eval/eval.py` - Evaluation example script

### Advanced Usage

#### Property Extraction

```python
from ifc_hydro import Pipe, Fitting, Valve

# Extract pipe properties
pipe = model.by_id(5399)  # Replace with actual pipe ID
pipe_props = Pipe.properties(pipe)
print(f"Pipe length: {pipe_props['len']} m")
print(f"Pipe diameter: {pipe_props['dim']} m")

# Extract fitting properties (requires path context)
fitting = model.by_id(7020)  # Replace with actual fitting ID
# Find the path containing this fitting
for path in all_paths:
    if fitting in path:
        fitting_props = Fitting.properties(fitting, path)
        print(f"Fitting type: {fitting_props['type']}")
        break
```

#### Hydraulic Calculations

```python
from ifc_hydro import DesignFlow, PressureDrop, Pressure

# Calculate design flow for all components
flow_calc = DesignFlow()
flows = flow_calc.calculate(all_paths)

# Calculate linear pressure drop in a pipe
pressure_drop_calc = PressureDrop()
pipe = model.by_id(5399)
pipe_pressure_drop = pressure_drop_calc.linear(pipe, all_paths)
print(f"Pipe pressure drop: {pipe_pressure_drop:.3f} m")

# Calculate local pressure drop in fittings/valves (requires path context)
fitting = model.by_id(7020)
for path in all_paths:
    if fitting in path:
        fitting_pressure_drop = pressure_drop_calc.local(fitting, path, all_paths)
        print(f"Fitting pressure drop: {fitting_pressure_drop:.3f} m")
        break

# Calculate available pressure at a terminal
pressure_calc = Pressure()
terminal = model.by_id(5423)
available_pressure = pressure_calc.available(terminal, all_paths)
print(f"Available pressure: {available_pressure:.2f} m")
```

## Classes

### `Base`
Base class providing logging functionality and common utilities.

- `configure_log(log_dir, log_name)`: Configure log file location and name
- `append_log(text)`: Append timestamped message to log
- `resource_path(relative_path)`: Get absolute path to resources

### `Topology`
Creates hydraulic system topology from IFC models.

**Constructor**: `Topology(model)` - Requires an opened IFC model

- `graph_creator()`: Creates a graph representation of the system using IfcRelNests and IfcRelConnectsPorts
- `path_finder(term_guid, tank_guid)`: Finds path between specific terminal and tank
- `all_paths_finder()`: Finds all paths from terminals to tanks
- **Error Handling**: Validates required IFC elements exist before processing:
  - Checks for IfcSanitaryTerminal elements
  - Checks for IfcTank elements
  - Checks for IfcRelNests and IfcRelConnectsPorts relationships
  - Raises descriptive ValueError exceptions if critical elements are missing

### `Vector`
Provides 3D vector operations for geometric calculations.

- `create_direction_vector(from_point, to_point)`: Creates a direction vector between two points
- `magnitude(vector)`: Calculates the magnitude (length) of a vector
- `normalize(vector)`: Normalizes a vector to unit length
- `dot_product(vector1, vector2)`: Calculates the dot product of two vectors
- `angle_between(vector1, vector2)`: Calculates the angle between two vectors in degrees

### `Pipe`
Extracts properties from IFC pipe segments.

- `properties(pipe)`: Extracts length and diameter from pipe segments

### `Fitting`
Extracts properties from IFC pipe fittings.

- `properties(fitt, path)`: Extracts dimensions, directions, and type from fittings (requires the path)
  - Calculates direction change angle using vector analysis
  - Returns incoming/outgoing direction vectors for flow analysis
  - Validates fitting position in path with error handling

### `Valve`
Extracts properties from IFC valves.

- `properties(valv, path)`: Extracts dimensions and type from valves (requires the path)

### `DesignFlow`
Calculates design flow rates for hydraulic system components.

- `calculate(all_paths)`: Calculates design flow for all components based on terminal types

### `PressureDrop`
Calculates pressure drops in hydraulic system components.

- `linear(pipe, all_paths)`: Calculates linear pressure drop in pipes using Fair Whipple-Hsiao equation
- `local(conn, path, all_paths)`: Calculates local pressure drop in fittings and valves using equivalent length method
  - **Angle-aware junctions**: Automatically detects direction changes and applies appropriate coefficients
    - 0° direction change (straight): coefficient 0.9
    - Other angles: coefficient 2.4
  - Uses standardized reference diameter (25mm) for equivalent length calculations

### `Pressure`
Calculates available pressure at sanitary terminals.

- `available(term, all_paths)`: Calculates available pressure at terminals accounting for gravity potential and all losses
  - Handles multiple IFC representation types (SweptSolid, MappedRepresentation)
  - Provides per-component error handling with detailed logging
  - Returns pressure in meters of water column

### `Graph`
Graph data structure for representing system topology.

- `add(node1, node2)`: Adds connection between nodes
- `remove(node)`: Removes node from graph
- `is_connected(node1, node2)`: Checks if nodes are connected
- `find_path(node1, node2)`: Finds path between nodes (depth-first, not guaranteed shortest)

## Hydraulic Calculation Methods

### Flow Calculations
Uses standardized design flow rates from lookup tables (`input_tables.py`):
- Shower: 0.2 L/s
- Wash basin: 0.15 L/s
- WC seat: 1.70 L/s
- Bath, Bidet, Cistern, Fountain, Sink, Toilet Pan, Urinal, and more

### Pressure Drop Calculations
- **Linear losses**: Fair Whipple-Hsiao empirical equation for PVC pipes
  - Formula: `pressure_drop = pipe_len × (0.000859 × (flow^1.75) × (diameter^-4.75))`
  - Recommended for PVC pipes, diameter range 12.5-100mm
- **Local losses**: Equivalent length method with tabulated coefficients (for 25 mm diameter pipes):
  - Junction (straight): 0.9
  - Junction (with angle): 2.4
  - Bend: 1.2
  - Exit: 1.2
  - Isolating valve: 0.2
  - Regulating valve: 11.4
- **Angle detection**: Automatically calculates direction change in junctions using 3D vector analysis
- **Available pressure**: Gravity potential minus total pressure losses

### Reference Data Module
All hydraulic coefficients and design parameters are centralized in `input_tables.py` for easy maintenance and extension.

## File Structure

```
ifc-hydro/
├── ifc_hydro/                      # Main library package
│   ├── __init__.py                 # Package initialization and exports
│   ├── core/                       # Core classes and data structures
│   │   ├── __init__.py
│   │   ├── base.py                 # Base class with logging utilities
│   │   ├── graph.py                # Graph data structure
│   │   └── vector.py               # Vector operations for 3D calculations
│   ├── topology/                   # Topology creation module
│   │   ├── __init__.py
│   │   └── topology.py             # Topology class with error handling
│   ├── properties/                 # Property extraction module
│   │   ├── __init__.py
│   │   ├── pipe.py                 # Pipe property extraction
│   │   ├── fitting.py              # Fitting property extraction
│   │   └── valve.py                # Valve property extraction
│   ├── hydraulics/                 # Hydraulic calculations module
│   │   ├── __init__.py
│   │   ├── design_flow.py          # Design flow calculations
│   │   ├── pressure_drop.py        # Pressure drop calculations
│   │   ├── pressure.py             # Pressure calculations
│   │   └── input_tables.py         # Centralized input parameters tables
│   └── examples/                   # Example scripts
│       ├── demo/                   # Demo example
│       │   ├── demo.py             # Interactive demonstration script
│       │   └── demo-project.ifc    # Sample IFC model for demo
│       └── eval/                   # Evaluation example
│           ├── eval.py             # Interactive evaluation script
│           └── eval-project.ifc    # Sample IFC model for evaluation
├── old/                            # Legacy versions (1.0.0, 2.0.0, 3.0.0)
├── setup.py                        # Package setup configuration
├── requirements.txt                # Python dependencies
├── LICENSE.md                      # MIT License
└── README.md                       # This file
```

## Requirements

- Python 3.7+
- IfcOpenShell >= 0.7.0
- Standard library modules: `datetime`, `collections`, `sys`, `os`

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Authors

- [André Buchmann Müller](https://andrebmuller.notion.site/abm-eng)

## Version History

- **1.0.0** (2024) - First version with Hazen-Williams formula implementation
  - Basic hydraulic calculations for IFC models
  - Monolithic script architecture

- **2.0.0** (2024) - Improved hydraulic calculations
  - Switched to Fair Whipple-Hsiao equation for PVC pipes
  - Enhanced accuracy for pressure drop calculations

- **3.0.0** (2025) - Added error handling and angle-aware analysis for JUNCTIION fittings
    - Angle-aware junction analysis with direction change detection
    - Comprehensive error handling and validation throughout
    - Model-as-parameter design (no longer hardcoded file paths)
    - Per-component error handling with detailed logging

- **4.0.0** (2026) - Major refactoring and feature additions
  - **Modular library structure**: Separated into `core`, `topology`, `properties`, `hydraulics` modules
  - **Class refactoring**:
    - Renamed `TopologyCreator` → `Topology`
    - Split `PropertyExtractor` into `Pipe`, `Fitting`, and `Valve` classes
    - Split `HydroCalculator` into `DesignFlow`, `PressureDrop`, and `Pressure` classes
    - Added `Vector` and `Graph` utility classes
    - Added 'input_tables.py' for centralized reference data
    - Interactive demo and evaluation examples with sample IFC models
  - **Improvements**:
    - Support for multiple IFC representation types
    - Enhanced property extraction with geometric analysis
    - Multi-diameter support in pressure drop calculations
  - **Package installation**: Added `setup.py` for pip installation

## Support

For questions and support, please open an issue on the GitHub repository or send an e-mail to andre@abm.eng.br

## Acknowledgments

- Built for my Master's degree at the Polytechnic School of Universidade de São Paulo