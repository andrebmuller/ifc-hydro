# ifc-hydro

A Python library for hydraulic system analysis of IFC (Industry Foundation Classes) building models. ifc-hydro provides tools for analyzing cold water supply systems in building information models, including topology creation, property extraction, and hydraulic calculations.

## Features

- **Topology Analysis**: Create graph representations of hydraulic system connections from IFC models
- **Property Extraction**: Extract geometric and type properties from pipes, fittings, and valves
- **Hydraulic Calculations**: Perform flow analysis, pressure drop calculations, and available pressure determination
- **IFC Integration**: Direct integration with IFC models using IfcOpenShell
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
from ifc_hydro import Base, TopologyCreator, HydroCalculator

# Configure logging
Base.configure_log(Base, log_dir="", log_name="my-analysis")

# Load IFC model
model = ifc.open('your_model.ifc')

# Create topology from the model
topology = TopologyCreator(model)
all_paths = topology.all_paths_finder()

# Initialize hydraulic calculator
hydro_calc = HydroCalculator()

# Calculate available pressure at a terminal
terminal = model.by_id(5423)  # Replace with actual terminal ID
available_pressure = hydro_calc.available_pressure(terminal, all_paths)
print(f"Available pressure: {available_pressure:.2f} m")
```

For a complete working example, see `example.py` in the repository root.

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
# Calculate linear pressure drop in a pipe
pipe = model.by_id(5399)
pipe_pressure_drop = hydro_calc.linear_pressure_drop(pipe, all_paths)
print(f"Pipe pressure drop: {pipe_pressure_drop:.3f} m")

# Calculate local pressure drop in fittings/valves (requires path context)
fitting = model.by_id(7020)
for path in all_paths:
    if fitting in path:
        fitting_pressure_drop = hydro_calc.local_pressure_drop(fitting, path, all_paths)
        print(f"Fitting pressure drop: {fitting_pressure_drop:.3f} m")
        break
```

## Classes

### `Base`
Base class providing logging functionality and common utilities.

- `configure_log(log_dir, log_name)`: Configure log file location and name
- `append_log(text)`: Append timestamped message to log
- `resource_path(relative_path)`: Get absolute path to resources

### `TopologyCreator`
Creates hydraulic system topology from IFC models.

**Constructor**: `TopologyCreator(model)` - Requires an opened IFC model

- `graph_creator()`: Creates a graph representation of the system
- `path_finder(term_guid, tank_guid)`: Finds path between specific terminal and tank
- `all_paths_finder()`: Finds all paths from terminals to tanks

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

### `Valve`
Extracts properties from IFC valves.

- `properties(valv, path)`: Extracts dimensions and type from valves (requires the path)

### `HydroCalculator`
Performs hydraulic calculations.

- `flow(all_paths)`: Calculates design flow for all components
- `linear_pressure_drop(pipe, all_paths)`: Calculates linear pressure drop in pipes
- `local_pressure_drop(conn, path, all_paths)`: Calculates local pressure drop in connections
- `available_pressure(term, all_paths)`: Calculates available pressure at terminals

### `Graph`
Graph data structure for representing system topology.

- `add(node1, node2)`: Adds connection between nodes
- `remove(node)`: Removes node from graph
- `is_connected(node1, node2)`: Checks if nodes are connected
- `find_path(node1, node2)`: Finds path between nodes (depth-first, not guaranteed shortest)

## Hydraulic Calculation Methods

### Flow Calculations
Uses standardized design flow rates:
- Shower: 0.2 L/s
- Wash basin: 0.15 L/s
- WC seat: 0.15 L/s

### Pressure Drop Calculations
- **Linear losses**: Fair Whipple-Hsiao empirical relation (applied here for PVC pipes)
- **Local losses**: Equivalent length method with tabulated coefficients
- **Available pressure**: Gravity potential minus total pressure losses

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
│   │   └── topology_creator.py     # TopologyCreator class
│   ├── properties/                 # Property extraction module
│   │   ├── __init__.py
│   │   ├── pipe.py                 # Pipe property extraction
│   │   ├── fitting.py              # Fitting property extraction
│   │   └── valve.py                # Valve property extraction
│   ├── hydraulics/                 # Hydraulic calculations module
│   │   ├── __init__.py
│   │   └── hydro_calculator.py     # HydroCalculator class
│   └── examples/                   # Example scripts
│       └── basic_usage.py          # Interactive example
├── ifc-hydro-main.py               # Legacy monolithic script (deprecated)
├── example.py                      # Simple usage example
├── setup.py                        # Package setup configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── projeto-demonstracao.ifc        # Sample IFC model
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

- **1.0.0** - First version with Hazen-Williams formula implementation.
- **2.0.0** - Version with improved hydraulic calculations (Fair Whipple-Hsiao).
- **3.0.0** - Major refactoring into a library structure:
  - Separated implementation into modular package (`ifc_hydro`)
  - Organized code into logical modules: `core`, `topology`, `properties`, `hydraulics`
  - Added `setup.py` for proper package installation
  - Updated `TopologyCreator` to accept IFC model as parameter (no longer hardcoded)
  - Created example scripts demonstrating library usage
  - Improved documentation with updated usage examples

## Support

For questions and support, please open an issue on the GitHub repository or send an e-mail to andre@abm.eng.br

## Acknowledgments

- Built for my Master's degree at the Polytechnic School of Universidade de São Paulo