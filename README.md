# ifc-hydro

A Python library for hydraulic system analysis of IFC (Industry Foundation Classes) building models. ifc-hydro provides tools for analyzing cold water supply systems in building information models, including topology creation, property extraction, and hydraulic calculations.

## Features

- **Topology Analysis**: Create graph representations of hydraulic system connections from IFC models
- **Property Extraction**: Extract geometric and type properties from pipes, fittings, and valves
- **Hydraulic Calculations**: Perform flow analysis, pressure drop calculations, and available pressure determination
- **IFC Geometry Operations**: Extract elevations, bounding boxes, and spatial centers from IFC elements
- **Graph Visualization**: Visualize hydraulic system topology with color-coded components using matplotlib and networkx

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

#### Geometry Operations

```python
from ifc_hydro.core.geom import Geom

# Get elevation of a pipe segment
pipe = model.by_id(5399)
top_z = Geom.get_top_elevation(pipe)
bot_z = Geom.get_bottom_elevation(pipe)
print(f"Pipe elevation range: {bot_z:.3f} m to {top_z:.3f} m")

# Get bounding box center of a fitting
fitting = model.by_id(7020)
cx, cy, cz = Geom.get_bbox_center(fitting)
print(f"Fitting center: ({cx}, {cy}, {cz})")

# Get full bounding box
min_x, min_y, min_z, max_x, max_y, max_z = Geom.get_bbox(fitting)
print(f"Bbox: ({min_x}, {min_y}, {min_z}) to ({max_x}, {max_y}, {max_z})")
```

#### Graph Visualization

```python
from ifc_hydro import Topology, GraphPlotter, Pressure

# Build topology paths first
topology = Topology(model)
all_paths = topology.all_paths_finder()

# Create plotter from topology paths
plotter = GraphPlotter()
plotter.from_topology_paths(all_paths)

# Print summary statistics
plotter.print_statistics()

# Standard topology plot (color-coded by component type)
plotter.plot(layout='hierarchical', title='Cold Water Supply System')

# Save a plot to file without displaying it
plotter.plot(layout='spring', save_path='topology.png', show=False)

# Highlight all paths in distinct colors
plotter.plot_paths(all_paths, layout='hierarchical')

# Color nodes by available pressure values
pressure_values = {}
pressure_calc = Pressure()
for path in all_paths:
    for element in path:
        if element.is_a('IfcSanitaryTerminal'):
            node_id = element.GlobalId
            pressure_values[node_id] = pressure_calc.available(element, all_paths)

plotter.plot_with_data(
    node_values=pressure_values,
    colormap='RdYlGn',
    title='Available Pressure (m)',
    colorbar_label='Pressure (m)'
)
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
- `local(conn, path, all_paths)`: Calculates local pressure drop in fittings and valves using the equivalent length method; uses multi-diameter coefficient tables indexed by nominal diameter with angle-aware junction analysis

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

### `Geom`
Provides geometric operations for IFC elements using IfcOpenShell geometry processing.

- `create_settings(use_world_coords=True)`: Create geometry settings; when `use_world_coords=True` all coordinates are returned in world (absolute) space
- `create_shape(element, settings=None)`: Create a geometry shape from an IFC element
- `get_top_elevation(element, settings=None)`: Get the maximum Z coordinate of an element; returns a float rounded to 3 decimal places
- `get_bottom_elevation(element, settings=None)`: Get the minimum Z coordinate of an element; returns a float rounded to 3 decimal places
- `get_bbox_center(element, settings=None)`: Get the bounding box center as an `(x, y, z)` tuple; handles Triangulation geometry objects
- `get_bbox(element, settings=None)`: Get the full bounding box as `(min_x, min_y, min_z, max_x, max_y, max_z)` tuple

### `GraphPlotter`
Visualizes hydraulic system topology graphs using matplotlib and networkx.

**Constructor**: `GraphPlotter(graph=None)` - Optionally accepts an ifc_hydro `Graph` object to convert immediately

- `from_ifc_graph(graph)`: Convert an ifc_hydro `Graph` object to internal networkx format
- `from_topology_paths(all_paths)`: Build the graph from `Topology.all_paths_finder()` output
- `plot(figsize, layout, color_by_type, show_labels, node_size, font_size, title, highlight_path, save_path, show)`: Render a standard topology view; nodes are color-coded by IFC component type
- `plot_with_data(node_values, colormap, figsize, layout, show_labels, node_size, font_size, title, colorbar_label, save_path, show)`: Render nodes colored by numeric values (e.g., flow rate or pressure) using a continuous colorbar
- `plot_paths(paths, figsize, layout, show_labels, node_size, font_size, title, save_path, show)`: Highlight multiple paths in distinct colors, with a per-path legend
- `get_statistics()`: Returns a dict with `num_nodes`, `num_edges`, `node_types` (counts per IFC type), `is_connected`, `num_components`
- `print_statistics()`: Print the statistics to the console in a formatted block

**Layout algorithms** (`layout` parameter): `spring` (default), `kamada_kawai`, `circular`, `shell`, `spectral`, `hierarchical` (tanks at top, terminals at bottom)

**Node color scheme:**
- `IfcSanitaryTerminal`: Green · `IfcPipeSegment`: Blue · `IfcPipeFitting`: Orange
- `IfcValve`: Red · `IfcTank`: Purple · `IfcFlowController`: Brown

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
- **Local losses**: Equivalent length method with tabulated coefficients indexed by nominal diameter (0.015–0.100 m); values below are for 25 mm nominal diameter:
  - Junction at 0°/180° (straight-through): 0.9 m
  - Junction at 90°: 3.1 m
  - Bend 90°: 1.5 m
  - Bend 45°: 0.7 m
  - Exit: 1.2 m
  - Entry: 0.4 m
  - Isolating valve: 0.2 m
  - Regulating valve: 8.2 m
  - Check valve: 4.1 m
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
│   │   ├── vector.py               # Vector operations for 3D calculations
│   │   └── geom.py                 # IFC geometry operations (elevations, bounding boxes)
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
│   ├── visualization/              # Visualization module
│   │   ├── __init__.py
│   │   └── graph_plotter.py        # Graph plotting with matplotlib and networkx
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
- matplotlib >= 3.5.0
- networkx >= 2.6.0
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

- **1.0.0** (2024) - First version; basic hydraulic calculations for IFC models using Hazen-Williams formula; monolithic script architecture

- **2.0.0** (2024) - Switched to Fair Whipple-Hsiao equation for improved pressure drop accuracy in PVC pipes

- **3.0.0** (2025) - Added angle-aware junction analysis with direction change detection; comprehensive error handling; model-as-parameter design

- **4.0.0** (2026) - Major refactoring into modular library structure (`core`, `topology`, `properties`, `hydraulics`); split classes into `Topology`, `Pipe`, `Fitting`, `Valve`, `DesignFlow`, `PressureDrop`, `Pressure`; added pip installation support and interactive example scripts

- **4.1.0** (2026) - Added `Geom` class for IFC geometry operations and `GraphPlotter` for topology visualization; multi-diameter pressure drop tables indexed by nominal diameter; bug fixes for circular imports and example scripts

## Support

For questions and support, please open an issue on the GitHub repository or send an e-mail to andre@abm.eng.br

## Acknowledgments

Developed as part of MSc research at the Polytechnic School of Universidade de São Paulo. Results are published in: [Müller, A.B. et al. — SBTIC 2024](https://eventos.antac.org.br/index.php/sbtic/article/view/7418)