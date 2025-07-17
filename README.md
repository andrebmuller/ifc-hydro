# IfcHydro

A Python library for hydraulic system analysis of IFC (Industry Foundation Classes) building models. IfcHydro provides tools for analyzing water supply and drainage systems in building information models, including topology creation, property extraction, and hydraulic calculations.

## Features

- **Topology Analysis**: Create graph representations of hydraulic system connections from IFC models
- **Property Extraction**: Extract geometric and type properties from pipes, fittings, and valves
- **Hydraulic Calculations**: Perform flow analysis, pressure drop calculations, and available pressure determination
- **IFC Integration**: Direct integration with IFC models using IfcOpenShell
- **Logging**: Built-in logging system for debugging and analysis tracking

## Installation

### Prerequisites

- Python 3.6 or higher
- IfcOpenShell library

## Usage

### Basic Example

'''
from IfcHydroMain import TopologyCreator, PropCalculator, HydroCalculator import ifcopenshell as ifc
'''

- **Initialize topology creator**:

'''
topology = TopologyCreator()
'''

- **Create system topology graph**:

'''
graph = topology.graph_creator()
'''

- **Find all paths from terminals to tanks**:

'''
all_paths = topology.all_paths_finder()
'''

- **Initialize calculators**:

'''
prop_calc = PropCalculator() hydro_calc = HydroCalculator()
'''

- **Load IFC model**:

'''
model = ifc.open('your_model.ifc')
'''

- **Calculate available pressure at a terminal**:

'''
terminal = model.by_id(5423)  # Replace with actual terminal ID available_pressure = hydro_calc.available_pressure(terminal, all_paths) print(f"Available pressure: {available_pressure:.2f} m")
'''

### Advanced Usage

#### Property Extraction

- **Extract pipe properties**:

'''
pipe = model.by_id(5399)  # Replace with actual pipe ID 
pipe_props = prop_calc.pipe_properties(pipe)
print(f"Pipe length: {pipe_props['len']} m")
print(f"Pipe diameter: {pipe_props['dim']} m")
'''

- **Extract fitting properties**:

'''
fitting = model.by_id(7020)  # Replace with actual fitting ID
fitting_props = prop_calc.fitt_properties(fitting)
print(f"Fitting type: {fitting_props['type']}")
'''

#### Hydraulic Calculations

- **Calculate flow rates throughout the system**:

'''
pipe_pressure_drop = hydro_calc.linear_pressure_drop(pipe, all_paths)
'''

- **Calculate local pressure drop in fittings/valves**:

'''
fitting_pressure_drop = hydro_calc.local_pressure_drop(fitting, all_paths)
'''

## Classes

### `TopologyCreator`
Creates hydraulic system topology from IFC models.

- `graph_creator()`: Creates a graph representation of the system
- `path_finder(term_guid, tank_guid)`: Finds path between specific terminal and tank
- `all_paths_finder()`: Finds all paths from terminals to tanks

### `PropCalculator`
Extracts properties from IFC components.

- `pipe_properties(pipe)`: Extracts length and diameter from pipe segments
- `fitt_properties(fitt)`: Extracts dimensions, directions, and type from fittings
- `valv_properties(valv)`: Extracts dimensions and type from valves

### `HydroCalculator`
Performs hydraulic calculations.

- `flow(all_paths)`: Calculates design flow for all components
- `linear_pressure_drop(pipe, all_paths)`: Calculates linear pressure drop in pipes
- `local_pressure_drop(conn, all_paths)`: Calculates local pressure drop in connections
- `available_pressure(term, all_paths)`: Calculates available pressure at terminals

### `Graph`
Graph data structure for representing system topology.

- `add(node1, node2)`: Adds connection between nodes
- `remove(node)`: Removes node from graph
- `is_connected(node1, node2)`: Checks if nodes are connected
- `find_path(node1, node2)`: Finds path between nodes

## Hydraulic Calculation Methods

### Flow Calculations
Uses standardized design flow rates:
- Shower: 0.2 L/s
- Wash basin: 0.15 L/s
- WC seat: 0.15 L/s

### Pressure Drop Calculations
- **Linear losses**: Fair Whipple-Hsiao equation for PVC pipes
- **Local losses**: Equivalent length method with tabulated coefficients
- **Available pressure**: Gravity potential minus total pressure losses

## File Structure

IfcHydro/ 	
		├── 2.0.0_IfcHydroMain.py    		# Main module with all classes
		├── README.md                		# This file
		├── IfcHydro.log            		# Log file (generated during execution) 
		└── ProjetoRamalBanheiro_Revit.ifc  # Sample IFC model
		
## Requirements

- Python 3.6+
- IfcOpenShell
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

- [André Buchmann Müller] (https://abm.eng.br/)

## Version History

- **1.0.0** - First version with Hazen-Williams formula implementation
- **2.0.0** - Current version with improved hydraulic calculations (Fair Whipple-Hsiao)

## Support

For questions and support, please open an issue on the GitHub repository or send an e-mail to andrebuchmannmuller@gmail.com

## Acknowledgments

- Built for my Master's degree at the Polytechnic School of Universidade de São Paulo