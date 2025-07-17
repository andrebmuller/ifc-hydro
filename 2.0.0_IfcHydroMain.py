"""
IfcHydro Implementation - Main Module

This module provides hydraulic system analysis capabilities for IFC (Industry Foundation Classes) models.
It includes functionality for topology creation, property calculation, and hydraulic calculations
for building water supply and drainage systems.

The module contains the following main classes:
- Base: Base class with logging functionality
- Graph: Graph data structure for representing system topology
- TopologyCreator: Creates hydraulic system topology from IFC models
- PropCalculator: Extracts properties from IFC components
- HydroCalculator: Performs hydraulic calculations

Author: IfcHydro Development Team
Version: 2.0.0
"""

from datetime import datetime as time
from collections import defaultdict
import ifcopenshell as ifc
import sys
import os

class Base:
    """
    Base class providing logging functionality and common utilities.
    
    This class serves as a foundation for other classes in the IfcHydro system,
    providing centralized logging capabilities and resource path management.
    
    Attributes:
        _log (str): Name of the log file (class variable)
        _counter (int): Instance counter for tracking object creation (class variable)
    """
    
    _log     = "IfcHydro.log"        # name of log file
    _counter = 0                     # instance counter

    def __init__(self, log: str = ""):
        """
        Initialize the Base class instance.
        
        Args:
            log (str, optional): Custom log file name. If empty, uses default log file.
        """
        if log != "": 
            Base._log = log

        Base._counter += 1

    def append_log(self, text: str):
        """
        Append a timestamped message to the log file and print to console.
        
        Args:
            text (str): The message to log
        """
        t = time.now()
        tstamp = "%2.2d.%2.2d.%2.2d " % (t.hour, t.minute, t.second)

        otext = tstamp + text
        f = open(Base._log, "a")
        f.write(otext + "\n")
        f.close()
        print(otext)
    
    def resource_path(self, relative_path: str) -> str:
        """
        Retrieve the absolute path of resources used in the application.
        
        This method handles both development and PyInstaller bundled environments.
        
        Args:
            relative_path (str): The relative path to the resource
            
        Returns:
            str: The absolute path to the resource
        """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

class Graph(object):
    """
    Graph data structure for representing hydraulic system topology.
    
    This class implements an undirected graph by default, used to represent
    connections between hydraulic components in the system.
    
    Attributes:
        _graph (defaultdict): Internal graph representation using adjacency lists
        _directed (bool): Flag indicating if the graph is directed
    """

    def __init__(self, connections: list, directed: bool = False):
        """
        Initialize the graph with connections.
        
        Args:
            connections (list): List of tuple pairs representing connections
            directed (bool, optional): Whether the graph is directed. Defaults to False.
        """
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections: list):
        """
        Add multiple connections to the graph.
        
        Args:
            connections (list): List of tuple pairs representing node connections
        """
        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """
        Add a single connection between two nodes.
        
        Args:
            node1: First node to connect
            node2: Second node to connect
        """
        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """
        Remove all references to a node from the graph.
        
        Args:
            node: The node to remove
        """
        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2) -> bool:
        """
        Check if two nodes are directly connected.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            bool: True if nodes are directly connected, False otherwise
        """
        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path: list = []) -> list:
        """
        Find any path between two nodes using depth-first search.
        
        Note: This may not be the shortest path.
        
        Args:
            node1: Starting node
            node2: Destination node
            path (list, optional): Current path being explored. Defaults to [].
            
        Returns:
            list: Path from node1 to node2, or None if no path exists
        """
        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self) -> str:
        """
        String representation of the graph.
        
        Returns:
            str: String representation showing class name and graph structure
        """
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

class TopologyCreator():
    """
    Creates hydraulic system topology from IFC models.
    
    This class analyzes IFC files to extract hydraulic component relationships
    and create a graph representation of the system topology.
    
    Attributes:
        model: The loaded IFC model
    """
    
    def __init__(self):
        """
        Initialize the TopologyCreator with an IFC model.
        
        Loads the default IFC file for hydraulic analysis.
        """
        self.model = ifc.open('ProjetoRamalBanheiro_Revit.ifc')

    def graph_creator(self) -> Graph:
        """
        Create a graph representing the hydraulic system topology.
        
        Analyzes the IFC model to find connections between hydraulic components
        using IfcRelNests and IfcRelConnectsPorts relationships.
        
        Returns:
            Graph: Undirected graph representing the hydraulic system topology
        """
        model = self.model

        Base.append_log(self, "> Model imported...")
        Base.append_log(self, "> Creating topology...")

        connections = []

        # Extract nest and connection relationships from IFC model
        nest_list = model.by_type("IfcRelNests")
        conn_list = model.by_type("IfcRelConnectsPorts")

        # Create connections by analyzing port relationships and nesting
        for conn in conn_list:
            for nest in nest_list:
                for other_nest in nest_list:
                    if conn[4] in nest[5] or conn[5] in nest[5]:
                        if conn[5] in other_nest[5] or conn[4] in other_nest[5]:
                            nest1 = nest
                            nest2 = other_nest
                
                if nest1 != nest2:
                    connections.append((nest1[4], nest2[4]))
        
        graph = Graph(connections)

        return graph
    
    def path_finder(self, term_guid: str, tank_guid: str) -> list:
        """
        Find the path between a specific sanitary terminal and tank.
        
        Args:
            term_guid (str): GUID of the sanitary terminal
            tank_guid (str): GUID of the tank
            
        Returns:
            list: Path from terminal to tank, wrapped in a list
        """
        model = self.model
        graph = self.graph_creator()

        Base.append_log(self, "> Finding path...")
        path = []

        # Find components by GUID and calculate path
        term = model.by_guid(term_guid)
        tank = model.by_guid(tank_guid)

        path.append(graph.find_path(term, tank))
                
        return path

    def all_paths_finder(self) -> list:
        """
        Find paths between all sanitary terminals and tanks in the system.
        
        Returns:
            list: List of all paths from terminals to tanks
        """
        model = self.model
        graph = self.graph_creator()

        Base.append_log(self, "> Finding path...")
        all_paths = []

        # Get all terminals and tanks from the model
        term_list = model.by_type("IfcSanitaryTerminal")
        tank_list = model.by_type("IfcTank")

        # Calculate paths between all terminal-tank combinations
        for term in term_list:
            for tank in tank_list:
                all_paths.append(graph.find_path(term, tank))
                
        return all_paths
    
class PropCalculator():
    """
    Extracts properties from IFC hydraulic system components.
    
    This class provides methods to extract geometric and type properties
    from pipes, fittings, and valves in the IFC model.
    """
    
    def __init__(self) -> None:
        """Initialize the PropCalculator."""
        pass

    def pipe_properties(self, pipe) -> dict:
        """
        Extract properties from a pipe segment.
        
        Args:
            pipe: IFC pipe segment object
            
        Returns:
            dict: Dictionary containing pipe length ('len') and diameter ('dim')
        """
        Base.append_log(self, f"> Getting pipe properties for pipe with ID {pipe.id()}...")
        pipe_prop = {}

        # Extract pipe length from IFC geometry representation
        pipe_len = pipe[6][2][0][3][0][3]
        pipe_prop['len'] = round(pipe_len, 3)

        # Extract pipe diameter (radius * 2) from IFC geometry
        pipe_dim = pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        pipe_prop['dim'] = round(pipe_dim, 3)

        Base.append_log(self, f"> Pipe properties:")
        Base.append_log(self, f"> {pipe_prop}")
        return pipe_prop
    
    def fitt_properties(self, fitt) -> dict:
        """
        Extract properties from a pipe fitting.
        
        Analyzes the fitting's position in the hydraulic network to determine
        dimensions and flow directions.
        
        Args:
            fitt: IFC pipe fitting object
            
        Returns:
            dict: Dictionary containing dimensions ('dim'), directions ('dir'), and type ('type')
        """
        Base.append_log(self, f"> Getting fitting properties for fitting with ID {fitt.id()}...")
        fitt_prop = {}        

        # Get topology and all paths in the system
        topo = TopologyCreator()
        all_paths = topo.all_paths_finder()

        # Create path list with component IDs for easier searching
        all_paths_id = []
        n = 0
        for path in all_paths:
            all_paths_id.append([])
            for item in path:
                all_paths_id[n].append(all_paths[n][all_paths[n].index(item)].id())
            n += 1

        # Find fitting position in each path
        fitt_index = []
        for path in all_paths_id:
            for item in path:
                if item == fitt.id():
                    fitt_index.append(path.index(item))
            if fitt.id() not in path:
                fitt_index.append(None)

        # Get adjacent pipes for each occurrence of the fitting
        pipe_list = []
        start_index = 0
        for item in fitt_index:
            if item is not None:
                pipe_list.append((all_paths[fitt_index.index(item, start_index)][item-1], 
                                all_paths[fitt_index.index(item, start_index)][item], 
                                all_paths[fitt_index.index(item)][item+1]))
                start_index = fitt_index.index(item) + 1
            else:
                pipe_list.append(None)

        # Extract diameters from adjacent pipes
        pipe_dim_list = []
        for item in pipe_list:
            if item is not None:
                pipe_dim_1 = pipe_list[pipe_list.index(item)][0][6][2][0][3][0][0][2][0][0][0][0] * 2
                pipe_dim_2 = pipe_list[pipe_list.index(item)][2][6][2][0][3][0][0][2][0][0][0][0] * 2
                pipe_dim_list.append((round(pipe_dim_1, 3), round(pipe_dim_2, 3)))
            else:
                pipe_dim_list.append(None)
        
        fitt_prop['dim'] = pipe_dim_list

        # Calculate flow direction vectors
        exit_vec_list = []
        for item in pipe_list:
            if item is not None:
                exit_pipe = pipe_list[pipe_list.index(item)][0][6][2][0][3][0][1][0][0]
                fitt_vec = pipe_list[pipe_list.index(item)][1][5][1][0][0]
                exit_vec_list.append((exit_pipe, fitt_vec))
            else:
                exit_vec_list.append(None)

        # Calculate resultant vectors between pipes and fitting
        exit_res_vec_list = []
        for item in exit_vec_list:
            if item is not None:
                exit_res_vec_list.append(tuple((x-y) for x, y in zip(exit_vec_list[exit_vec_list.index(item)][0], 
                                                                   exit_vec_list[exit_vec_list.index(item)][1])))
            else:
                exit_res_vec_list.append(None)

        # Determine flow direction (normalized to 0 or 1 for each coordinate)
        exit_dir_list = []
        for item in exit_res_vec_list:
            exit_dir_tuple = ()
            if item is not None:
                for coordinate in item:
                    if round(coordinate, 2) != 0:
                        exit_dir_tuple += (1, )
                    else:
                        exit_dir_tuple += (0, )
                exit_dir_list.append(exit_dir_tuple)
                del(exit_dir_tuple)
            else:
                exit_dir_list.append(None)

        fitt_prop['dir'] = exit_dir_list

        # Extract fitting type from IFC properties
        fitt_type = fitt[8]
        fitt_prop['type'] = fitt_type

        Base.append_log(self, f"> Fitting properties:")
        Base.append_log(self, f"> {fitt_prop}")
        return fitt_prop

    def valv_properties(self, valv) -> dict:
        """
        Extract properties from a valve.
        
        Args:
            valv: IFC valve object
            
        Returns:
            dict: Dictionary containing dimensions ('dim') and type ('type')
        """
        Base.append_log(self, f"> Getting valve properties for valve with ID {valv.id()}...")
        valv_prop = {}

        # Get topology and all paths (similar to fitting analysis)
        topo = TopologyCreator()
        all_paths = topo.all_paths_finder()
        all_paths_id = []

        n = 0
        for path in all_paths:
            all_paths_id.append([])
            for item in path:
                all_paths_id[n].append(all_paths[n][all_paths[n].index(item)].id())
            n += 1

        # Find valve position in paths
        valv_index = []
        for path in all_paths_id:
            for item in path:
                if item == valv.id():
                    valv_index.append(path.index(item))
            if valv.id() not in path:
                valv_index.append(None)

        # Get adjacent pipes
        pipe_list = []
        start_index = 0
        for item in valv_index:
            if item is not None:
                pipe_list.append((all_paths[valv_index.index(item, start_index)][item-1], 
                                all_paths[valv_index.index(item, start_index)][item], 
                                all_paths[valv_index.index(item)][item+1]))
                start_index = valv_index.index(item) + 1
            else:
                pipe_list.append(None)

        # Extract diameters from adjacent pipes
        pipe_dim_list = []
        for item in pipe_list:
            if item is not None:
                pipe_dim_1 = pipe_list[pipe_list.index(item)][0][6][2][0][3][0][0][2][0][0][0][0] * 2
                pipe_dim_2 = pipe_list[pipe_list.index(item)][2][6][2][0][3][0][0][2][0][0][0][0] * 2
                pipe_dim_list.append((round(pipe_dim_1, 3), round(pipe_dim_2, 3)))
            else:
                pipe_dim_list.append(None)
        
        valv_prop['dim'] = pipe_dim_list

        # Extract valve type
        valv_type = valv[8]
        valv_prop['type'] = valv_type

        Base.append_log(self, f"> Valve properties:")
        Base.append_log(self, f"> {valv_prop}")
        return valv_prop

class HydroCalculator():
    """
    Performs hydraulic calculations for the water supply system.
    
    This class implements hydraulic analysis methods including flow calculations,
    pressure drop analysis, and available pressure determination using industry
    standard equations and coefficients.
    """

    def __init__(self) -> None:
        """Initialize the HydroCalculator."""
        pass

    def flow(self, all_paths: list) -> list:
        """
        Calculate design flow for every component in the hydraulic system.
        
        Uses standardized design flow rates for different sanitary terminal types
        and propagates these flows through the network.
        
        Args:
            all_paths (list): List of all hydraulic paths in the system
            
        Returns:
            list: Flow rates for each component in each path
        """
        # Design flow rates by sanitary terminal type (L/s)
        design_flow_table = {'SHOWER': 0.2, 'WASHHANDBASIN': 0.15, 'WCSEAT': 0.15}

        # Calculate cumulative flow for each component in each path
        flow_list = []
        n = 0
        for path in all_paths:
            flow_list.append([])  
            i = -1          
            for component in path:
                if component.is_a() == "IfcSanitaryTerminal":
                    # Assign design flow for terminals
                    flow_list[n].append((component[0], design_flow_table[component[8]]))
                else:
                    # Propagate flow from previous component
                    flow_list[n].append((component[0], flow_list[n][i][1]))
                i += 1
            n += 1
        
        return flow_list                   

    def linear_pressure_drop(self, pipe, all_paths: list) -> float:
        """
        Calculate linear pressure drop in a pipe using Fair Whipple-Hsiao equations.
        
        Implements the Fair Whipple-Hsiao equation for PVC pipes, recommended
        for pipes with diameter between 12.5 mm and 100 mm.
        
        Args:
            pipe: IFC pipe segment object
            all_paths (list): List of all hydraulic paths
            
        Returns:
            float: Linear pressure drop in meters of water column
        """
        # Initialize calculation components
        prop_calc = PropCalculator()
        pipe_prop = prop_calc.pipe_properties(pipe)
        flow = self.flow(all_paths)
        design_flow = 0

        Base.append_log(self, f"> Getting linear pressure drop for pipe with ID {pipe.id()}...")

        # Calculate cumulative design flow for the specified pipe
        for path in flow:
            for component in path:
                if component[0] == pipe[0]:
                    design_flow += component[1]
        
        # Fair Whipple-Hsiao equation for PVC pipes
        # Recommended for pipes with d between 12.5 mm and 100 mm
        pressure_drop = pipe_prop.get('len') * (0.000859 * ((design_flow * 0.001) ** 1.75) *  (pipe_prop.get('dim') ** -4.75))

        # Legacy Hazen-Williams equation (commented out)
        # pressure_drop = (10.67 * pipe_prop.get('len') * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (pipe_prop.get('dim') ** 4.87))

        Base.append_log(self, f"> Linear pressure drop:")
        Base.append_log(self, f"> {round(pressure_drop, 5)} m")
        return pressure_drop

    def local_pressure_drop(self, conn, all_paths: list) -> float:
        """
        Calculate local pressure drop in fittings and valves using equivalent length method.
        
        Uses tabulated equivalent length values for different connection types
        and applies the Hazen-Williams equation.
        
        Args:
            conn: IFC connection object (fitting or valve)
            all_paths (list): List of all hydraulic paths
            
        Returns:
            float: Local pressure drop in meters of water column
        """
        # Equivalent length factors for different connection types
        local_pressure_drop_table = {
            'JUNCTION': 2.4, 
            'BEND': 1.2, 
            'EXIT': 1.2, 
            'ISOLATING': 0.2, 
            'REGULATING': 11.4
        }

        # Initialize calculation components
        prop_calc = PropCalculator()
        flow = self.flow(all_paths)
        design_flow = 0

        Base.append_log(self, f"> Getting local pressure drop for connection with ID {conn.id()}...")

        # Calculate cumulative design flow for the specified connection
        for path in flow:
            for component in path:
                if component[0] == conn[0]:
                    design_flow += component[1]

        # Get connection properties based on type
        if conn.is_a() == 'IfcValve':
            conn_prop = prop_calc.valv_properties(conn)
        elif conn.is_a() == 'IfcPipeFitting':
            conn_prop = prop_calc.fitt_properties(conn)
        else:
            return 0

        # Fair Whipple-Hsiao equation for PVC pipes
        # Recommended for pipes with d between 12.5 mm and 100 mm
        pressure_drop = local_pressure_drop_table.get(conn_prop.get('type')) * (0.000859 * ((design_flow * 0.001) ** 1.75) *  (0.025 ** -4.75))
        
        # Hazen-Williams equation with equivalent length for PVC (C = 140)
        # Using standard reference diameter of 25mm for equivalent length calculations
        # pressure_drop = (10.67 * local_pressure_drop_table.get(conn_prop.get('type')) * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (0.025 ** 4.87))

        Base.append_log(self, f"> Local pressure drop:")
        Base.append_log(self, f"> {round(pressure_drop, 5)} m")
        return pressure_drop

    def available_pressure(self, term, all_paths: list) -> float:
        """
        Calculate available pressure at a sanitary terminal.
        
        Computes the net available pressure by starting with gravity potential
        and subtracting all pressure losses along the flow path.
        
        Args:
            term: IFC sanitary terminal object
            all_paths (list): List of all hydraulic paths
            
        Returns:
            float: Available pressure in meters of water column
        """
        # Find the path containing the specified terminal
        for path in all_paths:
            if term.id() == path[0].id():
                selected_path = path
                # Get elevation coordinates
                terminal_pipe_location = path[0][6][2][0][3][0][1][0][0]
                tank_pipe_location = path[len(path)-2][6][2][0][3][0][1][0][0]

        # Calculate initial pressure from elevation difference (gravity potential)
        pressure = (tank_pipe_location[2] + selected_path[len(selected_path)-2][5][0][1][0][0][2]) - terminal_pipe_location[2]

        Base.append_log(self, f"> Getting available pressure at sanitary terminal with ID {selected_path[0].id()}...")

        # Subtract pressure losses from each component along the path
        for component in selected_path:
            if component.is_a() == "IfcPipeSegment":
                # Linear pressure drop in pipes
                pressure_drop = self.linear_pressure_drop(component, all_paths)
                pressure -= pressure_drop
            elif component.is_a() == "IfcPipeFitting":
                # Local pressure drop in fittings
                pressure_drop = self.local_pressure_drop(component, all_paths)
                pressure -= pressure_drop
            elif component.is_a() == "IfcValve":
                # Local pressure drop in valves
                pressure_drop = self.local_pressure_drop(component, all_paths)
                pressure -= pressure_drop            
            else:
                # Skip other component types (terminals, tanks)
                pass

        Base.append_log(self, f"> Available pressure at the sanitary terminal:")
        Base.append_log(self, f"> {round(pressure, 2)} m")
        return pressure

# Test environment
if __name__ == '__main__':
    """
    Test environment for hydraulic calculations.
    
    This section demonstrates the usage of the IfcHydro classes and methods
    for analyzing hydraulic systems from IFC models.
    """
    
    # Initialize topology creator and calculate all paths
    topology = TopologyCreator()
    test_path = topology.all_paths_finder()

    # Load IFC model and initialize calculators
    model = ifc.open('ProjetoRamalBanheiro_Revit.ifc')
    prop_calc = PropCalculator()
    hydro_calc = HydroCalculator()

    # Commented test cases for individual component analysis
    """
    # Test flow calculations
    flow_calc_test = hydro_calc.flow(test_path)

    # Test pipe property extraction and pressure drop calculation
    pipe_test = model.by_id(5399)
    pipe_prop_test = prop_calc.pipe_properties(pipe_test)
    pipe_calc_test = hydro_calc.linear_pressure_drop(pipe_test, test_path)

    # Test fitting property extraction and pressure drop calculation
    fitt_test = model.by_id(7020)
    fitt_prop_test = prop_calc.fitt_properties(fitt_test)
    fitt_calc_test = hydro_calc.local_pressure_drop(fitt_test, test_path)

    # Test valve property extraction and pressure drop calculation
    valv_test = model.by_id(8087)
    valv_prop_test = prop_calc.valv_properties(valv_test)
    valv_calc_test = hydro_calc.local_pressure_drop(valv_test, test_path)    
    """
    
    # Test available pressure calculation for a specific terminal
    term_test = model.by_id(5423)
    press_test = hydro_calc.available_pressure(term_test, test_path)
