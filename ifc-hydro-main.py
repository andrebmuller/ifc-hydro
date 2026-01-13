"""
ifc-hydro

This module provides hydraulic system analysis capabilities for IFC (Industry Foundation Classes) models.
It includes functionality for topology creation, property calculation, and hydraulic calculations
for building water supply and drainage systems.

The module contains the following main classes:
- Base: Base class with logging functionality
- Graph: Graph data structure for representing system topology
- TopologyCreator: Creates hydraulic system topology from IFC models
- PropCalculator: Extracts properties from IFC components
- HydroCalculator: Performs hydraulic calculations

Version: 3.0.0
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
    
    _log     = "ifc-hydro"      # name of log file
    _counter = 0                # instance counter

    def __init__(self, log: str = ""):
        """
        Initialize the Base class instance.
        
        Args:
            log (str, optional): Custom log file name. If empty, uses default log file.
        """

        if log != "": 
            Base._log = log

        Base._counter += 1

    def configure_log(cls, log_dir: str = "", log_name: str = "") -> str:
        """
        Configure the log file location and name for the current run.

        Args:
            log_dir (str, optional): Directory where the log file should be stored.
                Defaults to the current working directory.
            log_name (str, optional): Log file name. Defaults to "ifc-hydro.log".

        Returns:
            str: The full path to the configured log file.
        """

        if log_name == "":
            log_name = "ifc-hydro"

        if log_dir == "":
            log_dir = os.getcwd()

        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        cls._log = os.path.join(log_dir, log_name+".log")

        Base.append_log(Base, f">>> Project {log_name}:")
        Base.append_log(Base, f">>> New run started at {time.now().strftime('%d/%m/%Y %H:%M:%S')}.")
        Base.append_log(Base, f"!!!")

        return cls._log

    def append_log(self, text: str):
        """
        Append a timestamped message to the log file and print to console.
        
        Args:
            text (str): The message to log
        """
        t = time.now()
        tstamp = "%2.2d.%2.2d.%2.2d " % (t.hour, t.minute, t.second)

        otext = tstamp + text
        with open(Base._log, "a") as f:
            f.write(otext + "\n")
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
        self.model = ifc.open('projeto-demonstracao.ifc')

    def graph_creator(self) -> Graph:
        """
        Create a graph representing the hydraulic system topology.
        
        Analyzes the IFC model to find connections between hydraulic components
        using IfcRelNests and IfcRelConnectsPorts relationships.
        
        Returns:
            Graph: Undirected graph representing the hydraulic system topology
        """
        model = self.model

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

        all_paths = []

        # Get all terminals and tanks from the model
        term_list = model.by_type("IfcSanitaryTerminal")
        tank_list = model.by_type("IfcTank")

        Base.append_log(Base, f"> Creating topology...")

        # Calculate paths between all terminal-tank combinations
        for term in term_list:
            for tank in tank_list:
                all_paths.append(graph.find_path(term, tank))
                
        Base.append_log(Base, f"> Topology created with {len(all_paths)} paths...")
        Base.append_log(Base, f"!!!")

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

    def _vector_magnitude(self, vector: tuple) -> float:
        """
        Calculate the magnitude (length) of a vector.
        Args:
            vector (tuple): 3D vector as (x, y, z)
        Returns:
            float: Magnitude of the vector
        """
        return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

    def _normalize_vector(self, vector: tuple) -> tuple:
        """
        Normalize a vector to unit length.
        Args:
            vector (tuple): 3D vector as (x, y, z)
        Returns:
            tuple: Unit vector in the same direction
        """
        magnitude = self._vector_magnitude(vector)
        if magnitude == 0:
            return (0.0, 0.0, 0.0)
        return (round(vector[0]/magnitude, 0), round(vector[1]/magnitude, 0), round(vector[2]/magnitude, 0))

    def _dot_product(self, vector1: tuple, vector2: tuple) -> float:
        """
        Calculate the dot product of two vectors.
        Args:
            vector1 (tuple): First 3D vector as (x, y, z)
            vector2 (tuple): Second 3D vector as (x, y, z)
        Returns:
            float: Dot product of the two vectors
        """
        return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]

    def _angle_between_vectors(self, vector1: tuple, vector2: tuple) -> float:
        """
        Calculate the angle between two vectors in degrees.
        Args:
            vector1 (tuple): First 3D vector as (x, y, z)
            vector2 (tuple): Second 3D vector as (x, y, z)
        Returns:
            float: Angle between vectors in degrees (0-180)
        """
        import math

        # Normalize vectors
        unit1 = self._normalize_vector(vector1)
        unit2 = self._normalize_vector(vector2)

        # Calculate dot product
        dot = self._dot_product(unit1, unit2)

        # Clamp dot product to [-1, 1] to handle numerical errors
        dot = max(-1.0, min(1.0, dot))

        # Calculate angle in radians then convert to degrees
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        return round(angle_deg, 2)

    def _create_direction_vector(self, from_point: tuple, to_point: tuple) -> tuple:
        """
        Create a direction vector from one point to another.
        Args:
            from_point (tuple): Starting point as (x, y, z)
            to_point (tuple): Ending point as (x, y, z)
        Returns:
            tuple: Direction vector as (dx, dy, dz)
        """
        return (
            round(to_point[0] - from_point[0], 3),
            round(to_point[1] - from_point[1], 3),
            round(to_point[2] - from_point[2], 3)
        )

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
        real_dim = 0

        # Extract pipe length from IFC geometry representation
        pipe_len = pipe[6][2][0][3][0][3]
        pipe_prop['len'] = round(pipe_len, 3)

        # Extract pipe diameter (radius * 2) from IFC geometry
        pipe_dim = pipe[6][2][0][3][0][0][2][0][0][0][0] * 2

        if round(pipe_dim, 3) == 0.015:
            real_dim = 0.0170
        if round(pipe_dim, 3) == 0.020:
            real_dim = 0.0216
        elif round(pipe_dim, 3) == 0.025:
            real_dim = 0.0278
        elif round(pipe_dim, 3) == 0.032:
            real_dim = 0.0352
        elif round(pipe_dim, 3) == 0.040:
            real_dim = 0.0440
        elif round(pipe_dim, 3) == 0.050:
            real_dim = 0.0534
        elif round(pipe_dim, 3) == 0.065:
            real_dim = 0.0666
        elif round(pipe_dim, 3) == 0.075:
            real_dim = 0.0756
        elif round(pipe_dim, 3) == 0.100:
            real_dim = 0.0978
        else:
            real_dim = 0.000
        
        pipe_prop['dim'] = real_dim
          
        Base.append_log(self, f"> Pipe properties:")
        Base.append_log(self, f"> {pipe_prop}")
        return pipe_prop
    
    def fitt_properties(self, fitt, path: list) -> dict:
        """
        Extract properties from a pipe fitting for a specific path.

        Analyzes the fitting's position in the provided hydraulic path to determine
        
        Args:
            fitt: IFC pipe fitting object
            path (list): The specific hydraulic path containing this fitting
            
        Returns:
            dict: Dictionary containing dimensions ('dim'), directions ('dir'), and type ('type')
        """
        Base.append_log(self, f"> Getting fitting properties for fitting with ID {fitt.id()}...")
        fitt_prop = {}        

        # Find fitting position in the provided path
        fitt_index = None
        for i, component in enumerate(path):
            if component.id() == fitt.id():
                fitt_index = i
                break

        if fitt_index is None:
            Base.append_log(self, f"> ERROR: Fitting with ID {fitt.id()} not found in the provided path")
            return None

        # Get adjacent components (incoming pipe, fitting, outgoing pipe)
        incoming_pipe = path[fitt_index - 1]
        outgoing_pipe = path[fitt_index + 1]

        # Get adjacent components (incoming pipe, fitting, outgoing pipe)
        incoming_pipe = path[fitt_index - 1]
        outgoing_pipe = path[fitt_index + 1]

        # Extract diameters from adjacent pipes
        pipe_dim_1 = incoming_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        pipe_dim_2 = outgoing_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        fitt_prop['dim'] = (round(pipe_dim_1, 3), round(pipe_dim_2, 3))

        # Calculate unit vectors and angles for flow direction change
        # Get center points from IFC geometry
        incoming_pipe_center = incoming_pipe[6][2][0][3][0][1][0][0]
        fitting_center = fitt[5][1][0][0]
        outgoing_pipe_center = outgoing_pipe[6][2][0][3][0][1][0][0]

        # Create direction vectors between points
        # Incoming: from incoming pipe center TO fitting center
        incoming_dir = self._create_direction_vector(incoming_pipe_center, fitting_center)
        # Outgoing: from fitting center TO outgoing pipe center
        outgoing_dir = self._create_direction_vector(fitting_center, outgoing_pipe_center)

        # Normalize to unit vectors
        incoming_unit = self._normalize_vector(incoming_dir)
        outgoing_unit = self._normalize_vector(outgoing_dir)

        # Calculate angle between vectors
        angle = self._angle_between_vectors(incoming_dir, outgoing_dir)

        # Store as dictionary with all relevant information
        fitt_prop['dir'] = {
            'incoming_unit_vector': incoming_unit,
            'outgoing_unit_vector': outgoing_unit,
            'direction_change_angle': angle
        }

        # Extract fitting type from IFC properties
        fitt_type = fitt[8]
        fitt_prop['type'] = fitt_type

        Base.append_log(self, f"> Fitting properties:")
        Base.append_log(self, f"> {fitt_prop}")
        return fitt_prop

    def valv_properties(self, valv, path: list) -> dict:
        """
        Extract properties from a valve for a specific path.

        Analyzes the valve's position in the provided hydraulic path to determine
        dimensions and flow directions.
        
        Args:
            valv: IFC valve object
            path (list): The specific hydraulic path containing this valve
            
        Returns:
            dict: Dictionary containing dimensions ('dim'), direction ('dir'), and type ('type')
        """
        Base.append_log(self, f"> Getting valve properties for valve with ID {valv.id()}...")
        valv_prop = {}

        # Find valve position in the provided path
        valv_index = None
        for i, component in enumerate(path):
            if component.id() == valv.id():
                valv_index = i
                break

        if valv_index is None:
            Base.append_log(self, f"> ERROR: Valve with ID {valv.id()} not found in the provided path")
            return None

        # Get adjacent components (incoming pipe, valve, outgoing pipe)
        incoming_pipe = path[valv_index - 1]
        outgoing_pipe = path[valv_index + 1]

        # Extract diameters from adjacent pipes
        pipe_dim_1 = incoming_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        pipe_dim_2 = outgoing_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        valv_prop['dim'] = (round(pipe_dim_1, 3), round(pipe_dim_2, 3))

        # Calculate unit vectors and angles for flow direction change
        # Get center points from IFC geometry
        incoming_pipe_center = incoming_pipe[6][2][0][3][0][1][0][0]
        valve_center = valv[5][1][0][0]
        outgoing_pipe_center = outgoing_pipe[6][2][0][3][0][1][0][0]

        # Create direction vectors between points
        # Incoming: from incoming pipe center TO valve center
        incoming_dir = self._create_direction_vector(incoming_pipe_center, valve_center)
        # Outgoing: from valve center TO outgoing pipe center
        outgoing_dir = self._create_direction_vector(valve_center, outgoing_pipe_center)

        # Normalize to unit vectors
        incoming_unit = self._normalize_vector(incoming_dir)
        outgoing_unit = self._normalize_vector(outgoing_dir)

        # Calculate angle between vectors
        angle = self._angle_between_vectors(incoming_dir, outgoing_dir)

        # Store as dictionary with all relevant information
        valv_prop['dir'] = {
            'incoming_unit_vector': incoming_unit,
            'outgoing_unit_vector': outgoing_unit,
            'direction_change_angle': angle
        }

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

        Base.append_log(self, f"> Linear pressure drop: {round(pressure_drop, 3)} m")
        return pressure_drop

    def local_pressure_drop(self, conn, path: list, all_paths: list) -> float:
        """
        Calculate local pressure drop in fittings and valves using equivalent length method.
        
        Uses tabulated equivalent length values for different connection types
        and applies the Hazen-Williams equation.
        
        Args:
            conn: IFC connection object (fitting or valve)
            path (list): The specific hydraulic path containing this connection
            all_paths (list): List of all hydraulic paths (for flow calculation)
            
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
        for flow_path in flow:
            for component in flow_path:
                if component[0] == conn[0]:
                    design_flow += component[1]

        # Get connection properties based on type (pass the specific path)
        if conn.is_a() == 'IfcValve':
            conn_prop = prop_calc.valv_properties(conn, path)
        elif conn.is_a() == 'IfcPipeFitting':
            conn_prop = prop_calc.fitt_properties(conn, path)
        else:
            return 0

        # Fair Whipple-Hsiao equation for PVC pipes
        # Recommended for pipes with d between 12.5 mm and 100 mm
        pressure_drop = local_pressure_drop_table.get(conn_prop.get('type')) * (0.000859 * ((design_flow * 0.001) ** 1.75) *  (0.0278 ** -4.75))
        
        # Hazen-Williams equation with equivalent length for PVC (C = 140)
        # Using standard reference diameter of 25mm for equivalent length calculations
        # pressure_drop = (10.67 * local_pressure_drop_table.get(conn_prop.get('type')) * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (0.025 ** 4.87))

        Base.append_log(self, f"> Local pressure drop: {round(pressure_drop, 3)} m")
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
        Base.append_log(self, f"> Tank height: {round((tank_pipe_location[2] + selected_path[len(selected_path)-2][5][0][1][0][0][2]), 3)} m")
        Base.append_log(self, f"> Terminal height: {round(terminal_pipe_location[2], 3)} m")    
        Base.append_log(self, f"> Initial pressure from gravity potential: {round(pressure, 3)} m")
        Base.append_log(self, f"!!!")

        # Subtract pressure losses from each component along the path
        for component in selected_path:
            if component.is_a() == "IfcPipeSegment":
                # Linear pressure drop in pipes
                pressure_drop = self.linear_pressure_drop(component, all_paths)
                pressure -= pressure_drop
                Base.append_log(self, f"> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
                Base.append_log(self, f"!!!")
            elif component.is_a() == "IfcPipeFitting":
                # Local pressure drop in fittings
                pressure_drop = self.local_pressure_drop(component, selected_path, all_paths)
                pressure -= pressure_drop
                Base.append_log(self, f"> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
                Base.append_log(self, f"!!!")
            elif component.is_a() == "IfcValve":
                # Local pressure drop in valves
                pressure_drop = self.local_pressure_drop(component, selected_path, all_paths)
                pressure -= pressure_drop  
                Base.append_log(self, f"> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
                Base.append_log(self, f"!!!")                
            else:
                # Skip other component types (terminals, tanks)
                pass 

        Base.append_log(self, f"> Available pressure at the sanitary terminal:")
        Base.append_log(self, f"> {round(pressure, 3)} m")
        Base.append_log(self, f"!!!")   
        return pressure

# Test environment
if __name__ == '__main__':
    """
    Test environment for hydraulic calculations.
    
    This section demonstrates the usage of the IfcHydro classes and methods
    for analyzing hydraulic systems from IFC models.
    """
    
    # Configure log file for this run
    log_dir_input = input("Enter log directory (leave blank for current directory): ").strip()
    log_name_input = input("Enter log file name (leave blank for ifc-hydro.log): ").strip()
    Base.configure_log(Base,log_dir=log_dir_input, log_name=log_name_input)

    # Initialize topology creator and calculate all paths
    topology = TopologyCreator()
    test_path = topology.all_paths_finder()
            
    # Load IFC model and initialize calculators
    model = ifc.open('projeto-demonstracao.ifc')
    prop_calc = PropCalculator()  
    hydro_calc = HydroCalculator()

    # Commented test cases for individual component analysis
    """
    # Test flow calculations
    flow_calc_test = hydro_calc.flow(test_path)
    Base.append_log(Base, f"> Flow calculated...")

    # Test pipe property extraction and pressure drop calculation
    pipe_test = model.by_id(5399)
    pipe_prop_test = prop_calc.pipe_properties(pipe_test)
    pipe_calc_test = hydro_calc.linear_pressure_drop(pipe_test, test_path)
    Base.append_log(Base, f"> Pipes calculated...")

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
    # Shower         --> 5423
    # Wash and Basin --> 6986
    # WC Seat        --> 7061
    term_test = model.by_id(6986)
    press_test = hydro_calc.available_pressure(term_test, test_path)