"""
Hydraulic calculations module for water supply systems.

This module implements hydraulic analysis methods including flow calculations,
pressure drop analysis, and available pressure determination using industry
standard equations and coefficients.
"""

from ..core.base import Base
from ..properties.pipe import Pipe
from ..properties.fitting import Fitting
from ..properties.valve import Valve


class HydroCalculator:
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
        pipe_prop = Pipe.properties(pipe)
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

        # Legacy Hazen-Williams equation
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
            conn_prop = Valve.properties(conn, path)
        elif conn.is_a() == 'IfcPipeFitting':
            conn_prop = Fitting.properties(conn, path)
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
        Base.append_log(self, f"{'-'*100}")

        # Subtract pressure losses from each component along the path
        for component in selected_path:
            if component.is_a() == "IfcPipeSegment":
                # Linear pressure drop in pipes
                pressure_drop = self.linear_pressure_drop(component, all_paths)
                pressure -= pressure_drop
                Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
            elif component.is_a() == "IfcPipeFitting":
                # Local pressure drop in fittings
                pressure_drop = self.local_pressure_drop(component, selected_path, all_paths)
                pressure -= pressure_drop
                Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
            elif component.is_a() == "IfcValve":
                # Local pressure drop in valves
                pressure_drop = self.local_pressure_drop(component, selected_path, all_paths)
                pressure -= pressure_drop
                Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
            else:
                # Skip other component types (terminals, tanks)
                pass

        Base.append_log(self, f"{'='*100}")
        Base.append_log(self, f"> Available pressure at the sanitary terminal {selected_path[0].id()} - Type: {selected_path[0][8]}:")
        Base.append_log(self, f"> {round(pressure, 3)} m")
        Base.append_log(self, f"{'='*100}")
        return pressure