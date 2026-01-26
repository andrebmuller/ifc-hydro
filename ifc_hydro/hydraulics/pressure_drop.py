"""
Pressure drop calculation module for water supply systems.

This module implements pressure drop analysis for pipes, fittings, and valves
using industry standard equations such as Fair Whipple-Hsiao.
"""

from ..core.base import Base
from ..properties.pipe import Pipe
from ..properties.fitting import Fitting
from ..properties.valve import Valve
from .design_flow import DesignFlow
from .input_tables import local_pressure_drop_table


class PressureDrop:
    """
    Calculates pressure drops in hydraulic system components.

    This class implements pressure drop calculations for pipes (linear losses)
    and fittings/valves (local losses) using industry standard equations.
    """

    def __init__(self) -> None:
        """Initialize the PressureDrop calculator."""
        self.design_flow = DesignFlow()

    def linear(self, pipe, all_paths: list) -> float:
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
        flow = self.design_flow.calculate(all_paths)
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

    def local(self, conn, path: list, all_paths: list) -> float:
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
        # Initialize calculation components
        flow = self.design_flow.calculate(all_paths)
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

        # Get the coefficient from the table
        table_value = local_pressure_drop_table.get(conn_prop.get('type'))

        # Get direction change angle for angle-based coefficient lookup
        direction_info = conn_prop.get('dir', {})
        direction_angle = direction_info.get('direction_change_angle', None)

        # Handle angle-based coefficients based on table value type
        if isinstance(table_value, tuple):
            # JUNCTION: tuple structure (straight flow, flow with direction change)
            # Select coefficient based on angle: index 0 for 0.0°, index 1 for others
            if direction_angle == 0.0:
                coefficient = table_value[0]
            else:
                coefficient = table_value[1]
        elif isinstance(table_value, dict):
            # BEND: dict structure {angle: coefficient} for angle-based selection
            # Use threshold of 67.5° (midpoint between 45° and 90°) to select coefficient
            if direction_angle is not None and direction_angle < 67.5:
                coefficient = table_value.get(45, 0.7)
            else:
                coefficient = table_value.get(90, 1.2)
        else:
            # Simple numeric coefficient
            coefficient = table_value

        # Fair Whipple-Hsiao equation for PVC pipes
        # Recommended for pipes with d between 12.5 mm and 100 mm
        pressure_drop = coefficient * (0.000859 * ((design_flow * 0.001) ** 1.75) *  (0.0278 ** -4.75))

        # Hazen-Williams equation with equivalent length for PVC (C = 140)
        # Using standard reference diameter of 25mm for equivalent length calculations
        # pressure_drop = (10.67 * local_pressure_drop_table.get(conn_prop.get('type')) * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (0.025 ** 4.87))

        Base.append_log(self, f"> Local pressure drop: {round(pressure_drop, 3)} m")
        return pressure_drop
