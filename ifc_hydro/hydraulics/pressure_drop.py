"""
Pressure drop calculation module for water supply systems.

This module implements pressure drop analysis for pipes, fittings, and valves
using industry standard equations such as Fair Whipple-Hsiao.

Multi-diameter support: Uses diameter-specific equivalent length coefficients
from input_tables.py for accurate calculations across different pipe sizes.
"""

from ..core.base import Base
from ..properties.pipe import Pipe
from ..properties.fitting import Fitting
from ..properties.valve import Valve
from .design_flow import DesignFlow
from .input_tables import (
    get_nominal_diameter,
    get_internal_diameter,
    get_fitting_coefficient,
    get_valve_coefficient,
)


class PressureDrop:
    """
    Calculates pressure drops in hydraulic system components.

    This class implements pressure drop calculations for pipes (linear losses)
    and fittings/valves (local losses) using industry standard equations.
    Supports multiple diameters with diameter-specific coefficients.
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

        Uses diameter-specific tabulated equivalent length values for different
        connection types and applies the Fair Whipple-Hsiao equation.

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

        # Extract diameter from connection properties
        # conn_prop['dim'] is a tuple: (incoming_diameter, outgoing_diameter)
        dim_tuple = conn_prop.get('dim', (0.025, 0.025))
        # Use the average of incoming and outgoing diameters for coefficient lookup
        avg_diameter_m = (dim_tuple[0] + dim_tuple[1]) / 2
        nominal_diameter = get_nominal_diameter(avg_diameter_m)
        internal_diameter = get_internal_diameter(nominal_diameter)

        # Use the actual internal diameter if available, otherwise fallback
        if internal_diameter == 0.0:
            internal_diameter = avg_diameter_m

        Base.append_log(self, f"> Connection nominal diameter: {nominal_diameter} mm, internal: {internal_diameter} m")

        # Get direction change angle for angle-based coefficient lookup
        direction_info = conn_prop.get('dir', {})
        direction_angle = direction_info.get('direction_change_angle', None)

        # Get coefficient based on connection type and diameter
        conn_type = conn_prop.get('type')

        if conn.is_a() == 'IfcValve':
            coefficient = get_valve_coefficient(conn_type, nominal_diameter)
        else:
            coefficient = get_fitting_coefficient(conn_type, nominal_diameter, direction_angle)

        Base.append_log(self, f"> Using equivalent lenght: {coefficient} for {conn_type} at {nominal_diameter}mm")

        # Fair Whipple-Hsiao equation for PVC pipes
        # Uses actual internal diameter for accurate calculation
        pressure_drop = coefficient * (0.000859 * ((design_flow * 0.001) ** 1.75) * (internal_diameter ** -4.75))

        # Legacy Hazen-Williams equation with equivalent length for PVC (C = 140)
        # pressure_drop = (10.67 * coefficient * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (internal_diameter ** 4.87))

        Base.append_log(self, f"> Local pressure drop: {round(pressure_drop, 3)} m")
        return pressure_drop
