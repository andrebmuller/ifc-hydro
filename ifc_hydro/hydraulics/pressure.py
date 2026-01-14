"""
Pressure calculation module for water supply systems.

This module implements available pressure calculations at sanitary terminals,
accounting for gravity potential and all pressure losses along the flow path.
"""

from ..core.base import Base
from .pressure_drop import PressureDrop


class Pressure:
    """
    Calculates available pressure at sanitary terminals.

    This class computes the net available pressure by starting with gravity
    potential and subtracting all pressure losses along the flow path.
    """

    def __init__(self) -> None:
        """Initialize the Pressure calculator."""
        self.pressure_drop = PressureDrop()

    def available(self, term, all_paths: list) -> float:
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
                pressure_loss = self.pressure_drop.linear(component, all_paths)
                pressure -= pressure_loss
                Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
            elif component.is_a() == "IfcPipeFitting":
                # Local pressure drop in fittings
                pressure_loss = self.pressure_drop.local(component, selected_path, all_paths)
                pressure -= pressure_loss
                Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
            elif component.is_a() == "IfcValve":
                # Local pressure drop in valves
                pressure_loss = self.pressure_drop.local(component, selected_path, all_paths)
                pressure -= pressure_loss
                Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
            else:
                # Skip other component types (terminals, tanks)
                pass

        Base.append_log(self, f"{'='*100}")
        Base.append_log(self, f"> Available pressure at the sanitary terminal {selected_path[0].id()} - Type: {selected_path[0][8]}:")
        Base.append_log(self, f"> {round(pressure, 3)} m")
        Base.append_log(self, f"{'='*100}")
        return pressure
