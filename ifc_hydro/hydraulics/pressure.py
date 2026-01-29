"""
Pressure calculation module for water supply systems.

This module implements available pressure calculations at sanitary terminals,
accounting for gravity potential and all pressure losses along the flow path.
"""

import ifcopenshell as ifc
from ifcopenshell import geom
from ifcopenshell import util
from ifcopenshell.util import shape
from ..core.base import Base
from .pressure_drop import PressureDrop
from .input_tables import tank_height_adjustment

class Pressure:
    """
    Calculates available pressure at sanitary terminals.

    This class computes the net available pressure by starting with gravity
    potential and subtracting all pressure losses along the flow path.
    """

    def __init__(self, model) -> None:
        """
        Initialize the Pressure calculator.

        Args:
            model: The IFC model object (from ifcopenshell.open())
        """
        self.model = model
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

        Raises:
            ValueError: If paths are empty or terminal path is not found
            IndexError: If IFC element structure is invalid or missing required properties
        """
        # Validate input
        if not all_paths:
            error_msg = "> ERROR: No paths provided for pressure calculation. Cannot proceed."
            Base.append_log(self, error_msg)
            raise ValueError(error_msg)

        # Find the path containing the specified terminal
        selected_path = None
        for path in all_paths:
            if not path or len(path) == 0:
                continue
            try:
                if term.id() == path[0].id():
                    selected_path = path
                    # Get elevation coordinates with error handling
                    try:
                        settings = ifc.geom.settings()
                        settings.set(settings.USE_WORLD_COORDS, True)

                        terminal_guid = path[0][0]
                        terminal = self.model.by_guid(terminal_guid)
                        terminal_shape = ifc.geom.create_shape(settings, terminal)
                        terminal_pipe_location = round(ifc.util.shape.get_top_elevation(terminal_shape.geometry), 3)

                        tank_guid = path[len(path)-1][0]
                        tank = self.model.by_guid(tank_guid)
                        tank_shape = ifc.geom.create_shape(settings, tank)
                        tank_pipe_location = round(ifc.util.shape.get_bottom_elevation(tank_shape.geometry), 3)
                                                                                                              
                    except (IndexError, TypeError) as e:
                        error_msg = f"> ERROR: Representation type not yet implemented: {path[0][6][2][0][2]}. Details: {str(e)}"
                        Base.append_log(self, error_msg)
                        raise NotImplementedError(error_msg)
                    break
            except (AttributeError, RuntimeError) as e:
                # Skip invalid paths
                continue

        if selected_path is None:
            error_msg = f"> ERROR: No path found for terminal with ID {term.id()}. The terminal may not be connected to any tank in the topology."
            Base.append_log(self, error_msg)
            raise ValueError(error_msg)

        # Calculate initial pressure from elevation difference (gravity potential)
        # Total head = tank bottom elevation + water level above pipe connection
        try:
            tank_height_adjustment = 0.5
            tank_height_adjustment = 0.5
            tank_height_adjustment = 0.5
            tank_height_adjustment = 0.5
            tank_height_adjustment = 0.5
            total_tank_height = tank_pipe_location + tank_height_adjustment
            terminal_height = terminal_pipe_location

            pressure = total_tank_height - terminal_height
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to calculate pressure from elevation data. IFC element structure may be invalid. Details: {str(e)}"
            Base.append_log(self, error_msg)
            raise IndexError(error_msg)

        Base.append_log(self, f"> Getting available pressure at sanitary terminal with ID {selected_path[0].id()}...")
        Base.append_log(self, f"> Tank height: {round(total_tank_height, 3)} m")
        Base.append_log(self, f"> Terminal height: {round(terminal_height, 3)} m")
        Base.append_log(self, f"> Initial pressure from gravity potential: {round(pressure, 3)} m")
        Base.append_log(self, f"{'-'*100}")

        # Subtract pressure losses from each component along the path
        for component in selected_path:
            try:
                component_type = component.is_a()
                if component_type == "IfcPipeSegment":
                    # Linear pressure drop in pipes
                    try:
                        pressure_loss = self.pressure_drop.linear(component, all_paths)
                        pressure -= pressure_loss
                        Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
                    except Exception as e:
                        error_msg = f"> WARNING: Failed to calculate pressure loss for pipe component ID {component.id()}: {str(e)}"
                        Base.append_log(self, error_msg)
                        # Continue with next component
                        continue
                elif component_type == "IfcPipeFitting":
                    # Local pressure drop in fittings
                    try:
                        pressure_loss = self.pressure_drop.local(component, selected_path, all_paths)
                        pressure -= pressure_loss
                        Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
                    except Exception as e:
                        error_msg = f"> WARNING: Failed to calculate pressure loss for fitting component ID {component.id()}: {str(e)}"
                        Base.append_log(self, error_msg)
                        # Continue with next component
                        continue
                elif component_type == "IfcValve":
                    # Local pressure drop in valves
                    try:
                        pressure_loss = self.pressure_drop.local(component, selected_path, all_paths)
                        pressure -= pressure_loss
                        Base.append_log(self, f"==> Pressure after component ID {component.id()}: {round(pressure, 3)} m")
                    except Exception as e:
                        error_msg = f"> WARNING: Failed to calculate pressure loss for valve component ID {component.id()}: {str(e)}"
                        Base.append_log(self, error_msg)
                        # Continue with next component
                        continue
                else:
                    # Skip other component types (terminals, tanks)
                    pass
            except (AttributeError, RuntimeError) as e:
                error_msg = f"> WARNING: Failed to process component in path: {str(e)}"
                Base.append_log(self, error_msg)
                continue

        try:
            terminal_type = selected_path[0][8]
        except (IndexError, TypeError):
            terminal_type = "Unknown"

        Base.append_log(self, f"{'='*100}")
        Base.append_log(self, f"> Available pressure at the sanitary terminal {selected_path[0].id()} - Type: {terminal_type}:")
        Base.append_log(self, f"> {round(pressure, 2)} m")
        Base.append_log(self, f"{'='*100}")
        return pressure
