"""
Design flow calculation module for water supply systems.

This module implements flow calculations for hydraulic systems,
determining design flow rates for each component based on
standardized terminal flow rates.
"""

from ..core.base import Base
from .input_tables import design_flow_table


class DesignFlow:
    """
    Calculates design flow rates for hydraulic system components.

    This class implements flow calculations using standardized design flow
    rates for different sanitary terminal types and propagates these flows
    through the network.
    """

    def __init__(self) -> None:
        """Initialize the DesignFlow calculator."""
        pass

    def calculate(self, all_paths: list) -> list:
        """
        Calculate design flow for every component in the hydraulic system.

        Uses standardized design flow rates for different sanitary terminal types
        and propagates these flows through the network.

        Args:
            all_paths (list): List of all hydraulic paths in the system

        Returns:
            list: Flow rates for each component in each path
        """
        # Calculate cumulative flow for each component in each path
        flow_list = []
        n = 0


        for path in all_paths:
            flow_list.append([])
            i = -1
            for component in path:
                # Base.append_log(self, f"> Processing component with ID {component[0]} in path {n}...")
                if component.is_a() == "IfcSanitaryTerminal":
                    # Assign design flow for terminals
                    flow_list[n].append((component[0], design_flow_table[component[8]]))
                else:
                    # Propagate flow from previous component
                    flow_list[n].append((component[0], flow_list[n][i][1]))
                i += 1
            n += 1

        return flow_list
