"""
Topology creation module for hydraulic systems.

This module analyzes IFC files to extract hydraulic component relationships
and create a graph representation of the system topology.
"""

from ..core.base import Base
from ..core.graph import Graph


class TopologyCreator:
    """
    Creates hydraulic system topology from IFC models.

    This class analyzes IFC files to extract hydraulic component relationships
    and create a graph representation of the system topology.

    Attributes:
        model: The loaded IFC model
    """

    def __init__(self, model):
        """
        Initialize the TopologyCreator with an IFC model.

        Args:
            model: An opened IFC model object (from ifcopenshell)
        """
        self.model = model

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
