"""
Property extraction module for hydraulic system components.

This module provides methods to extract geometric and type properties
from pipes, fittings, and valves in the IFC model.
"""

from ..core.base import Base


class PropCalculator:
    """
    Extracts properties from IFC hydraulic system components.

    This class provides methods to extract geometric and type properties
    from pipes, fittings, and valves in the IFC model.
    """

    def __init__(self) -> None:
        """Initialize the PropCalculator."""
        pass
    
    def create_direction_vector(self, from_point: tuple, to_point: tuple) -> tuple:
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

    def vector_magnitude(self, vector: tuple) -> float:
        """
        Calculate the magnitude (length) of a vector.
        Args:
            vector (tuple): 3D vector as (x, y, z)
        Returns:
            float: Magnitude of the vector
        """
        return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

    def normalize_vector(self, vector: tuple) -> tuple:
        """
        Normalize a vector to unit length.
        Args:
            vector (tuple): 3D vector as (x, y, z)
        Returns:
            tuple: Unit vector in the same direction
        """
        magnitude = self.vector_magnitude(vector)
        if magnitude == 0:
            return (0.0, 0.0, 0.0)
        return (round(vector[0]/magnitude, 0), round(vector[1]/magnitude, 0), round(vector[2]/magnitude, 0))

    def dot_product(self, vector1: tuple, vector2: tuple) -> float:
        """
        Calculate the dot product of two vectors.
        Args:
            vector1 (tuple): First 3D vector as (x, y, z)
            vector2 (tuple): Second 3D vector as (x, y, z)
        Returns:
            float: Dot product of the two vectors
        """
        return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]

    def angle_between_vectors(self, vector1: tuple, vector2: tuple) -> float:
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
        unit1 = self.normalize_vector(vector1)
        unit2 = self.normalize_vector(vector2)

        # Calculate dot product
        dot = self.dot_product(unit1, unit2)

        # Clamp dot product to [-1, 1] to handle numerical errors
        dot = max(-1.0, min(1.0, dot))

        # Calculate angle in radians then convert to degrees
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        return round(angle_deg, 2)

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
        incoming_dir = self.create_direction_vector(incoming_pipe_center, fitting_center)
        # Outgoing: from fitting center TO outgoing pipe center
        outgoing_dir = self.create_direction_vector(fitting_center, outgoing_pipe_center)

        # Normalize to unit vectors
        incoming_unit = self.normalize_vector(incoming_dir)
        outgoing_unit = self.normalize_vector(outgoing_dir)

        # Calculate angle between vectors
        angle = self.angle_between_vectors(incoming_dir, outgoing_dir)

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
        incoming_dir = self.create_direction_vector(incoming_pipe_center, valve_center)
        # Outgoing: from valve center TO outgoing pipe center
        outgoing_dir = self.create_direction_vector(valve_center, outgoing_pipe_center)

        # Normalize to unit vectors
        incoming_unit = self.normalize_vector(incoming_dir)
        outgoing_unit = self.normalize_vector(outgoing_dir)

        # Calculate angle between vectors
        angle = self.angle_between_vectors(incoming_dir, outgoing_dir)

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
