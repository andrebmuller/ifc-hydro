"""
Pipe property extraction module.

This module provides methods to extract geometric and type properties
from pipe segments in the IFC model.
"""

from ..core.base import Base
from ..core.constants import get_nominal_diameter, get_internal_diameter


class Pipe:
    """
    Extracts properties from IFC pipe segments.

    This class provides methods to extract geometric properties including
    length and diameter from pipe segments in the IFC model.
    """

    @staticmethod
    def properties(pipe) -> dict:
        """
        Extract properties from a pipe segment.

        Args:
            pipe: IFC pipe segment object

        Returns:
            dict: Dictionary containing pipe length ('len'), diameter ('dim'),
                  and nominal diameter in mm ('nominal_dim')

        Raises:
            ValueError: If pipe properties cannot be extracted from the IFC element
        """
        Base.append_log(None, f"> Getting pipe properties for pipe with ID {pipe.id()}...")
        pipe_prop = {}

        # Extract pipe length from IFC geometry representation with error handling
        try:
            pipe_len = pipe[6][2][0][3][0][3]
            pipe_prop['len'] = round(pipe_len, 3)
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"ERROR: Failed to extract length from pipe ID {pipe.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Extract pipe diameter (radius * 2) from IFC geometry with error handling
        try:
            pipe_dim = pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"ERROR: Failed to extract diameter from pipe ID {pipe.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Map nominal pipe diameter to real internal diameter using centralized table
        nominal_diameter = get_nominal_diameter(pipe_dim)
        internal_diameter = get_internal_diameter(nominal_diameter)

        # Fallback if diameter not in table
        if internal_diameter == 0.0:
            Base.append_log(None, f"> WARNING: Nominal diameter {nominal_diameter}mm not found in table, using measured value")
            internal_diameter = pipe_dim

        pipe_prop['dim'] = internal_diameter
        pipe_prop['nominal_dim'] = nominal_diameter

        Base.append_log(None, f"> Pipe properties:")
        Base.append_log(None, f"> {pipe_prop}")
        return pipe_prop
