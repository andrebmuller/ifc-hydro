"""
Pipe property extraction module.

This module provides methods to extract geometric and type properties
from pipe segments in the IFC model.
"""

from ..core.base import Base


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
            dict: Dictionary containing pipe length ('len') and diameter ('dim')

        Raises:
            ValueError: If pipe properties cannot be extracted from the IFC element
        """
        Base.append_log(None, f"> Getting pipe properties for pipe with ID {pipe.id()}...")
        pipe_prop = {}
        real_dim = 0

        # Extract pipe length from IFC geometry representation with error handling
        try:
            pipe_len = pipe[6][2][0][3][0][3]
            pipe_prop['len'] = round(pipe_len, 3)
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f> ERROR: Failed to extract length from pipe ID {pipe.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Extract pipe diameter (radius * 2) from IFC geometry with error handling
        try:
            pipe_dim = pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract diameter from pipe ID {pipe.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Map nominal pipe diameter to real internal diameter (in meters)
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

        Base.append_log(None, f"> Pipe properties:")
        Base.append_log(None, f"> {pipe_prop}")
        return pipe_prop
