"""
Basic usage example for ifc-hydro library.

This example demonstrates how to use the ifc-hydro library to analyze hydraulic systems from IFC models.
"""

import ifcopenshell as ifc
from ifc_hydro import Base, Topology, Pressure
import sys
import os


# Get the directory where this script is located for portable paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """
    Main function demonstrating usage of ifc-hydro library.
    """

    # Configure log file for this run
    log_dir_input = input("Enter log directory (leave blank for current directory): ").strip()
    if not log_dir_input:
        log_dir_input = SCRIPT_DIR

    log_name_input = input("Enter log file name (leave blank for demo.log): ").strip()
    if not log_name_input:
        log_name_input = "demo"
    Base.configure_log(Base, log_dir=log_dir_input, log_name=log_name_input)

    # Load IFC model
    ifc_file_path = input("Enter IFC file path (leave blank for 'demo-project.ifc'): ").strip()
    if not ifc_file_path:
        ifc_file_path = os.path.join(SCRIPT_DIR, "demo-project.ifc")

    # Validate IFC file exists
    if not os.path.exists(ifc_file_path):
        error_msg = f"ERROR: IFC file not found at path: {ifc_file_path}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Load IFC model with error handling
    try:
        model = ifc.open(ifc_file_path)
        Base.append_log(Base, f"> Successfully loaded IFC model from: {ifc_file_path}")
    except Exception as e:
        error_msg = f"ERROR: Failed to open IFC file: {str(e)}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Initialize topology creator with the model and calculate all paths
    try:
        topology = Topology(model)
        all_paths = topology.all_paths_finder()
    except ValueError as e:
        error_msg = f"ERROR: Topology creation failed: {str(e)}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        Base.append_log(Base, "Program halted due to topology errors.")
        sys.exit(1)
    except Exception as e:
        error_msg = f"ERROR: Unexpected error during topology creation: {str(e)}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Validate that paths were created
    if not all_paths or len(all_paths) == 0:
        error_msg = "ERROR: No paths were created. Cannot proceed with pressure calculations."
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Initialize pressure calculator
    pressure_calc = Pressure()

    # Get terminal ID from user or calculate for all terminals
    terminal_id_input = input("Enter terminal ID (leave blank to calculate all terminals): ").strip()

    if terminal_id_input:
        # Calculate pressure for a specific terminal
        try:
            terminal = model.by_id(int(terminal_id_input))
            pressure_calc.available(terminal, all_paths)
        except RuntimeError:
            error_msg = f"ERROR: Terminal with ID {terminal_id_input} not found in the IFC model."
            Base.append_log(Base, error_msg)
            print(error_msg)
            sys.exit(1)
        except Exception as e:
            error_msg = f"ERROR: Failed to calculate pressure: {str(e)}"
            Base.append_log(Base, error_msg)
            print(error_msg)
            sys.exit(1)
    else:
        # Calculate pressure for all sanitary terminals
        terminals = model.by_type("IfcSanitaryTerminal")

        if not terminals:
            error_msg = "ERROR: No IfcSanitaryTerminal elements found in the IFC model."
            Base.append_log(Base, error_msg)
            print(error_msg)
            sys.exit(1)

        for terminal in terminals:
            try:
                pressure_calc.available(terminal, all_paths)
            except Exception as e:
                error_msg = f"ERROR: Failed to calculate pressure for terminal {terminal.id()}: {str(e)}"
                Base.append_log(Base, error_msg)
                print(error_msg)
                continue


if __name__ == '__main__':
    main()
