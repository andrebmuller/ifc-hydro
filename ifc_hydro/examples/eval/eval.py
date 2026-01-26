"""
Evaluation script for ifc-hydro library.

This script is used for evaluating the ifc-hydro library against
test IFC models to verify calculation accuracy.
"""

import os
import sys

import ifcopenshell as ifc

from ifc_hydro import Base, Topology, Pressure


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """Main function for ifc-hydro evaluation."""
    # Configure log file for this run
    log_dir_input = input("Enter log directory (leave blank for script directory): ").strip()
    if not log_dir_input:
        log_dir_input = SCRIPT_DIR

    log_name_input = input("Enter log file name (leave blank for ifc-hydro.log): ").strip()
    Base.configure_log(Base, log_dir=log_dir_input, log_name=log_name_input)

    # Load IFC model
    ifc_file_path = input("Enter IFC file path (leave blank for 'eval-project.ifc'): ").strip()
    if not ifc_file_path:
        ifc_file_path = os.path.join(SCRIPT_DIR, 'eval-project.ifc')

    if not os.path.exists(ifc_file_path):
        print(f"ERROR: IFC file not found at path: {ifc_file_path}")
        sys.exit(1)

    try:
        model = ifc.open(ifc_file_path)
        Base.append_log(Base, f"> Successfully loaded IFC model from: {ifc_file_path}")
    except Exception as e:
        print(f"ERROR: Failed to open IFC file: {e}")
        sys.exit(1)

    # Initialize topology and calculate all paths
    try:
        topology = Topology(model)
        all_paths = topology.all_paths_finder()
    except Exception as e:
        print(f"ERROR: Topology creation failed: {e}")
        sys.exit(1)

    if not all_paths:
        print("ERROR: No paths were created. Cannot proceed with pressure calculations.")
        sys.exit(1)

    # Initialize pressure calculator
    pressure_calc = Pressure()

    # Calculate available pressure at terminals
    terminal_id_input = input("Enter terminal ID (leave blank to calculate all terminals): ").strip()

    if terminal_id_input:
        try:
            terminal = model.by_id(int(terminal_id_input))
            pressure_calc.available(terminal, all_paths)
        except RuntimeError:
            print(f"ERROR: Terminal with ID {terminal_id_input} not found in the IFC model.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to calculate pressure: {e}")
            sys.exit(1)
    else:
        terminals = model.by_type("IfcSanitaryTerminal")

        if not terminals:
            print("ERROR: No IfcSanitaryTerminal elements found in the IFC model.")
            sys.exit(1)

        for terminal in terminals:
            try:
                pressure_calc.available(terminal, all_paths)
            except Exception as e:
                print(f"ERROR: Failed to calculate pressure for terminal {terminal.id()}: {e}")
                continue


if __name__ == '__main__':
    main()
