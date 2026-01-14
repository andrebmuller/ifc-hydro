"""
Basic usage example for ifc-hydro library.

This example demonstrates how to use the ifc-hydro library to analyze hydraulic systems from a synthetic IFC modes.
"""

import ifcopenshell as ifc
from ifc_hydro import Base, Topology, Pressure


def main():
    """
    Main function demonstrating usage of ifc-hydro library.
    """

    # Configure log file for this run
    log_dir_input = input("Enter log directory (leave blank for current directory): ").strip()
    if not log_dir_input:
        log_dir_input = '.\ifc_hydro\examples'

    log_name_input = input("Enter log file name (leave blank for ifc-hydro.log): ").strip()
    Base.configure_log(Base, log_dir=log_dir_input, log_name=log_name_input)

    # Load IFC model
    ifc_file_path = input("Enter IFC file path (leave blank for 'demo-project.ifc'): ").strip()
    if not ifc_file_path:
        ifc_file_path = '.\ifc_hydro\examples\demo-project.ifc'

    model = ifc.open(ifc_file_path)

    # Initialize topology creator with the model and calculate all paths
    topology = Topology(model)
    test_path = topology.all_paths_finder()

    # Initialize pressure calculator
    pressure_calc = Pressure()

    # Example: Calculate available pressure at a specific terminal
    # Get terminal ID from user or use default
    terminal_id_input = input("Enter terminal ID (leave blank to calculate all terminals): ").strip()
    if terminal_id_input:

        # Default terminal IDs from the 'projeto-demonstracao.ifc' model:
        # Shower         --> 5423
        # Wash and Basin --> 6986
        # WC Seat        --> 7061
        term_test = model.by_id(int(terminal_id_input))

    else:
        
        terminals = model.by_type("IfcSanitaryTerminal")
        
        for terminal in terminals:
            # Get step numerical identifier for the terminal
            terminal_id = terminal.id()

            # Calculate available pressure at the terminal
            term_test = model.by_id(terminal_id)
            press_test = pressure_calc.available(term_test, test_path)

if __name__ == '__main__':
    main()
