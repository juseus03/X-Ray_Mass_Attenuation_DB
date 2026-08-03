import argparse
import sys
from data import Data
import numpy as np
import polars as pl

MAX_ENERGY = 200  # keV
MIN_ENERGY = 3  # keV


def set_arguments() -> argparse.ArgumentParser:
    """Sets arguments for the CLI tool

    Returns:
        argparse.ArgumentParser: The argument parser object
    """
    parser = argparse.ArgumentParser(
        description="X-Ray mass attenuation calculator for NIST elements and compounds"
    )

    parser.add_argument(
        "--material_name",
        "-m",
        nargs="?",
        help='Material name ("-" to show material list)',
        default=None,
    )

    parser.add_argument(
        "--thickness",
        "-t",
        type=float,
        help="Thickness [cm] of material",
        default=None,
    )

    parser.add_argument(
        "--energy",
        "-e",
        type=float,
        help=f"Photon energy ({MIN_ENERGY:.0f} keV - {MAX_ENERGY:.0f} keV)",
        default=None,
    )

    return parser


def get_user_input(input_name: str) -> float:
    """Prompts the user for either a thickness value or an energy value

    Args:
        input_name (str): thickness or energy, indicating which value to prompt for

    Returns:
        float: The user-supplied value for thickness or energy. -1 if the input_name is not recognized
    """

    if input_name.lower() == "thickness":
        sys.stderr.write("--- Material thickness [cm]: ")
        try:
            thickness = float(input(""))
            if thickness <= 0:
                print("Error: Thickness must be greater than 0")
                return get_user_input("thickness")
            return thickness
        except ValueError:
            print("Error: Please enter a valid number for thickness.")
            return get_user_input("thickness")

    if input_name.lower() == "energy":
        sys.stderr.write("--- Photon energy [keV]: ")
        try:
            energy = float(input(""))
            energy = np.round(energy, 1)
            if energy < 3 or energy > 200:
                print("Error: Energy not in the database (3 keV - 200 keV)")
                return get_user_input("energy")
            return energy
        except ValueError:
            print("Error: Please enter a valid number for energy.")
            return get_user_input("energy")
    return -1


def ask_for_materials(df_e_names: pl.DataFrame, df_c_names: pl.DataFrame) -> str:
    """User input for selection of a material in the data-base

    Args:
        df_e_names (pl.DataFrame): DataFrame containing the names of the elements in the database
        df_c_names (pl.DataFrame): DataFrame containing the names of the compounds in the database

    Returns:
        str: The name of the selected material
    """
    sys.stderr.write("\n--- Available Materials:\n")
    materials = []

    sys.stderr.write("\n--- Elements (Symbol):\n")
    for i, info_tuple in enumerate(df_e_names.rows()):
        e_symbol = info_tuple[1]
        e_name = info_tuple[2]
        sys.stderr.write(f"--- {i:2}: {e_name:20} ({e_symbol:2})\n")
        materials.append(e_name)

    sys.stderr.write("\n--- Compounds:\n")
    carry = len(materials)
    for i, info_tuple in enumerate(df_c_names.rows()):
        e_name = info_tuple[0]
        sys.stderr.write(f"--- {i + carry:2}: {e_name:20}\n")
        materials.append(e_name)

    materials = np.array(materials)

    while True:
        sys.stderr.write("--- Enter material index or full name: ")
        name = input("")
        index = -1
        try:
            index = int(name)
            if not 0 <= index < len(materials):
                sys.stderr.write("--- Invalid index!\n")
                continue
        except ValueError:
            pass
        else:
            name = materials[index]
        return name


def get_transmission(mu: float, thickness: float) -> float:
    """Calculates the linear transmission coefficient

    Args:
        mu (float): linear attenuation coefficient [1/cm]
        thickness (float): material thickness

    Returns:
        float: linear transmission coefficient
    """
    return np.exp(-1 * mu * thickness)


def main():
    """Entrypoint for the CLI tool"""

    parser = set_arguments()
    args = parser.parse_args()
    data = Data()

    # Get material name
    if args.material_name is None or args.material_name == "-":
        name = ask_for_materials(data.df_elements_names, data.df_compounds_names)
    else:
        name = args.material_name

    resolved = data.resolve_material_name(name)

    if resolved is None:
        sys.stderr.write(f"Error: '{name}' is not in the database\n")
        sys.exit(1)

    material_name, is_compound = resolved

    # Get thickness
    if args.thickness is None:
        thickness = get_user_input("thickness")
    elif args.thickness > 0:
        thickness = args.thickness
    else:
        print("Error: Thickness must be greater than 0")
        thickness = get_user_input("thickness")

    # Get energy
    if args.energy is None:
        energy = get_user_input("energy")
    elif args.energy >= MIN_ENERGY and args.energy <= MAX_ENERGY:
        energy = np.round(args.energy, 1)
    else:
        print(
            f"Error: Energy not in the database ({MIN_ENERGY:.0f} keV - {MAX_ENERGY:.0f} keV)"
        )
        energy = get_user_input("energy")

    # Get linear attenuation coefficient
    mu = data.get_linear_attenuation(material_name, energy, is_compound)

    if mu is None:
        sys.stderr.write(f"Error: no data for {energy} keV\n")
        sys.exit(1)

    # Calculate transmission
    transmission = np.round((get_transmission(mu, thickness) * 100), 2)
    print(
        f"For {thickness} cm of '{material_name}' the transmission of photons, with energy {energy} keV, is around {transmission} %"
    )


if __name__ == "__main__":
    main()
