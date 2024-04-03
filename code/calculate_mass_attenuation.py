import argparse
import sys
from typing import Tuple

import numpy as np
import polars as pl

# from icecream import ic
from icecream import ic

ic.disable()


def load_data_elements() -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_elements_0 = pl.scan_csv("data\\1-19.dat", separator="\t")
    df_elements_1 = pl.scan_csv("data\\20-69.dat", separator="\t")
    df_elements_2 = pl.scan_csv("data\\70-92.dat", separator="\t")

    df_elements = df_elements_0.join(df_elements_1, on="Energy", how="left")
    df_elements = df_elements.join(df_elements_2, on="Energy", how="left")

    df_elements_names = pl.read_csv("data\\names_elements.txt", separator="\t")

    return df_elements, df_elements_names


def load_data_compounds() -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_compounds = pl.scan_csv("data\\compounds.dat", separator="\t")
    df_compounds_names = pl.read_csv("data\\names_compounds.txt", separator="\t")
    return df_compounds, df_compounds_names


def get_mass_attenuation(
    df_elements: pl.DataFrame, element_name: str, energy: float
) -> Tuple[float, bool]:
    try:
        mu = (
            df_elements.select(pl.col("Energy"), pl.col(element_name))
            .filter(pl.col("Energy") == energy)
            .collect()
        )[element_name][0]

    except pl.exceptions.ColumnNotFoundError:
        return None, False

    return mu, True


def get_user_input(
    test: bool = False,
    thickness: float = 0.1,
    energy: float = 32,
) -> Tuple[str, float, float]:
    if test:
        return thickness, energy

    sys.stderr.write("--- Material thickness [cm]: ")
    thickness = float(input(""))

    sys.stderr.write("--- Photon energy [keV]: ")
    energy = float(input(""))
    energy = np.round(energy, 1)

    if energy < 3 or energy > 200:
        print("Error: Energy not in the database (3 keV - 200 keV)")
        exit(0)

    return thickness, energy


def set_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="X-Ray mass attenuation calculator for NIST elements and compounds"
    )

    parser.add_argument(
        "material_name",
        nargs="?",
        help='Material name ("-" to show material list)',
        default=None,
    )

    parser.add_argument(
        "thickness",
        nargs="?",
        help="Thickness [cm] of material",
        default=0,
    )

    parser.add_argument(
        "energy",
        nargs="?",
        help="Photon energy (3 keV - 200 keV)",
        default=None,
    )

    return parser


def ask_for_materials(df_e_names: pl.DataFrame, df_c_names: pl.DataFrame) -> str:
    sys.stderr.write("\n--- Available Materials:\n")
    materials = []

    sys.stderr.write("\n--- Elements (Symbol):\n")
    for i, info_tuple in enumerate(df_e_names.rows()):
        e_symbol = info_tuple[1]
        e_name = info_tuple[2]
        sys.stderr.write("--- {:2}: {:20} ({:2})\n".format(i, e_name, e_symbol))
        materials.append(e_name)

    sys.stderr.write("\n--- Compounds:\n")
    carry = len(materials)
    for i, info_tuple in enumerate(df_c_names.rows()):
        e_name = info_tuple[0]
        sys.stderr.write("--- {:2}: {:20}\n".format(i + carry, e_name))
        materials.append(e_name)

    materials = np.array(materials)

    while True:
        sys.stderr.write("--- Enter material index or full name: ")
        name = input("")
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


def main():
    df_elements, df_elements_names = load_data_elements()
    df_compounds, df_compounds_names = load_data_compounds()

    parser = set_arguments()
    args = parser.parse_args()

    if args.material_name is None or args.material_name == "-":
        material_name = ask_for_materials(df_elements_names, df_compounds_names)

    elif args.material_name is not None and len(args.material_name) <= 2:
        try:
            material_name = (
                df_elements_names.filter(pl.col("Symbol") == args.material_name)
                .select("Element")
                .to_series()
            )[0]
            ic(material_name)
        except IndexError:
            print(f"Symbol {args.material_name} not in the data base")
            exit(0)
    else:
        material_name = args.material_name

    thickness, energy = get_user_input(test=False)
    ic(material_name)
    mu, is_in_db = get_mass_attenuation(df_elements, material_name, energy)
    if not is_in_db:
        mu, is_in_db = get_mass_attenuation(df_compounds, material_name, energy)

    if not is_in_db:
        print(f"Warning: '{material_name}' column not in data")
        exit(0)

    transmission = np.round(np.exp(-1 * mu * thickness) * 100, 2)
    print(
        f"For {thickness} cm of '{material_name}' the transmission of photons, with energy {energy} keV, is around {transmission} %"
    )


if __name__ == "__main__":
    main()
