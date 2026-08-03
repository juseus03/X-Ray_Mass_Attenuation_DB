import argparse
import sys

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
from icecream import ic
from xray_attenuation.data import Data
from xray_attenuation.physics import get_transmission, calculate_filtered_spectrum

DATA_PATH = Path(__file__).parent / "data"
plt.style.use(DATA_PATH / "presentation.mplstyle")

MAX_ENERGY = 200  # keV
MIN_ENERGY = 3  # keV
MAX_ENERGY_FULL_SPECTRUM = 100  # keV
MIN_ENERGY_FULL_SPECTRUM = 9  # keV


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

    parser.add_argument(
        "--full-spectrum",
        "-f",
        action="store_true",
        help="Calculate full spectrum",
        default=False,
    )

    return parser


class Filter:
    def __init__(self, pname: str, pthickness: float, p_is_compound: bool) -> None:
        self.name = pname
        self.thickness = pthickness
        self.is_compound = p_is_compound


class CLI:
    def __init__(self, is_full_spectrum: bool = False):
        self.data = Data()
        self.spectrum_list = {}
        self.filters = []
        self.is_full_spectrum = is_full_spectrum
        if self.is_full_spectrum:
            self.min_energy = MIN_ENERGY_FULL_SPECTRUM
            self.max_energy = MAX_ENERGY_FULL_SPECTRUM
        else:
            self.min_energy = MIN_ENERGY
            self.max_energy = MAX_ENERGY

    def get_user_input(self, input_name: str) -> float:
        """Prompts the user for either a thickness value or an energy value

        Args:
            input_name (str): thickness or energy, indicating which value to prompt for

        Returns:
            float: The user-supplied value for thickness or energy. -1 if the
                input_name is not recognized
        """

        if input_name.lower() == "thickness":
            sys.stderr.write("--- Material thickness [cm]: ")
            try:
                thickness = float(input(""))
                if thickness <= 0:
                    print("Error: Thickness must be greater than 0")
                    return self.get_user_input("thickness")
                return thickness
            except ValueError:
                print("Error: Please enter a valid number for thickness.")
                return self.get_user_input("thickness")

        if input_name.lower() == "energy":
            sys.stderr.write("--- Photon energy [keV]: ")
            try:
                energy = float(input(""))
                energy = np.round(energy, 1)
                if energy < 3 or energy > 200:
                    print("Error: Energy not in the database (3 keV - 200 keV)")
                    return self.get_user_input("energy")
                return energy
            except ValueError:
                print("Error: Please enter a valid number for energy.")
                return self.get_user_input("energy")
        return -1

    def ask_for_materials(self) -> str:
        """User input for selection of a material in the data-base
        Args:
            self: The CLI object
        Returns:
            str: The name of the selected material
        """

        df_e_names = self.data.df_elements_names
        df_c_names = self.data.df_compounds_names

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

    def filter_spectrum(
        self, material_name: str, energy: float, thickness: float, is_compound: bool
    ) -> None:
        filter = Filter(material_name, thickness, is_compound)

        mu = self.data.get_linear_attenuation(
            material_name, self.spectrum_list[str(int(energy))][:, 0], is_compound
        )
        ic(mu)

        base_spectrum = self.spectrum_list[str(int(energy))]
        filtered_spectrum = calculate_filtered_spectrum(base_spectrum, mu, thickness)
        self.add_spectrum(filtered_spectrum, filter)

    def add_base_spectrum(self, energy: float) -> None:
        energy_column_name = str(int(energy))
        spectrum = self.data.df_spectra.select(
            pl.col("Energy[keV]"), pl.col(energy_column_name)
        )
        self.spectrum_list[energy_column_name] = spectrum.to_numpy()
        self.filters.append(None)  # No filter for the base spectrum

    def add_spectrum(self, spectrum: pl.DataFrame, pfilter: Filter = None) -> None:
        energy_column_name = str(int(np.max(spectrum[:, 0]) + 1))
        self.spectrum_list[energy_column_name] = spectrum
        self.filters.append(pfilter)

    def plot_spectra(self) -> None:

        _, ax = plt.subplots()
        for dictuple, flt in zip(self.spectrum_list.items(), self.filters):
            ic(flt)
            key, spectrum = dictuple
            lbl = (
                f"{key} keV"
                if flt is None
                else f"{key} keV with {flt.thickness} cm of {flt.name}"
            )
            ax.plot(
                spectrum[:, 0],
                spectrum[:, 1],
                label=lbl,
            )
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Intensity [a.u.]")
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1)
        plt.legend()


def main():
    """Entrypoint for the CLI tool"""

    parser = set_arguments()
    args = parser.parse_args()

    is_full_spectrum = args.full_spectrum

    cli = CLI(is_full_spectrum)
    # Get material name
    if args.material_name is None or args.material_name == "-":
        name = cli.ask_for_materials()
    else:
        name = args.material_name

    resolved = cli.data.resolve_material_name(name)

    if resolved is None:
        sys.stderr.write(f"Error: '{name}' is not in the database\n")
        sys.exit(1)

    material_name, is_compound = resolved

    # Get thickness
    if args.thickness is None:
        thickness = cli.get_user_input("thickness")
    elif args.thickness > 0:
        thickness = args.thickness
    else:
        print("Error: Thickness must be greater than 0")
        thickness = cli.get_user_input("thickness")

    # Get energy
    if args.energy is None:
        energy = cli.get_user_input("energy")
    elif args.energy >= cli.min_energy and args.energy <= cli.max_energy:
        energy = np.round(args.energy, 1)
    else:
        print(
            f"Error: Energy not in the database "
            f"({cli.min_energy:.0f} keV - {cli.max_energy:.0f} keV)"
        )
        energy = cli.get_user_input("energy")

    if is_full_spectrum:
        cli.add_base_spectrum(energy)
        cli.filter_spectrum(material_name, energy, thickness, is_compound)
        cli.plot_spectra()
        plt.show()
        sys.exit(0)

    # Get linear attenuation coefficient
    mu = cli.data.get_linear_attenuation(material_name, energy, is_compound)

    if mu is None:
        sys.stderr.write(f"Error: no data for {energy} keV\n")
        sys.exit(1)

    # Calculate transmission
    transmission = np.round((get_transmission(mu, thickness) * 100), 2)
    print(
        f"For {thickness} cm of '{material_name}' the transmission of photons, "
        f"with energy {energy} keV, is around {transmission} %"
    )


if __name__ == "__main__":
    main()
