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
        nargs="+",
        help='Material name ("-" to show material list)',
        default=[],
    )

    parser.add_argument(
        "--thickness",
        "-t",
        nargs="+",
        type=float,
        help="Thickness [cm] of material",
        default=[],
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

    def __str__(self) -> str:
        return f"Filter({self.name},{self.thickness}cm)"


class CLI:
    def __init__(self, is_full_spectrum: bool = False):
        self.data = Data()
        self.spectrum_df = None
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
                if energy < self.min_energy or energy > self.max_energy:
                    print(
                        f"Error: Energy not in the database ({self.min_energy} keV - {self.max_energy} keV)"
                    )
                    return self.get_user_input("energy")
                return energy
            except ValueError:
                print("Error: Please enter a valid number for energy.")
                return self.get_user_input("energy")
        raise ValueError

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

    def add_filter(
        self, material_name: str, energy: float, thickness: float, is_compound: bool
    ) -> None:
        """Registers a filter in the program, and applies it to the last spectrum

        Args:
            material_name (str): filter material
            energy (float): maximum energy of the spectrum (keV)
            thickness (float): thickness of the filter
            is_compound (bool): flag if the filter is a compound or an element
        """
        if thickness <= 0:
            print("WARNING: Thickness must be > 0, no filter added")
            return

        f = Filter(material_name, thickness, is_compound)
        self.filter_spectrum(energy, f)
        self.filters.append(f)

    def filter_spectrum(self, energy: float, pfilter: Filter) -> None:
        """Calculates the filtered spectrum by accumulation. i.e. it filters
        the last added spectrum

        Args:
            energy (float): maximum energy of the spectrum (keV)
            pfilter (Filter): filter object
        """

        mu_curve = self.data.get_linear_attenuation_curve(
            pfilter.name, pfilter.is_compound
        )

        self.spectrum_df = self.spectrum_df.join(
            mu_curve, left_on="Energy[keV]", right_on="Energy", validate="1:1"
        )
        mu = self.spectrum_df.select(pfilter.name).to_numpy().flatten()

        # Makes spectrum acumulative, so the last filter is applied to the last filtered spectrum
        if len(self.filters) == 0:
            spectrum = (
                self.spectrum_df.select(str(int(np.round(energy, 0))))
                .to_numpy()
                .flatten()
            )
        else:
            spectrum = (
                self.spectrum_df.select(
                    f"{len(self.filters)}_{self.filters[-1].name}_{self.filters[-1].thickness}cm"
                )
                .to_numpy()
                .flatten()
            )

        filtered_spectrum = pl.Series(
            calculate_filtered_spectrum(spectrum, mu, pfilter.thickness)
        )
        self.spectrum_df = self.spectrum_df.with_columns(
            pl.when(filtered_spectrum < 1e-35)
            .then(0)
            .otherwise(filtered_spectrum)
            .alias(f"{len(self.filters)+1}_{pfilter.name}_{pfilter.thickness}cm")
        ).drop(pfilter.name)

    def remove_filter(self, filter_index: int) -> None:
        """Removes a filter and updates the current state of the spectrum df

        Args:
            filter_index (int): filter index to remove
        """

        if not (0 <= filter_index < len(self.filters)):
            print(f"ERROR: Filter index {filter_index} doesn't exist")
            return

        old_columns = self.spectrum_df.columns.copy()

        if filter_index == len(self.filters) - 1:
            self.filters.pop(filter_index)
            self.spectrum_df.drop_in_place(old_columns[-1])
            return

        new_filters = self.filters[filter_index + 1 :]
        self.filters = self.filters[:filter_index]
        self.spectrum_df = self.spectrum_df.drop(
            old_columns[filter_index + 2 :]
        )  # +2 because of Energy and base spectrum columns

        for f in new_filters:
            self.add_filter(f.name, float(old_columns[1]), f.thickness, f.is_compound)

    def add_base_spectrum(self, energy: float) -> None:
        """Adds the first and unfiltered energy spectrum

        Args:
            energy (float): Maximum value of the energy spectrum in keV
        """
        energy_column_name = str(int(np.round(energy, 0)))
        spectrum = self.data.df_spectra.select(
            pl.col("Energy[keV]"), pl.col(energy_column_name)
        )
        self.spectrum_df = spectrum

    def plot_spectra(self) -> None:
        """Plots all the spectra in the current spectrum DF"""

        _, ax = plt.subplots()
        for i, f in enumerate(self.spectrum_df.columns):
            if f == "Energy[keV]":
                continue
            if i == 1:
                lbl = f"{f} keV"
            else:
                flt = self.filters[i - 2]
                lbl = f"{flt.name} - {flt.thickness} cm"
            ax.plot(
                self.spectrum_df["Energy[keV]"],
                self.spectrum_df[f],
                label=lbl,
            )
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Intensity [a.u.]")
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1e-5)
        plt.legend()


def _get_user_energy(cli: CLI, args) -> float:

    if args.energy is None:
        return cli.get_user_input("energy")
    if args.energy >= cli.min_energy and args.energy <= cli.max_energy:
        return np.round(args.energy, 1)
    print(
        f"Error: Energy not in the database "
        f"({cli.min_energy:.0f} keV - {cli.max_energy:.0f} keV)"
    )
    return cli.get_user_input("energy")


def _get_single_material_name(cli: CLI, material_name: str) -> tuple[str, bool] | None:
    if material_name is None or material_name == "-":
        name = cli.ask_for_materials()
    else:
        name = material_name

    return cli.data.resolve_material_name(name)


# def _get_single_thickness(cli:CLI, thickness:float)->float


def run_single_value(cli: CLI, args) -> None:
    """Single value calculation for transmission

    Args:
        cli (CLI): CLI object
        args (_type_): cmd line arguments object
    """
    # Get material name
    if (
        args.material_name is None
        or len(args.material_name) == 0
        or args.material_name[0] == "-"
    ):
        name = cli.ask_for_materials()
    else:
        name = args.material_name[0]

    resolved = cli.data.resolve_material_name(name)

    if resolved is None:
        sys.stderr.write(f"Error: '{name}' is not in the database\n")
        sys.exit(1)

    material_name, is_compound = resolved

    # Get thickness
    if args.thickness is None or len(args.thickness) == 0:
        thickness = cli.get_user_input("thickness")
    elif args.thickness[0] > 0:
        thickness = args.thickness[0]
    else:
        print("Error: Thickness must be greater than 0")
        thickness = cli.get_user_input("thickness")

    # Get energy
    energy = _get_user_energy(cli, args)

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


def main():
    """Entrypoint for the CLI tool"""

    parser = set_arguments()
    args = parser.parse_args()

    is_full_spectrum = args.full_spectrum

    cli = CLI(is_full_spectrum)

    # Single transmission value
    if not is_full_spectrum:
        if len(args.thickness) > 1 or len(args.material_name) > 1:
            print("ERROR: For multiple filter/material values use the option -f")
            exit(-1)
        run_single_value(cli, args)
        return

    # Full spectrum transmission w/ filters
    energy = _get_user_energy(cli, args)
    cli.add_base_spectrum(energy)

    nfilters = max(len(args.material_name), len(args.thickness))
    print(f"Calculating {nfilters} filters")

    if len(args.material_name) == len(args.thickness):
        for m, t in zip(args.material_name, args.thickness):
            resolved = _get_single_material_name(cli, m)
            if resolved is None:
                sys.stderr.write(f"Error: '{m}' is not in the database\n")
                sys.exit(1)
            mname, is_compound = resolved
            cli.add_filter(mname, energy, t, is_compound)
    else:
        # Map n->m:
        for m in args.material_name:
            resolved = _get_single_material_name(cli, m)
            if resolved is None:
                sys.stderr.write(f"Error: '{m}' is not in the database\n")
                sys.exit(1)
            mname, is_compound = resolved
            for t in args.thickness:
                cli.add_filter(mname, energy, t, is_compound)

    cli.plot_spectra()
    plt.show()
    sys.exit(0)


if __name__ == "__main__":
    main()
