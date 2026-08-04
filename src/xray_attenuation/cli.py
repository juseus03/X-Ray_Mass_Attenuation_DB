import argparse
import re
import sys

import numpy as np
import polars as pl
import tempfile

from datetime import datetime
from pathlib import Path
from xray_attenuation.data import Data
from xray_attenuation.physics import get_transmission, calculate_filtered_spectrum
from dataclasses import dataclass

DATA_PATH = Path(__file__).parent / "data"
TMP_PATH = Path(tempfile.gettempdir())


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

    parser.add_argument(
        "--save-plot",
        "-s",
        action="store_true",
        help="Save plot",
        default=False,
    )

    return parser


def _sanitize_for_filename(name: str) -> str:
    """Turns a material name into a filename-safe token

    Args:
        name (str): material name, possibly with spaces, commas or parentheses

    Returns:
        str: the name with every run of unsupported characters collapsed into a
            single underscore
    """
    return re.sub(r"[^\w.-]+", "_", name.strip()).strip("_")


@dataclass
class Filter:
    """Class to model a material filter"""

    name: str
    thickness: float
    is_compound: bool


class CLI:
    def __init__(self, is_full_spectrum: bool = False, save_plot: bool = False):
        self.data = Data()
        self.spectrum_df: pl.DataFrame | None = None
        self.max_kv: str | None = None
        self.filters: list[Filter] = []
        self.is_full_spectrum = is_full_spectrum
        self.save_plot = save_plot

        if self.is_full_spectrum:
            self.min_energy = MIN_ENERGY_FULL_SPECTRUM
            self.max_energy = MAX_ENERGY_FULL_SPECTRUM
        else:
            self.min_energy = MIN_ENERGY
            self.max_energy = MAX_ENERGY

    def get_user_input(self, input_name: str) -> float:
        """Prompts the user for either a thickness value or an energy value

        Re-prompts until the value is valid, so a stream of bad input costs nothing.

        Args:
            input_name (str): thickness or energy, indicating which value to prompt for

        Returns:
            float: The user-supplied value for thickness or energy

        Raises:
            ValueError: if input_name is neither "thickness" nor "energy"
        """

        kind = input_name.lower()
        if kind not in ("thickness", "energy"):
            raise ValueError(
                f"Unknown input '{input_name}', expected 'thickness' or 'energy'"
            )

        while True:
            if kind == "thickness":
                sys.stderr.write("--- Material thickness [cm]: ")
            else:
                sys.stderr.write("--- Photon energy [keV]: ")

            try:
                value = float(input(""))
            except ValueError:
                print(f"Error: Please enter a valid number for {kind}.")
                continue

            if kind == "thickness":
                if value <= 0:
                    print("Error: Thickness must be greater than 0")
                    continue
                return value

            energy = np.round(value, 1)
            if energy < self.min_energy or energy > self.max_energy:
                print(
                    f"Error: Energy not in the database "
                    f"({self.min_energy} keV - {self.max_energy} keV)"
                )
                continue
            return energy

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

        # Inner join on purpose: it trims the spectrum's 0.1-2.9 keV rows, which the
        # NIST tables (3-200 keV) have no mu for. 1000 rows in, 971 out
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
            self.add_filter(f.name, float(self.max_kv), f.thickness, f.is_compound)

    def add_base_spectrum(self, energy: float) -> None:
        """Adds the first and unfiltered energy spectrum

        Args:
            energy (float): Maximum value of the energy spectrum in keV
        """
        energy_column_name = str(int(np.round(energy, 0)))

        # Spectra exist only at integer kV, so anything else is snapped. np.round is
        # half-to-even, hence 60.5 -> 60 but 61.5 -> 62; say so rather than snap quietly
        if float(energy_column_name) != energy:
            print(
                f"NOTICE: No {energy} kV spectrum in the database, "
                f"using the {energy_column_name} kV one instead"
            )

        spectrum = self.data.df_spectra.select(
            pl.col("Energy[keV]"), pl.col(energy_column_name)
        )
        self.spectrum_df = spectrum
        self.max_kv = energy_column_name

    def build_plot_path(self) -> Path:
        """Builds a readable, collision-free PNG path in the system temp directory

        Returns:
            Path: temp-directory path named after the tube voltage, the filters
                and the current time
        """
        parts = ["xray_spectrum", f"{self.max_kv}kV"]
        parts += [
            f"{_sanitize_for_filename(f.name)}-{f.thickness:g}cm" for f in self.filters
        ]
        stem = "_".join(parts)

        # Stay well inside the 255 byte filename limit when many filters are stacked
        if len(stem) > 180:
            stem = f"xray_spectrum_{self.max_kv}kV_{len(self.filters)}filters"

        return TMP_PATH / f"{stem}_{datetime.now():%Y%m%d-%H%M%S}.png"

    def plot_spectra(self) -> None:
        """Plots all the spectra in the current spectrum DF"""

        import matplotlib.pyplot as plt

        if self.spectrum_df is None:
            print("ERROR: No spectrum to plot, call add_base_spectrum first")
            return

        # style.context rather than style.use: the style stays on this figure instead of
        # leaking into the global rcParams of whatever else imports this module
        with plt.style.context(DATA_PATH / "presentation.mplstyle"):
            fig, ax = plt.subplots()
            for i, f in enumerate(self.spectrum_df.columns):
                if f == "Energy[keV]":
                    continue
                if i == 1:
                    lbl = f"{f} kV"
                else:
                    flt = self.filters[i - 2]
                    lbl = f"+ {flt.name} {flt.thickness} cm"
                ax.plot(
                    self.spectrum_df["Energy[keV]"],
                    self.spectrum_df[f],
                    label=lbl,
                )
            ax.set_xlabel("Energy [keV]")
            ax.set_ylabel("Intensity [a.u.]")
            ax.set_yscale("log")
            ax.set_ylim(1e-10, 1e-5)
            ax.legend()

            if self.save_plot:
                path = self.build_plot_path()
                try:
                    fig.savefig(path)
                except OSError as exc:
                    sys.stderr.write(f"Error: could not save plot to {path} ({exc})\n")
                else:
                    print(f"Plot saved to: {path}")


def _get_user_energy(cli: CLI, args: argparse.Namespace) -> float:
    """Interactive mode for getting the energy value from the user

    Args:
        cli (CLI): cli object
        args (argparse.Namespace): parsed command line arguments

    Returns:
        float: energy value
    """
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
    """Interactive mode for getting the material name from the user

    Args:
        cli (CLI): cli object
        material_name (str): given material name from the user

    Returns:
        tuple[str, bool] | None: (canonical name, is_compound) if the material is in the
            database, otherwise None
    """
    if material_name is None or material_name == "-":
        name = cli.ask_for_materials()
    else:
        name = material_name

    return cli.data.resolve_material_name(name)


def run_single_value(cli: CLI, args: argparse.Namespace) -> None:
    """Single value calculation for transmission

    Args:
        cli (CLI): CLI object
        args (argparse.Namespace): parsed command line arguments
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


def main() -> None:
    """Entrypoint for the CLI tool"""

    parser = set_arguments()
    args = parser.parse_args()

    is_full_spectrum = args.full_spectrum

    cli = CLI(is_full_spectrum, args.save_plot)

    if args.save_plot and not is_full_spectrum:
        print("WARNING: --save_plot only applies to full-spectrum mode (-f), ignoring")

    # Single transmission value
    if not is_full_spectrum:
        if len(args.thickness) > 1 or len(args.material_name) > 1:
            print("ERROR: For multiple filter/material values use the option -f")
            sys.exit(1)
        run_single_value(cli, args)
        return

    # Full spectrum transmission w/ filters
    import matplotlib.pyplot as plt

    if len(args.thickness) == 0 or len(args.material_name) == 0:
        print("WARNING: At least one thickness and one material should be passed")
        print("\n Plotting unfiltered spectrum")

    energy = _get_user_energy(cli, args)
    cli.add_base_spectrum(energy)

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

    # Counted after the fact: the two branches build different numbers of filters, and
    # add_filter skips any with a non-positive thickness
    print(f"Calculated {len(cli.filters)} filters")

    cli.plot_spectra()
    plt.show()
    sys.exit(0)


if __name__ == "__main__":
    main()
