from typing import Tuple
import numpy as np
import polars as pl
from icecream import ic

ic.disable()


class Filter:
    def __init__(self, pname: str, pthickness: float, p_is_element: bool) -> None:
        self.name = pname
        self.thickness = pthickness
        self.is_element = p_is_element


class Simulation:
    def __init__(self) -> None:
        self.fname_spectra = "./data/W-Spectra/Spectra_9-100.csv"
        self.fname_compounds = "./data/compounds.dat"
        self.fname_elements = [
            "./data/1-19.dat",
            "./data/20-69.dat",
            "./data/70-92.dat",
        ]
        self.fname_info_compounds = "./data/names_compounds.txt"
        self.fname_info_elements = "./data/names_elements.txt"

        self.load_data_bases()

        self.spectrum_list = []
        self.base_spectrum_energy = ""
        self.filter_list = []

    def load_data_bases(self):
        """Loads the data bases"""
        self.spectra_db = (
            pl.scan_csv(self.fname_spectra)
            .with_columns(pl.col("Energy[keV]").round(1))
            .collect()
        )
        self.mu_compounds = pl.scan_csv(self.fname_compounds, separator="\t").collect()
        df_elements_0 = pl.scan_csv(self.fname_elements[0], separator="\t").collect()
        df_elements_1 = pl.scan_csv(self.fname_elements[1], separator="\t").collect()
        df_elements_2 = pl.scan_csv(self.fname_elements[2], separator="\t").collect()

        self.mu_elements = df_elements_0.join(
            df_elements_1, on="Energy", how="left", coalesce=True
        )
        self.mu_elements = self.mu_elements.join(
            df_elements_2, on="Energy", how="left", coalesce=True
        )

        self.info_elements = pl.read_csv(self.fname_info_elements, separator="\t")
        self.info_compounds = pl.read_csv(self.fname_info_compounds, separator="\t")

    def set_base_spectrum(self, max_energy: str):
        """Returns a tuple with the given energy spectrum

        Args:
            energy (str): Maximum spectrum energy

        Returns:
            List[np.ndarray, np.ndarray]: Energy values of the spectrum, intensity of the spectrum
        """

        if len(self.spectrum_list) > 0:
            self.spectrum_list = []

        self.spectrum_list.append(
            self.spectra_db.select(pl.col("Energy[keV]"), pl.col(max_energy))
        )
        self.base_spectrum_energy = max_energy

    def get_base_spectrum(self):
        return self.spectrum_list[0]

    def get_current_spectrum(self):
        return self.spectrum_list[-1]

    def get_mass_attenuation(
        self, material_name: str, is_element: bool
    ) -> pl.DataFrame:
        """Based on the material name, it returns the mass_attenuation spectrum

        Args:
            material_name (str): Material name
            is_element (bool): If it is a pure element or a compound

        Returns:
            List[np.ndarray, np.ndarray]: Energy values of the spectrum, intensity of the spectrum (Already multiplied by density)
        """
        if is_element:
            return self.mu_elements.select(pl.col("Energy"), pl.col(material_name))
        return self.mu_compounds.select(pl.col("Energy"), pl.col(material_name))

    def calculate_transmited_spectrum(self, filter: Filter) -> None:

        spectrum = self.spectrum_list[-1]
        mu = self.get_mass_attenuation(filter.name, filter.is_element)

        emax = mu.select("Energy").max().item()
        # Filter values so both spectrum can coexist
        if spectrum.select("Energy[keV]").max().item() > emax:
            print(
                "WARNING::Input spectrum has more energy values than the mass-attenuation spectrum."
            )

        df = spectrum.join(mu, left_on="Energy[keV]", right_on="Energy", validate="1:1")
        df = (
            df.with_columns(
                (
                    pl.col(self.base_spectrum_energy)
                    * np.exp(-1 * pl.col(filter.name) * filter.thickness)
                ).alias(self.base_spectrum_energy)
            )
            .with_columns(
                pl.when(pl.col(self.base_spectrum_energy) < 1e-35)
                .then(0)
                .otherwise(pl.col(self.base_spectrum_energy))
                .alias(self.base_spectrum_energy)
            )
            .drop(filter.name)
        )
        ic(df)
        self.spectrum_list.append(df)

    def add_filter(
        self,
        material_name: str,
        material_thickness: float,
        is_element: bool,
        index: int = -1,
    ):
        if index >= len(self.filter_list) or len(self.filter_list) == 0:
            self.filter_list.append(
                Filter(material_name, material_thickness, is_element)
            )
            self.calculate_transmited_spectrum(self.filter_list[-1])
            return

        self.filter_list[index] = Filter(material_name, material_thickness, is_element)
        for i in range(index, len(self.filter_list)):
            self.calculate_transmited_spectrum(self.filter_list[i])

    def remove_filter(self, filter_indx: int):

        self.spectrum_list.pop(filter_indx + 1)
        self.filter_list.pop(filter_indx)
        for f in self.filter_list:
            self.calculate_transmited_spectrum(f)
