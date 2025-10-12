from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from icecream import ic

ic.disable()

plt.style.use("code/presentation.mplstyle")


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

    def get_spectrum(self, max_energy: str) -> pl.DataFrame:
        """Returns a tuple with the given energy spectrum

        Args:
            energy (str): Maximum spectrum energy

        Returns:
            List[np.ndarray, np.ndarray]: Energy values of the spectrum, intensity of the spectrum
        """
        return self.spectra_db.select(pl.col("Energy[keV]"), pl.col(max_energy))

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

    def calculate_transmited_spectrum(
        self,
        spectrum_energy: str,
        material_name: str,
        material_thickness: float,
        is_element: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:

        spectrum = self.get_spectrum(spectrum_energy)
        mu = self.get_mass_attenuation(material_name, is_element)

        emax = mu.select("Energy").max().item()
        # Filter values so both spectrum can coexist
        if spectrum.select("Energy[keV]").max().item() > emax:
            print(
                "WARNING::Input spectrum has more energy values than the mass-attenuation spectrum."
            )

        df = spectrum.join(mu, left_on="Energy[keV]", right_on="Energy", validate="1:1")
        ic(df)
        df = df.with_columns(
            (
                pl.col(spectrum_energy)
                * np.exp(-1 * pl.col(material_name) * material_thickness)
            ).alias("Transmitted")
        ).with_columns(
            pl.when(pl.col("Transmitted") < 1e-35)
            .then(0)
            .otherwise(pl.col("Transmitted"))
            .alias("Transmitted")
        )
        ic(df)
        energy = df.select("Energy[keV]").to_numpy().reshape(-1)
        intensity = df.select("Transmitted").to_numpy().reshape(-1)
        return energy, intensity


# thickness = 0.1  # cm
# material = "Aluminum"
# spectrum = "95"
# sim = Simulation()
# s_spectrum = sim.get_spectrum(spectrum)
# energy = s_spectrum["Energy[keV]"].to_numpy().reshape(-1)
# intensity = s_spectrum[spectrum].to_numpy().reshape(-1)

# energy2, intensity2 = sim.calculate_transmited_spectrum(
#     spectrum, material, thickness, True
# )

# _, ax = plt.subplots(1)
# ax.plot(energy, intensity, label="Open Beam")
# ax.plot(energy2, intensity2, "k--", label=f"{material} ({thickness} cm)")

# ax.set_xlabel("Energy [keV]")
# ax.set_ylabel("Counts")
# ax.set_yscale("log")
# # ax.set_xlim((0, 100))
# ax.set_ylim((1e-13, 1e-5))
# plt.legend()
# plt.show()
