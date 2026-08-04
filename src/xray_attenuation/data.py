from pathlib import Path

import polars as pl

DATA_PATH = Path(__file__).parent / "data"


class MaterialNotFoundError(KeyError):
    """Raised when a material is not present in the attenuation tables"""


class Data:
    """Class for managing the data"""

    def __init__(self) -> None:
        self._load_data_elements()
        self._load_data_compounds()
        self._load_spectra()

    def _load_data_elements(self) -> None:
        """Loads the NIST photon attenuation coefficient ([mu]=1/cm) for pure
        elements, from the compilation files, as a Polars dataframe

        Returns:
        """
        df_elements_0 = pl.scan_csv(DATA_PATH / "1-19.dat", separator="\t")
        df_elements_1 = pl.scan_csv(DATA_PATH / "20-69.dat", separator="\t")
        df_elements_2 = pl.scan_csv(DATA_PATH / "70-92.dat", separator="\t")

        self.df_elements = df_elements_0.join(
            df_elements_1, on="Energy", how="left", coalesce=True
        )
        self.df_elements = self.df_elements.join(
            df_elements_2, on="Energy", how="left", coalesce=True
        )

        self.df_elements_names = pl.read_csv(
            DATA_PATH / "names_elements.txt", separator="\t"
        )

    def _load_data_compounds(self) -> None:
        """Loads the NIST photon attenuation coefficient ([mu]=1/cm) for some
        compounds, from the compilation files, as a Polars dataframe.

        Returns:
        """
        self.df_compounds = pl.scan_csv(DATA_PATH / "compounds.dat", separator="\t")
        self.df_compounds_names = pl.read_csv(
            DATA_PATH / "names_compounds.txt", separator="\t"
        )

    def _load_spectra(self) -> None:
        """Loads the tungsten x-ray spectra"""

        self.df_spectra = (
            pl.scan_csv(DATA_PATH / "W-Spectra/Spectra_9-100.csv")
            .with_columns(pl.col("Energy[keV]").round(1, mode="half_away_from_zero"))
            .collect()
        )

    def resolve_material_name(self, name: str) -> tuple[str, bool] | None:
        """Resolves a user-supplied material name to its canonical database name

        Accepts an element symbol ("al"), an element name ("aluminum") or a
        compound name ("cadmium telluride"), in any capitalisation.

        Args:
            name (str): user-supplied material name or element symbol

        Returns:
            tuple[str, bool] | None: (canonical name, is_compound) if the material
                is in the database, otherwise None
        """
        key = name.strip().lower()

        def normalize(column: str) -> pl.Expr:
            return pl.col(column).str.strip_chars().str.to_lowercase()

        elements = self.df_elements_names.filter(
            (normalize("Symbol") == key) | (normalize("Element") == key)
        )
        if elements.height:
            return elements["Element"][0], False

        compounds = self.df_compounds_names.filter(normalize("Material") == key)
        if compounds.height:
            return compounds["Material"][0], True

        return None

    def get_linear_attenuation_curve(
        self, material: str, is_compound: bool = False
    ) -> pl.DataFrame:
        """Returns the whole linear attenuation curve of a material

        Args:
            material (str): material name. Either from the elements or compounds lists
            is_compound (bool, optional): If the material is a compound.
                Defaults to False.

        Returns:
            pl.DataFrame: two columns, "Energy" [keV] and the material's mu [1/cm]

        Raises:
            MaterialNotFoundError: if the material is not in the corresponding table
        """
        frame = self.df_compounds if is_compound else self.df_elements

        if material not in frame.collect_schema().names():
            raise MaterialNotFoundError(material)

        return frame.select(pl.col("Energy"), pl.col(material)).collect()

    def get_linear_attenuation(
        self, material: str, energy: float, is_compound: bool = False
    ) -> float | None:
        """Returns the linear attenuation coefficient for a material at one energy

        Args:
            material (str): material name. Either from the elements or compounds lists
            energy (float): energy in keV. Must be an exact value of the energy grid
            is_compound (bool, optional): If the material is a compound.
                Defaults to False.

        Returns:
            float | None: The linear attenuation coefficient, or None if the energy
                is not on the grid

        Raises:
            MaterialNotFoundError: if the material is not in the corresponding table
        """
        curve = self.get_linear_attenuation_curve(material, is_compound)
        mu = curve.filter(pl.col("Energy") == energy)

        if mu.is_empty():
            return None

        return mu[material][0]

    def get_spectrum_list(self) -> list[str]:
        return self.df_spectra.columns[1:]
