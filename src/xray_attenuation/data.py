from pathlib import Path
from typing import Tuple
from icecream import ic

import polars as pl

DATA_PATH = Path(__file__).parent / "data"


class Data:
    """Class for managing the data"""

    def __init__(self) -> None:
        self.load_data_elements()
        self.load_data_compounds()

    def load_data_elements(self) -> None:
        """Loads the NIST photon attenuation coefficient ([mu]=1/cm) for pure elements, from the
        compilation files, as a Polars dataframe

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

    def load_data_compounds(self) -> None:
        """Loads the NIST photon attenuation coefficient ([mu]=1/cm) for some compounds, from the compilation files, as a Polars dataframe.

        Returns:
        """
        self.df_compounds = pl.scan_csv(DATA_PATH / "compounds.dat", separator="\t")
        self.df_compounds_names = pl.read_csv(
            DATA_PATH / "names_compounds.txt", separator="\t"
        )


data = Data()

ic(data.df_elements.collect())
ic(data.df_elements.collect().shape)
ic(data.df_elements_names)
ic(data.df_compounds.collect())
ic(data.df_compounds_names)
