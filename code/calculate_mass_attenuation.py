import sys

import numpy as np
import polars as pl


def load_data():
    df_elements_0 = pl.scan_csv("data\\1-19.dat", separator="\t")
    df_elements_1 = pl.scan_csv("data\\20-69.dat", separator="\t")
    df_elements_2 = pl.scan_csv("data\\70-92.dat", separator="\t")

    df_elements = df_elements_0.join(df_elements_1, on="Energy", how="left")
    df_elements = df_elements.join(df_elements_2, on="Energy", how="left")

    df_elements_names = pl.read_csv("data\\names_elements.txt", separator="\t")

    return df_elements, df_elements_names


def get_mass_attenuation(
    df_elements: pl.DataFrame, element_name: str, energy: float
) -> float:
    try:
        mu = (
            df_elements.select(pl.col("Energy"), pl.col(element_name))
            .filter(pl.col("Energy") == energy)
            .collect()
        )[element_name][0]

    except pl.exceptions.ColumnNotFoundError:
        print(f"Error: '{element_name}' column not in data")
        exit(0)

    return mu


def main():
    df_elements, df_elements_names = load_data()

    sys.stderr.write("--- Material name or Symbol: ")
    element_name = input("")

    sys.stderr.write("--- Material thickness [cm]: ")
    thickness = float(input(""))

    sys.stderr.write("--- Photon energy [keV]: ")
    energy = float(input(""))
    energy = np.round(energy, 1)

    if energy < 3 and energy > 200:
        print("Error: Energy not in the database")
        exit(0)

    if 0 < len(element_name) <= 2:
        element_name = df_elements_names.filter(pl.col("Symbol") == element_name)[
            "Element"
        ][0]

    mu = get_mass_attenuation(df_elements, element_name, energy)

    transmission = np.round(np.exp(-1 * mu * thickness) * 100, 2)

    print(
        f"For {thickness} cm of '{element_name}' the transmission of photons, with energy {energy} keV, is around {transmission} %"
    )


if __name__ == "__main__":
    main()
