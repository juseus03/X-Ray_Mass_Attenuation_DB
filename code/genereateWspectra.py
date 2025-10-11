import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from icecream import ic

plt.style.use("code/presentation.mplstyle")

fname = "data/W-Spectra/data_40-50.txt"
data = np.loadtxt(
    fname,
    dtype={"names": ("energy", "E1", "E2"), "formats": ("f4", "f4", "f4")},
    delimiter=",",
)
intensity = data["E1"][data["energy"] >= 3]
energy = np.round(data["energy"][data["energy"] >= 3] - 0.05, 2)
energy_key = np.rint(energy * 100).astype(np.int32)


fname_mu = "data/compounds.dat"
df_mu = pl.scan_csv(fname_mu, separator="\t")
element_name = "Polymethyl Methacrylate, (PMMA)"
mu_pmma = (
    df_mu.with_columns(
        (pl.col("Energy") * 100).round(0).cast(pl.Int32).alias("energy_key")
    )
    .filter(pl.col("energy_key").is_in(pl.Series(energy_key)))
    .select(pl.col("Energy"), pl.col(element_name))
    .collect()
)[element_name]

ic(mu_pmma.shape)
thickness = 0.5  # cm
intensity_pmma = intensity * np.exp(-1 * mu_pmma * thickness)

_, ax = plt.subplots(1)
ax.plot(energy, intensity, label="Open Beam")
ax.plot(energy, intensity_pmma, "k--", label="PMMA")
ax.set_xlabel("Energy [keV]")
ax.set_ylabel("Counts")
ax.set_yscale("log")
# ax.set_xlim((0, 100))
# ax.set_ylim((1, 1e6))
plt.legend()
plt.show()
