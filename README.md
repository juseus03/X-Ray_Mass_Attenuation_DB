# X-Ray Attenuation

Software for calculation of the transmission coefficient for photons based on the NIST tables for X-rays Mass Attenuation Coefficients [1].

The main calculations are based on the Beer-lambert law, where the expected intensity $I$ of photons after traversing a material with thickness $t$ is:

$$I = I_0 \times exp(-\mu\times t)$$

where $I_0$ is the original intensity of photons, and $\mu$ the linear-mass attenuation coefficient for the specific material.

This software lets you calculate and visualize two things:
1. What percentage of photons survives x-cm of a material ?
2. How much of a 9 kV - 100 kV tungsten spectrum survives x-cm of a material?

## Installation

Requires Python >= 3.11. The `environment.yml` file is a thin conda bootstrap: it provides the interpreter and then pip-installs the package.

```bash
conda env create -f environment.yml -p ./envs
./envs/bin/pip install -e ".[dev]"
```

Or, into an environment you already have:

```bash
pip install -e .
```

Either way the `xray-transmission` command ends up on your PATH. The `[dev]` extra adds `pytest` and `ruff`; without it you only get the runtime dependencies.

## Usage
### CLI

The **cli** tool can be used in two ways. For a single energy, material, and thickness calculation:

```bash
xray-transmission # For interactive choose of material, thickness, and energy
# or
xray-transmission -m Al -t 0.1 -e 30 # For 1mm Al at 30 keV
```

which prints:

```
For 0.1 cm of 'Aluminum' the transmission of photons, with energy 30.0 keV, is around 73.52 %
```

Material names with spaces or commas have to be quoted:

```bash
xray-transmission -m "Cadmium Telluride" -t 0.05 -e 60
```
```
For 0.05 cm of 'Cadmium Telluride' the transmission of photons, with energy 60.0 keV, is around 13.17 %
```

For visualizing the transmission of a 100 kV tungsten X-ray spectrum through 1 mm Al and 1 mm CdTe:

```bash
xray-transmission -f -m Al "Cadmium Telluride" -t 0.1 -e 100
```

Filters **stack**, so this plots three curves: the bare 80 kV spectrum, the spectrum after 1 mm Al, and the spectrum after 1 mm Al *plus* 5 mm CdTe. The legend marks the accumulation with a `+` prefix.

Adding the option: `-s` or `--save-plot` will save the displayed figure as a .png file in the /tmp/ folder (linux).

![100 kV tungsten spectrum filtered by 1 mm Al and 1 mm CdTe](docs/images/example_spectrum.png)


#### Options

| Option | Meaning |
| --- | --- |
| `-m`, `--material_name` | Element symbol (`Al`), element name (`Aluminum`) or compound name. Accepts several values in full-spectrum mode. Use `-` to browse the material list interactively |
| `-t`, `--thickness` | Thickness in cm. Accepts several values in full-spectrum mode |
| `-e`, `--energy` | Photon energy in keV (3 - 200). In full-spectrum mode this is the tube voltage in kV (9 - 100) instead |
| `-f`, `--full-spectrum` | Filter a whole tungsten spectrum instead of a single energy, and plot the result |
| `-s`, `--save-plot` | Also write the plot as a .png. Full-spectrum mode only |

In single-value mode any argument you leave out is prompted for. Full-spectrum mode only prompts for the energy: if `-m` or `-t` is missing it warns and plots the unfiltered spectrum.

With several materials and thicknesses, equal-length lists are paired up one filter each (`-m Al Cu -t 0.1 0.2` gives Al 1 mm followed by Cu 2 mm). Otherwise every material is combined with every thickness.

## Other information
### List of materials

**Elements.** All 92, from Z = 1 (Hydrogen) to Z = 92 (Uranium). Either the symbol (`Al`) or the full name (`Aluminum`) works, and capitalisation does not matter.

**Compounds.** 17 in total:

| Material | Density [g/cm³] |
| --- | --- |
| Adipose Tissue (ICRU-44) | 0.95 |
| Air, Dry (near sea level) | 0.001205 |
| Blood, Whole (ICRU-44) | 1.06 |
| Bone, Cortical (ICRU-44) | 1.92 |
| Brain, Grey/White Matter (ICRU-44) | 1.04 |
| Breast Tissue (ICRU-44) | 1.02 |
| Cadmium Telluride | 6.2 |
| Gallium Arsenide | 5.31 |
| Glass, Borosilicate (Pyrex) | 2.23 |
| Muscle, Skeletal (ICRU-44) | 1.05 |
| Polyethylene Terephthalate, (Mylar) | 1.38 |
| Polymethyl Methacrylate, (PMMA) | 1.19 |
| Polytetrafluoroethylene, (Teflon) | 2.25 |
| Polyvinyl Chloride, (PVC) | 1.406 |
| Tissue, Soft (ICRU-44) | 1.06 |
| Water, Liquid | 1.0 |
| PLA, (XCOM) | 1.108 |

Compound names contain spaces and commas, so quote them on the command line. Running `xray-transmission -m -` lists every element and compound with an index you can select instead of typing the name.

### NIST Tables
The provided files in data/ are not kept up to date with the NIST database in [1], and they are derived from the original source. The units of the linear attenuation coefficient are in $1/cm$.

That last point is worth stressing: the tabulated values are $\mu$ **already multiplied by the density**, not the mass attenuation coefficient $\mu/\rho$ that NIST publishes. They go straight into the Beer-Lambert exponent with a thickness in cm — no density lookup needed. The densities listed above are given for reference only.

The elements are split across three files by atomic number, `1-19.dat`, `20-69.dat` and `70-92.dat`, which are joined on their common energy column at load time; the compounds live in `compounds.dat`. All four are tab-separated, and all four share the same energy grid: 3 keV to 200 keV in 0.1 keV steps. A lookup has to land exactly on that grid — energies in between currently return no result, and log-log interpolation is the planned replacement.

### Tungsten spectra
The provided spectra are partially simulated and partially linear interpolated between the simulated values.
9 kV, 10 kV, 15 kV, 20 kV, 25 kV, 30 kV, 40 kV, 50 kV, 60 kV, 70 kV, 80 kV, 90 kV, and 100 kV, spectra correspond to a simulated Geant4 X-ray source with tungsten anode based on the HAMMAMATSU L10101.
In between spectra was linear interpolated.

Those 13 simulated voltages are the `data_*.txt` files; the `Ip*.txt` files hold the interpolations that fill in every remaining integer kV. `organize_spectra.py` merges the lot into `Spectra_9-100.csv`, a 1000 × 93 grid: one energy column running 0.1 keV to 100 keV in 0.1 keV steps, plus one column per tube voltage from 9 kV to 100 kV.

Spectra are not all the same length, so shorter ones are padded with `1e-35` to mark "no data". The same value is used as a floor when filtering: anything that falls below it is set to zero so it disappears cleanly from the logarithmic plots.

Note that filtering a spectrum trims it to 3 keV, since the attenuation tables do not go any lower. Nothing of consequence is lost — the tungsten spectrum is negligible down there.

## References
[1] Hubbell, J.H. and Seltzer, S.M. (2004), Tables of X-Ray Mass Attenuation Coefficients and Mass Energy-Absorption Coefficients (version 1.4). [Online] Available: http://physics.nist.gov/xaamdi [2024, 03 28]. National Institute of Standards and Technology, Gaithersburg, MD.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2024 Juan Sebastián Useche.
