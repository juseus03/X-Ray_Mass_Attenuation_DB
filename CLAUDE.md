# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

X-Ray Mass Attenuation Database - Software for calculating X-ray transmission coefficients through NIST materials (elements and compounds). The project includes:
- A Streamlit web application for interactive visualization
- Command-line tools for transmission calculations
- Data processing scripts for organizing X-ray spectra

## Project Layout

The package is a PEP 621 / hatchling project using the `src/` layout:

```
pyproject.toml                 # single source of truth for deps + console script
src/xray_attenuation/          # the installed package
├── __init__.py                # exports Data, __version__
├── cli.py                     # argparse CLI (entry point: xray-transmission)
├── data.py                    # Data class: all loading + lookup
├── physics.py                 # (empty placeholder)
├── materials.py               # (empty placeholder)
└── data/                      # NIST tables, shipped inside the wheel
scripts/organize_spectra.py    # one-off preprocessing, NOT installed
tests/                         # pytest suite, NOT installed
code/                          # pre-restructure copy (Streamlit app); superseded
```

`code/` is the old, pre-restructure copy. It is excluded from ruff and is not part of the
package. Do not add features there.

## Running the Application

**Command-line Interface** (after `pip install -e .`):
```bash
xray-transmission -m Al -t 0.1 -e 30
```
- `-m/--material_name`: element symbol (`Al`), element name (`Aluminum`), or compound name;
  use `-` or omit to get an interactive material list
- `-t/--thickness`: thickness in cm
- `-e/--energy`: photon energy in keV (3-200)
- Any omitted argument is prompted for interactively

**Web Interface (Streamlit):** still the un-ported `code/app.py`
(`streamlit run code/app.py`). The GUI is being moved off Streamlit, so streamlit/plotly are
deliberately *not* declared in `pyproject.toml`.

## Environment Setup

`pyproject.toml` is the only dependency spec. `environment.yml` is a thin conda bootstrap that
provides the interpreter and then pip-installs the package:

```bash
conda env create -f environment.yml -p ./envs
./envs/bin/pip install -e ".[dev]"
```

Runtime deps: `polars>=1.0`, `numpy>=1.26`. Dev extra: `pytest`, `ruff`.
Requires Python >=3.11; the pinned dev interpreter is 3.12.2.

## Testing and Linting

```bash
pytest          # testpaths=tests, configured in pyproject.toml
ruff check .
```

Tests import the *installed* package (`from xray_attenuation.data import Data`). Because of
the `src/` layout there is deliberately no `__init__.py` in `tests/` and no path hackery —
`pip install -e .` is required before the suite will run.

## Data Architecture

### Data Files Structure

All data lives at `src/xray_attenuation/data/` and is bundled into the wheel. Resolve it with
`DATA_PATH = Path(__file__).parent / "data"` (see `data.py`) — never with a relative path or
a CWD assumption.

**Mass Attenuation Coefficients:**
- `1-19.dat` - Elements 1-19 (H to K)
- `20-69.dat` - Elements 20-69 (Ca to Tm)
- `70-92.dat` - Elements 70-92 (Yb to U)
- `compounds.dat` - NIST compound materials
- All files are tab-separated with Energy column and material columns
- Values are μ **already multiplied by density** (linear attenuation, cm⁻¹)

**Material Information:**
- `names_elements.txt` - Element metadata (Z, Symbol, Element name)
- `names_compounds.txt` - Compound names

**X-ray Spectra:**
- `W-Spectra/` - Tungsten target X-ray spectra (9-100 kV)
  - `data_*.txt` files: Simulated spectra (comma-separated, 3 columns: Energy, Data1, Data2)
  - `Ip*.txt` files: Interpolated spectra (tab-separated, 4 columns for intermediate energies)
  - `Spectra_9-100.csv` - Combined/organized spectra output (1000 rows × 93 columns)

### Core Classes

**`xray_attenuation.data.Data`** (src/xray_attenuation/data.py) — the current class:
- Loads elements (3 joined `.dat` files), compounds and spectra on construction
- Element/compound frames are lazy (`pl.scan_csv` → `LazyFrame`); spectra are eager
- `resolve_material_name(name)` - symbol/name → `(canonical_name, is_compound)` or `None`
- `get_linear_attenuation(material, energy, is_compound)` - μ [1/cm], or `None` if the exact
  energy is not on the grid (log-log interpolation is the planned replacement)

**`XrayTransmission.Simulation`** (code/XrayTransmission.py) — legacy, not in the package:
- Main simulation class that loads all databases
- `load_data_bases()` - Loads and joins element/compound data using Polars
- `get_spectrum(max_energy)` - Returns spectrum for given kV
- `get_mass_attenuation(material_name, is_element)` - Returns μ/ρ data
- `calculate_transmited_spectrum()` - Computes transmitted intensity using Beer-Lambert law

**Data Processing Pipeline:**
- Uses Polars DataFrames throughout (not Pandas)
- Element data joined on "Energy" column with left joins
- Spectra data has "Energy[keV]" column rounded to 1 decimal place
- Transmission calculated as: I = I₀ × exp(-μ × thickness)

## Code Organization

**`src/xray_attenuation/cli.py`:** (current CLI)
- argparse interface; `main()` is the `xray-transmission` console-script entry point
- Delegates all lookup to `Data`; prompts interactively for any missing argument
- `get_transmission(mu, thickness)` is pure Beer-Lambert and belongs in `physics.py`

**`scripts/organize_spectra.py`:** (not installed; still imports `icecream`/`matplotlib`)
- Processes raw W-Spectra files into unified CSV
- Column mapping: Second number in filename determines column placement
- Example: `data_10-15.txt` last column → Spectra_9-100.csv column 7
- Handles padding for different length spectra (pads with 1e-35)
- Special handling: Some Ip files have reversed column order (e0 in [15, 20, 25])

**`code/app.py`:**
- Streamlit interface with sidebar controls
- Caches Simulation object with `@st.cache_resource`
- Uses custom Plotly styling from `plotly_style.py`
- Supports log/linear Y-axis switching
- Filter by element or compound with adjustable thickness

**`code/calculate_mass_attenuation.py`:** (legacy, superseded by `cli.py`)
- Uses Windows-style paths (`\\` separators) and ignores its positional arguments

## Key Implementation Details

**Data Loading Pattern:**
Element data is split across 3 files and joined:
```python
df = df_elements_0.join(df_elements_1, on="Energy", how="left")
df = df.join(df_elements_2, on="Energy", how="left")
```

**Spectrum Processing:**
- Energy values: 0.05 to 99.95 keV in 0.1 keV steps (1000 points)
- Values below 1e-35 are sentinel values (no data)
- Transmission values < 1e-35 are set to 0 for plotting

**Column Indexing in organize_spectra.py:**
- `i0` tracks position for `data_*.txt` files
- `i1` tracks position for `Ip*.txt` files
- Update indices after each file: `i0 += e1 - e0` or `i1 = idx1`
- Interpolated files fill gaps between simulated energies

## Important Notes

- Energy range: 3-200 keV (enforced in `cli.py`); material thickness in cm
- `physics.py` and `materials.py` exist but are empty placeholders
- `get_linear_attenuation` requires an *exact* grid energy; off-grid values return `None`
- Legacy-only issues, confined to `code/` and `scripts/`: `icecream` debug output, hardcoded
  Windows paths, mixed path handling. The installed package uses none of these.
