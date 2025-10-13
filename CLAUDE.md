# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

X-Ray Mass Attenuation Database - Software for calculating X-ray transmission coefficients through NIST materials (elements and compounds). The project includes:
- A Streamlit web application for interactive visualization
- Command-line tools for transmission calculations
- Data processing scripts for organizing X-ray spectra

## Running the Application

**Web Interface (Streamlit):**
```bash
streamlit run code/app.py
```

**Command-line Interface:**
```bash
python code/calculate_mass_attenuation.py [material_name] [thickness] [energy]
```
- Use `-` or omit material_name to see available materials
- Symbol lookup: `python code/calculate_mass_attenuation.py Al` (2-letter element symbols)
- Interactive mode: Omit all arguments

## Environment Setup

The project uses conda for environment management:
```bash
conda env create -f environment.yml
conda activate envs
```

Python version: 3.12.2

## Data Architecture

### Data Files Structure

**Mass Attenuation Coefficients:**
- `data/1-19.dat` - Elements 1-19 (H to K)
- `data/20-69.dat` - Elements 20-69 (Ca to Tm)
- `data/70-92.dat` - Elements 70-92 (Yb to U)
- `data/compounds.dat` - NIST compound materials
- All files are tab-separated with Energy column and material columns

**Material Information:**
- `data/names_elements.txt` - Element metadata (Z, Symbol, Element name)
- `data/names_compounds.txt` - Compound names

**X-ray Spectra:**
- `data/W-Spectra/` - Tungsten target X-ray spectra (9-100 kV)
  - `data_*.txt` files: Simulated spectra (comma-separated, 3 columns: Energy, Data1, Data2)
  - `Ip*.txt` files: Interpolated spectra (tab-separated, 4 columns for intermediate energies)
  - `Spectra_9-100.csv` - Combined/organized spectra output (1000 rows × 93 columns)

### Core Classes

**`XrayTransmission.Simulation`** (code/XrayTransmission.py):
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

**`code/organize_spectra.py`:**
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

**`code/calculate_mass_attenuation.py`:**
- CLI tool with argparse interface
- Supports element symbol lookup (2 letters) or full names
- Interactive material selection by index or name
- Uses Windows-style paths (`\\` separators) - needs fixing for cross-platform

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

- All debug output uses `icecream` (ic) - disabled by default with `ic.disable()`
- Path handling inconsistent: Mix of forward slashes and `os.path.join()`
- `calculate_mass_attenuation.py` has hardcoded Windows paths - needs cross-platform fix
- Energy range: 3-200 keV (enforced in CLI tool)
- Material thickness in cm
