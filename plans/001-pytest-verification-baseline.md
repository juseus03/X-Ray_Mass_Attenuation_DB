# Plan 001: Establish a pytest verification baseline

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat b8a2fff..HEAD -- data/ pyproject.toml requirements.txt`
> If any of those changed since this plan was written, compare the "Current
> state" facts below against the live data before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `b8a2fff`, 2026-07-29

## Why this matters

The repository has **zero tests and no CI** — there is no one-command way to
know the code and data are correct. Every other planned change (especially the
stateless-simulation refactor in plan 003) is a risky edit with nothing to
catch a regression. This plan installs `pytest`, adds a `tests/` package, and
writes tests that **pass on the current code** — pinning the NIST data files and
the Beer–Lambert relationship. It is the green baseline that makes plans 002 and
003 safe. It deliberately does **not** test the buggy stacked-filter behavior;
that regression test belongs in plan 003, which fixes the bug.

## Current state

- No `tests/` directory exists; `pytest` is not installed in the environment.
- The Python environment lives at `./envs` (a conda env, gitignored). It already
  contains `numpy`, `polars` (0.20.31), `streamlit`, `plotly`. It does **not**
  contain `pytest` yet.
- Application code lives under `code/` (not an installed package). Modules are
  imported by bare name (e.g. `from XrayTransmission import Simulation`) and
  read data via **repo-root-relative** paths like `./data/1-19.dat`, so tests
  must run with the repo root as the working directory and `code/` on
  `sys.path`. A `conftest.py` will arrange both.
- Verified data facts (these become test assertions — confirmed by reading the
  files at commit `b8a2fff`):
  - `data/W-Spectra/Spectra_9-100.csv` → shape `(1000, 93)`; first column
    `"Energy[keV]"`, last column `"100"`.
  - Elements: `data/1-19.dat`, `data/20-69.dat`, `data/70-92.dat` are
    tab-separated, joined on `"Energy"` → 93 columns (`Energy` + 92 elements);
    `Energy` is sorted ascending, min `3.0`, max `200.0`; **no nulls**; minimum
    value across all element columns is `2.88e-05` (all positive).
  - `data/compounds.dat` (tab-separated) → 18 columns (`Energy` + 17 compounds),
    no nulls, all values positive.
  - `data/names_elements.txt` (tab-separated) has columns
    `['Z','Symbol','Element','Z/A','I[eV]','Density[g/cm3]']`, 92 rows; **every**
    value in the `Element` column is also a column name in the joined element
    table.
  - `data/names_compounds.txt` (tab-separated) has columns
    `['Material','<Z/A>','I [eV]','Density [g/cm3]']`, 17 rows; every value in
    `Material` is a column in `data/compounds.dat`.
  - `.dat` values are the **linear** attenuation coefficient μ in cm⁻¹ (already
    multiplied by density), so transmission is `exp(-μ·t)` with `t` in cm and no
    density factor. Pinned case: Aluminum at `Energy == 30` keV has
    μ = `3.075901785` cm⁻¹, giving `exp(-μ·0.1) = 0.735217` (6 dp).

## Commands you will need

| Purpose      | Command                                    | Expected on success            |
|--------------|--------------------------------------------|--------------------------------|
| Activate env | `conda activate envs` (from repo root)     | prompt shows the env           |
| Deps present | `python -c "import polars, numpy"`         | exit 0, no output              |
| Install test dep | `python -m pip install pytest`         | installs pytest, exit 0        |
| Run tests    | `python -m pytest -q`                      | all pass                       |

> The env at `./envs` is gitignored. If you are running in a fresh git worktree
> where `./envs` is absent, activate the original environment by its absolute
> path (`conda activate /mnt/sda3/Sebastian/FMF/PhD/X-Ray_Mass_Attenuation_DB/envs`)
> or otherwise ensure `python -c "import polars, numpy"` exits 0. If it does not,
> STOP and report.

## Scope

**In scope** (the only files you should create/modify):
- `tests/__init__.py` (create, empty)
- `tests/conftest.py` (create)
- `tests/test_data_integrity.py` (create)
- `tests/test_beer_lambert.py` (create)
- `pyproject.toml` (add pytest as a dev dependency + pytest config)
- `requirements.txt` (add `pytest`)

**Out of scope** (do NOT touch):
- Any file under `code/` — no source changes in this plan.
- Any file under `data/` — data is read-only reference; never edit.
- `environment.yml` — left for the packaging plan.

## Git workflow

- Branch: `advisor/001-pytest-baseline`
- One commit for the whole plan is fine; message style matches the repo's plain
  imperative log (e.g. `Add pytest baseline and data-integrity tests`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Install pytest and record it

Install into the active env:

```
python -m pip install pytest
```

Then add pytest to the dependency specs. In `pyproject.toml`, under
`[tool.poetry.dependencies]`, the block currently ends at:

```toml
plotly = "^6.3.1"
pillow = ">=12.3.0"
```

Add a dev-dependency group after the `[tool.poetry.dependencies]` table (do not
put pytest in the runtime dependencies):

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
```

And append the pytest configuration at the end of the file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

In `requirements.txt`, add a line `pytest` (the file currently ends with
`pillow>=12.3.0` and has no trailing newline — add a newline then `pytest`).

**Verify**: `python -c "import pytest; print(pytest.__version__)"` → prints a
version ≥ 8, exit 0.

### Step 2: Create the tests package and conftest

Create `tests/__init__.py` as an empty file.

Create `tests/conftest.py` so tests can import the `code/` modules and resolve
the repo-root-relative data paths regardless of where pytest is invoked:

```python
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The application modules live in code/ and are imported by bare name.
sys.path.insert(0, str(REPO_ROOT / "code"))

# XrayTransmission and the .dat loaders use repo-root-relative paths
# ("./data/..."), so tests must run with the repo root as CWD.
os.chdir(REPO_ROOT)
```

**Verify**: `python -m pytest -q` → collects 0 tests, exit 0 (no errors).

### Step 3: Data-integrity tests

Create `tests/test_data_integrity.py`. These read the data files directly with
polars — they do **not** import application code, so they are robust:

```python
import polars as pl


def _elements_table() -> pl.DataFrame:
    e0 = pl.read_csv("data/1-19.dat", separator="\t")
    e1 = pl.read_csv("data/20-69.dat", separator="\t")
    e2 = pl.read_csv("data/70-92.dat", separator="\t")
    return e0.join(e1, on="Energy", how="left", coalesce=True).join(
        e2, on="Energy", how="left", coalesce=True
    )


def test_spectra_csv_shape_and_columns():
    sp = pl.read_csv("data/W-Spectra/Spectra_9-100.csv")
    assert sp.shape == (1000, 93)
    assert sp.columns[0] == "Energy[keV]"
    assert sp.columns[-1] == "100"


def test_element_energy_grid_sorted_and_ranged():
    df = _elements_table()
    assert df.shape[1] == 93
    assert df["Energy"].is_sorted()
    assert df["Energy"].min() == 3.0
    assert df["Energy"].max() == 200.0


def test_no_nulls_or_negatives_in_element_table():
    df = _elements_table()
    assert df.null_count().sum_horizontal().item() == 0
    assert min(df[c].min() for c in df.columns) >= 0.0


def test_every_element_name_has_a_column():
    df = _elements_table()
    names = pl.read_csv("data/names_elements.txt", separator="\t")
    missing = [n for n in names["Element"].to_list() if n not in df.columns]
    assert missing == []


def test_every_compound_name_has_a_column():
    comp = pl.read_csv("data/compounds.dat", separator="\t")
    names = pl.read_csv("data/names_compounds.txt", separator="\t")
    assert comp.null_count().sum_horizontal().item() == 0
    missing = [n for n in names["Material"].to_list() if n not in comp.columns]
    assert missing == []
```

**Verify**: `python -m pytest -q tests/test_data_integrity.py` → 5 passed.

### Step 4: Beer–Lambert pinning test

Create `tests/test_beer_lambert.py`. This pins both the Aluminum μ value in the
data and the transmission formula. It reads μ from the file rather than
hardcoding it, then asserts the resulting transmission:

```python
import math

import polars as pl


def test_aluminum_transmission_at_30kev():
    e0 = pl.read_csv("data/1-19.dat", separator="\t")
    mu = e0.filter(pl.col("Energy") == 30.0)["Aluminum"][0]
    # .dat values are linear attenuation coefficients (cm^-1), density-included.
    assert mu == 3.075901785
    transmission = math.exp(-mu * 0.1)  # 0.1 cm of Al
    assert math.isclose(transmission, 0.735217, abs_tol=1e-6)


def test_transmission_limits():
    mu = 3.075901785
    assert math.exp(-mu * 0.0) == 1.0            # zero thickness -> full transmission
    assert math.exp(-mu * 1e6) < 1e-9            # very thick -> ~0
```

**Verify**: `python -m pytest -q tests/test_beer_lambert.py` → 2 passed.

### Step 5: Full run

**Verify**: `python -m pytest -q` → **7 passed**, exit 0.

## Test plan

This plan *is* the test plan — it creates the initial suite. New files:
`tests/test_data_integrity.py` (5 tests) and `tests/test_beer_lambert.py`
(2 tests). No existing test to model after (there are none); the structure above
is the pattern later plans should copy.

Verification: `python -m pytest -q` → 7 passed.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `python -c "import pytest"` exits 0
- [ ] `python -m pytest -q` exits 0 with **7 passed**
- [ ] `tests/conftest.py`, `tests/test_data_integrity.py`,
      `tests/test_beer_lambert.py` all exist
- [ ] `git status --porcelain` shows changes only to the in-scope files
- [ ] `plans/README.md` status row for 001 updated to DONE

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows a data file changed and the assertions above no longer
  match (e.g. spectra shape ≠ (1000, 93), or Aluminum@30keV μ ≠ `3.075901785`).
- `python -c "import polars, numpy"` fails (environment not available) — do not
  attempt to build a new environment.
- Any "should pass on current code" test fails — that means a data fact here is
  wrong; report the actual value rather than editing the data to fit the test.

## Maintenance notes

- These tests assume the current data files. If the NIST tables are ever
  regenerated, the pinned Aluminum value and shapes must be updated deliberately.
- `conftest.py` does `os.chdir(REPO_ROOT)`; this is a workaround for the
  repo-root-relative data paths in `code/XrayTransmission.py`. When the codebase
  is eventually packaged (deferred; see `CLAUDE_SUGGESTIONS.md` §2), that chdir
  can be removed.
- Reviewer should confirm no test reaches the network and none writes to `data/`.
