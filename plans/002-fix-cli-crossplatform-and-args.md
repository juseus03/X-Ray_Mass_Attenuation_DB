# Plan 002: Fix the CLI — cross-platform paths, honor positional args, off-grid energy

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat b8a2fff..HEAD -- code/calculate_mass_attenuation.py`
> If that file changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it as a
> STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001 (uses the pytest harness it creates)
- **Category**: bug
- **Planned at**: commit `b8a2fff`, 2026-07-29

## Why this matters

`code/calculate_mass_attenuation.py` is **completely broken on Linux/macOS** and
**silently ignores its own documented arguments**:

1. It builds data paths with Windows backslashes and fragile string slicing:
   `os.path.join(WORK_PATH[:-4], "data\\1-19.dat")`. On Linux this produces a
   literal file named `data\1-19.dat` and crashes with
   `FileNotFoundError: .../data\names_elements.txt` (verified by running it).
2. `main()` never reads `args.thickness`/`args.energy`; it always calls
   `get_user_input(test=False)` and prompts interactively. The usage documented
   in `CLAUDE.md` — `python code/calculate_mass_attenuation.py Al 0.1 32` —
   accepts those values and throws them away.
3. An energy not exactly on the data grid yields an empty polars result and
   raises an uncaught `IndexError`, instead of a clean "not in database" message.

After this plan the CLI runs on any OS, honors `material thickness energy` as
positional args (falling back to prompts when omitted), and fails gracefully on
off-grid energies.

## Current state

`code/calculate_mass_attenuation.py` — the CLI. Relevant excerpts as they exist
today:

```python
# lines 13-32
WORK_PATH = os.path.dirname(__file__)

def load_data_elements() -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_elements_0 = pl.scan_csv(os.path.join(WORK_PATH[:-4], "data\\1-19.dat"), separator="\t")
    df_elements_1 = pl.scan_csv(os.path.join(WORK_PATH[:-4],"data\\20-69.dat"), separator="\t")
    df_elements_2 = pl.scan_csv(os.path.join(WORK_PATH[:-4],"data\\70-92.dat"), separator="\t")
    df_elements = df_elements_0.join(df_elements_1, on="Energy", how="left")
    df_elements = df_elements.join(df_elements_2, on="Energy", how="left")
    df_elements_names = pl.read_csv(os.path.join(WORK_PATH[:-4],"data\\names_elements.txt"), separator="\t")
    return df_elements, df_elements_names

def load_data_compounds() -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_compounds = pl.scan_csv(os.path.join(WORK_PATH[:-4],"data\\compounds.dat"), separator="\t")
    df_compounds_names = pl.read_csv(os.path.join(WORK_PATH[:-4],"data\\names_compounds.txt"), separator="\t")
    return df_compounds, df_compounds_names
```

```python
# lines 35-48 — raises IndexError when the energy is off-grid (empty result [0])
def get_mass_attenuation(df_elements, element_name, energy) -> Tuple[float, bool]:
    try:
        mu = (
            df_elements.select(pl.col("Energy"), pl.col(element_name))
            .filter(pl.col("Energy") == energy)
            .collect()
        )[element_name][0]
    except pl.exceptions.ColumnNotFoundError:
        return None, False
    return mu, True
```

```python
# lines 51-70 — get_user_input has a test/injection path already
def get_user_input(test=False, thickness=0.1, energy=32) -> Tuple[str, float, float]:
    if test:
        return thickness, energy
    sys.stderr.write("--- Material thickness [cm]: ")
    thickness = float(input(""))
    sys.stderr.write("--- Photon energy [keV]: ")
    energy = float(input(""))
    energy = np.round(energy, 1)
    if energy < 3 or energy > 200:
        print("Error: Energy not in the database (3 keV - 200 keV)")
        exit(0)
    return thickness, energy
```

```python
# line 162 in main() — args.thickness / args.energy are parsed but never used
    thickness, energy = get_user_input(test=False)
```

The argparse setup (lines 73-99) already defines `thickness` (default `0`) and
`energy` (default `None`) as optional positionals.

Repo conventions to match:
- The library module `code/XrayTransmission.py` uses `logging`-free `print`/
  `icecream`; this plan should introduce the **standard `logging`** module for
  its one debug line (replacing the `icecream` import in *this file only*), which
  matches the direction set in `CLAUDE_SUGGESTIONS.md` §1.5. Use
  `logger = logging.getLogger(__name__)` and `logger.debug(...)`; do not
  configure global logging in a library-style module.
- Data files are tab-separated; element/compound tables join on `"Energy"`.
- pytest lives under `tests/` with a `conftest.py` that puts `code/` on
  `sys.path` and chdirs to the repo root (created in plan 001).

## Commands you will need

| Purpose      | Command                                                  | Expected on success        |
|--------------|----------------------------------------------------------|----------------------------|
| Deps present | `python -c "import polars, numpy"`                       | exit 0                     |
| Run CLI      | `python code/calculate_mass_attenuation.py Al 0.1 30`    | prints a transmission line |
| Run tests    | `python -m pytest -q tests/test_cli.py`                  | all pass                   |
| Full suite   | `python -m pytest -q`                                    | all pass                   |

> Same environment note as plan 001: ensure `python -c "import polars, numpy"`
> exits 0 (activate `./envs` or the env by absolute path). If not, STOP.

## Scope

**In scope** (the only files you should modify/create):
- `code/calculate_mass_attenuation.py`
- `tests/test_cli.py` (create)

**Out of scope** (do NOT touch):
- `code/XrayTransmission.py` — handled by plan 003. Do not change the app's data
  loading here even though it has the same relative-path smell.
- `code/app.py`, `code/organize_spectra.py`.
- The data-loading *column layout* — keep the same join-on-`Energy` structure;
  only fix the file paths.

## Git workflow

- Branch: `advisor/002-fix-cli`
- One commit is fine; message e.g. `Fix CLI paths, honor positional args, handle off-grid energy`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Replace Windows paths with a pathlib DATA_DIR

At the top of the file, replace the `WORK_PATH` line with a robust data
directory derived from the file location:

```python
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
```

Rewrite `load_data_elements` and `load_data_compounds` to build paths with
`DATA_DIR / "<file>"` (forward-slash-safe, OS-independent), e.g.:

```python
def load_data_elements() -> Tuple[pl.LazyFrame, pl.DataFrame]:
    df_elements_0 = pl.scan_csv(DATA_DIR / "1-19.dat", separator="\t")
    df_elements_1 = pl.scan_csv(DATA_DIR / "20-69.dat", separator="\t")
    df_elements_2 = pl.scan_csv(DATA_DIR / "70-92.dat", separator="\t")
    df_elements = df_elements_0.join(df_elements_1, on="Energy", how="left")
    df_elements = df_elements.join(df_elements_2, on="Energy", how="left")
    df_elements_names = pl.read_csv(DATA_DIR / "names_elements.txt", separator="\t")
    return df_elements, df_elements_names
```

Apply the same `DATA_DIR / "..."` change to `load_data_compounds`. Remove the now
unused `WORK_PATH`. Keep `import os` only if still used elsewhere (it is not after
this change — remove the `import os` line if nothing references `os`).

**Verify**: `python code/calculate_mass_attenuation.py Al 0.1 30` no longer
raises `FileNotFoundError`. (It will still prompt until Step 3 — feed it input
or Ctrl-C; the point is the path crash is gone. Confirm with:)
`python -c "import sys; sys.path.insert(0,'code'); import calculate_mass_attenuation as m; d,_=m.load_data_elements(); print(d.collect().shape)"`
→ prints `(1971, 93)` (1971 energy rows, 93 columns), exit 0.

### Step 2: Handle off-grid energy without crashing

In `get_mass_attenuation`, an energy not present in the grid makes the filtered
frame empty, so `[element_name][0]` raises `IndexError`. Guard it:

```python
def get_mass_attenuation(df_elements, element_name, energy) -> Tuple[float, bool]:
    try:
        result = (
            df_elements.select(pl.col("Energy"), pl.col(element_name))
            .filter(pl.col("Energy") == energy)
            .collect()
        )
    except pl.exceptions.ColumnNotFoundError:
        return None, False
    if result.height == 0:
        return None, False
    return result[element_name][0], True
```

**Verify**: covered by tests in Step 5 (`test_off_grid_energy_returns_not_found`).

### Step 3: Honor positional arguments in main()

In `main()`, after `material_name` is resolved, replace the unconditional
`thickness, energy = get_user_input(test=False)` (line 162) with logic that uses
the parsed positionals when the user supplied them and otherwise prompts. The
argparse defaults are `thickness=0` and `energy=None`.

```python
    if args.energy is not None:
        # non-interactive: both values come from the command line
        thickness = float(args.thickness)
        energy = float(np.round(float(args.energy), 1))
        if energy < 3 or energy > 200:
            print("Error: Energy not in the database (3 keV - 200 keV)")
            return
    else:
        thickness, energy = get_user_input(test=False)
```

Also change the two `exit(0)` calls that live *inside* `main()`'s success/lookup
paths (the "Symbol not in the data base", "column not in data" branches) to
`return` so the function is testable without killing the interpreter. Leave the
energy-range `exit(0)` inside `get_user_input` as-is (that path is only reached
interactively).

**Verify**:
`python code/calculate_mass_attenuation.py Al 0.1 30`
→ prints a line like
`For 0.1 cm of 'Aluminum' the transmission of photons, with energy 30.0 keV, is around 73.52 %`
and exits 0 **without prompting**.

### Step 4: Swap icecream for logging in this file

Replace the two icecream imports (lines 8-9) and `ic.disable()` (line 11) with:

```python
import logging

logger = logging.getLogger(__name__)
```

Replace each `ic(...)` call in this file (there are calls at lines 155 and 163)
with `logger.debug(...)`, e.g. `logger.debug("material_name=%s", material_name)`.
Do not add a `logging.basicConfig(...)` call (keep default WARNING level so
nothing prints in normal use).

**Verify**: `grep -n "icecream\|ic(" code/calculate_mass_attenuation.py` → no
matches. `python code/calculate_mass_attenuation.py Al 0.1 30` still prints the
transmission line and no debug output.

### Step 5: CLI tests

Create `tests/test_cli.py`. Import the module (conftest from plan 001 already
adds `code/` to `sys.path`) and drive `main()` via `subprocess` for the
end-to-end path plus unit-test the pure functions:

```python
import subprocess
import sys
from pathlib import Path

import calculate_mass_attenuation as cli

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "code" / "calculate_mass_attenuation.py"


def test_paths_load_element_table():
    df, names = cli.load_data_elements()
    assert df.collect().shape == (1971, 93)
    assert "Aluminum" in df.collect().columns


def test_get_mass_attenuation_known_value():
    df, _ = cli.load_data_elements()
    mu, ok = cli.get_mass_attenuation(df, "Aluminum", 30.0)
    assert ok is True
    assert mu == 3.075901785


def test_off_grid_energy_returns_not_found():
    df, _ = cli.load_data_elements()
    mu, ok = cli.get_mass_attenuation(df, "Aluminum", 30.05)  # not on the grid
    assert ok is False
    assert mu is None


def test_cli_end_to_end_uses_positional_args():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "Al", "0.1", "30"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "73.52" in result.stdout
    assert "Aluminum" in result.stdout
```

**Verify**: `python -m pytest -q tests/test_cli.py` → 4 passed.

### Step 6: Full run

**Verify**: `python -m pytest -q` → all tests pass (7 from plan 001 + 4 here = 11).

## Test plan

- New file `tests/test_cli.py` with 4 tests: element-table loads with the right
  shape (proves the path fix), known-μ lookup, off-grid energy returns
  `(None, False)` (the regression for the `IndexError`), and an end-to-end
  subprocess run proving positional args are honored (the regression for the
  ignored-args bug).
- Structural pattern: follows the direct-polars style of
  `tests/test_data_integrity.py` from plan 001 for the unit tests, plus a
  `subprocess.run` for the CLI path.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `python code/calculate_mass_attenuation.py Al 0.1 30` exits 0, prints the
      transmission line, and does **not** prompt
- [ ] `grep -rn "\\\\\\\\" code/calculate_mass_attenuation.py` returns no matches
      (no backslash paths remain)
- [ ] `grep -n "icecream" code/calculate_mass_attenuation.py` returns no matches
- [ ] `python -m pytest -q` exits 0; `tests/test_cli.py` has 4 passing tests
- [ ] `git status --porcelain` shows changes only to the two in-scope files
- [ ] `plans/README.md` status row for 002 updated

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `code/calculate_mass_attenuation.py` already changed and
  the excerpts above don't match the live code.
- `load_data_elements().collect().shape` is not `(1971, 93)` — the data layout
  assumption is wrong; report the actual shape.
- The end-to-end CLI test can't reach a passing state after honoring the args
  (e.g. output format differs) — report the actual stdout rather than loosening
  the assertion to nothing.

## Maintenance notes

- The interactive prompt path (`get_user_input`) is preserved for the no-args
  case; if the CLI is later migrated to `typer`/`click` (see
  `CLAUDE_SUGGESTIONS.md` §2), fold both paths into the new framework.
- `code/XrayTransmission.py` still loads data via `./data` relative paths — that
  fragility is deliberately out of scope here and is addressed when the project
  is packaged. A reviewer should not expect this plan to touch it.
- Off-grid energies now return "not found"; the follow-up (log-log interpolation,
  `CLAUDE_SUGGESTIONS.md` §5.1) would make them work — deferred.
