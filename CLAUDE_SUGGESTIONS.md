# Suggestions for X-Ray Mass Attenuation DB

Goal: turn this into a portfolio-quality project for a scientific data analyst — something a
recruiter can install in one command, run in two, and that demonstrates correct physics,
solid software engineering, and good communication.

The suggestions are ordered by priority. Sections 1–3 are the "must do" core; sections 4–6
are what elevates it from "working script" to "tool others can use".

---

## 1. Fix correctness bugs first (highest priority)

A portfolio project for a *scientific* data analyst lives or dies on correctness. These are
actual bugs found by reading the code; each should get a regression test when fixed (see §3).

### 1.1 Stacked filters are applied twice in the Streamlit app
`Simulation.add_filter()` (code/XrayTransmission.py:127) recalculates filters `index..end` by
appending onto `spectrum_list[-1]`, which already contains those filters' attenuation.
Because `@st.cache_resource` keeps the same `Simulation` (and its `filter_list`) alive across
reruns, the per-rerun loop in code/app.py:145 does this with 2 filters A, B:

- `add_filter(A, 0)` → recalculates A *and* B → spectrum list ends at `A·B`
- `add_filter(B, 1)` → recalculates B again on top → final spectrum is `A·B·B`

Filter B's attenuation is applied twice; with 3 filters it compounds further. Fix: make
`calculate_transmited_spectrum` recompute from the base spectrum for the whole filter stack
(it's cheap — one vectorized `exp` per filter), instead of maintaining incremental state.

### 1.2 Shared mutable state across users
`@st.cache_resource` (code/app.py:13) shares one `Simulation` object across *all* browser
sessions. Since `Simulation` stores `filter_list`/`spectrum_list` as instance state, two
concurrent users would corrupt each other's results. Fix: make `Simulation` stateless
(pure functions: spectrum in → spectrum out) and keep per-user state only in
`st.session_state`. This also makes `remove_filter()` — which currently re-applies filters
on top of an already-filtered spectrum (code/XrayTransmission.py:145) — unnecessary.

### 1.3 CLI silently ignores its positional arguments
`calculate_mass_attenuation.py` defines `thickness` and `energy` as arguments
(code/calculate_mass_attenuation.py:85-97) but `main()` never reads them — it always calls
`get_user_input(test=False)` and prompts interactively. The documented usage
`python code/calculate_mass_attenuation.py Al 0.1 30` accepts the values and discards them.

### 1.4 Windows-only paths in the CLI
`os.path.join(WORK_PATH[:-4], "data\\1-19.dat")` (code/calculate_mass_attenuation.py:17)
breaks on Linux/macOS, and the `[:-4]` string slicing to strip `code` is fragile. Use
`pathlib`: `DATA_DIR = Path(__file__).resolve().parent.parent / "data"`. Same for the
`"./data/..."` relative paths in `XrayTransmission.py`, which only work when the process is
started from the repo root.

### 1.5 Debug output is on in the library
`ic.disable()` is commented out in code/XrayTransmission.py:6, so `ic(df)` prints a full
DataFrame on every filter calculation. Replace `icecream` with the standard `logging` module
(`logger.debug(...)`) — it's the professional pattern and removes a dependency.

### 1.6 Smaller issues
- `get_mass_attenuation` in the CLI returns `(None, False)` for unknown columns, but an
  energy that isn't exactly on the grid raises an uncaught `IndexError` (empty filter result,
  code/calculate_mass_attenuation.py:39-43).
- The exact-match energy lookup should become log-log interpolation (see §5.1).
- `class Filter` shadows the `filter` builtin; rename to `AttenuationFilter` and make it a
  `@dataclass`.
- Docstrings promise `List[np.ndarray, np.ndarray]` returns but the functions return
  `pl.DataFrame` (code/XrayTransmission.py:63, 90). Fix hints and docstrings together.
- Hardcoded log-scale y-range `[-10, -5]` (code/app.py:193) clips data instead of adapting
  to it.

---

## 2. Restructure into an installable package

Right now this is a `code/` folder of scripts that must be run from the repo root. The single
biggest portfolio upgrade is making it a real Python package:

```
xray_attenuation/
├── pyproject.toml            # single source of truth for deps + console scripts
├── src/xray_attenuation/
│   ├── __init__.py
│   ├── data.py               # all loading (currently duplicated between
│   │                         #   XrayTransmission.py and calculate_mass_attenuation.py)
│   ├── physics.py            # Beer-Lambert, interpolation, HVL — pure functions
│   ├── materials.py          # element/compound lookup, symbol resolution
│   ├── cli.py                # argparse/typer CLI
│   ├── app.py                # Streamlit UI (thin layer over physics.py)
│   └── data/                 # ship the NIST tables inside the package
├── scripts/organize_spectra.py   # one-off preprocessing, kept out of the package
└── tests/
```

Concrete steps:
- **Console entry points** in `pyproject.toml`: `xray-transmission = "xray_attenuation.cli:main"`,
  so after `pip install .` the tool is a real command. That's the "tools others can use" goal.
- **One dependency spec.** You currently have three that can drift (`pyproject.toml`,
  `requirements.txt`, `environment.yml`). Keep `pyproject.toml` (+ a lock file); delete the
  others or generate them. Consider `uv` — modern, fast, and a good signal.
- **Deduplicate loading logic.** `load_data_bases()` and `load_data_elements()`/
  `load_data_compounds()` are parallel implementations of the same thing. One `data.py`
  using `importlib.resources` to find the bundled `.dat` files fixes both the duplication
  and the path problems in one move.
- **Refactor `organize_spectra.py`** from a top-level script with magic indices (`i0 = 2`,
  reversed columns for `e0 in [15, 20, 25]`) into functions with a `main()`, docstrings
  explaining the column mapping, and a `--help`. It's preprocessing, so it belongs in
  `scripts/`, not the installed package.

---

## 3. Add tests and CI (the credibility layer)

There are currently zero tests. For a scientific tool this is the first thing a technical
reviewer checks.

- **Physics regression tests** (pytest):
  - Transmission through known cases, e.g. 1 mm Al at 30 keV, checked against a hand
    calculation from the NIST value — this pins both the data and the Beer-Lambert code.
  - Zero thickness → transmission = 1; thickness → ∞ → transmission → 0; monotonic in
    thickness.
  - Stacking two filters equals one filter of combined thickness (this test would have
    caught bug 1.1).
  - Order independence: filter A then B == B then A.
- **Data integrity tests**: every element/compound listed in `names_*.txt` has a matching
  column; energy grids are sorted; no negative μ values; spectra CSV has the expected
  1000 × 93 shape.
- **CLI tests** with `subprocess` or by calling `main()` with injected args.
- **GitHub Actions**: on every push run `ruff check`, `ruff format --check`, `pytest`
  (matrix: 3.11/3.12/3.13). Add the status badge to the README.
- **Pre-commit** with ruff — cheap and shows workflow maturity.
- Type-check with `mypy` or `pyright` once the type hints are fixed (§1.6).

---

## 4. Documentation and data provenance

The README is currently four lines. For a portfolio it's the landing page — most visitors
never read the code.

- **README.md** should include:
  - A screenshot/GIF of the Streamlit app (this alone doubles engagement).
  - One-paragraph physics background: Beer–Lambert law, `I = I₀·exp(−μ·t)`, what mass
    attenuation is, what the tool answers ("how much of a 40 kV tungsten spectrum survives
    2 mm of aluminum?").
  - Install (`pip install .`) and usage for both the app and the CLI, with example output.
  - A link to a **live deployment on Streamlit Community Cloud** — free, and turns the repo
    into a clickable demo.
- **Data provenance — critical for scientific credibility.** Document in the README and in a
  `data/README.md`:
  - Attenuation coefficients: NIST XCOM / Hubbell & Seltzer tables — cite them properly.
  - **Units.** The `.dat` files appear to hold μ/ρ already multiplied by density (linear
    attenuation in cm⁻¹) — this is non-obvious and currently documented nowhere except a
    docstring aside. State units for every file.
  - W-spectra: how were they simulated (which code, target angle, filtration?), what the
    `Ip*` interpolated files are, and what the 1e-35 sentinel means.
- **Docstrings** in NumPy style throughout `physics.py`/`data.py`; they become free API docs
  if you later add mkdocs-material + mkdocstrings.
- Optional but strong for a PhD portfolio: archive a release on **Zenodo** for a citable DOI,
  and add a `CITATION.cff`.

---

## 5. Scientific feature upgrades

These add analytical depth that distinguishes a data-analyst portfolio from a CRUD app:

1. **Log-log interpolation of μ(E)** instead of exact grid matching, so any energy in range
   works in the CLI and enables finer spectra. (Standard practice for attenuation data;
   handle absorption edges by interpolating between grid points, not across.)
2. **Derived quantities panel** in the app: total transmitted fraction, mean/effective
   energy of the filtered spectrum, first half-value layer (HVL) in mm Al — these are the
   numbers radiology physicists actually want, and each is ~5 lines once physics.py exists.
3. **Beam-hardening visualization**: plot the normalized spectra before/after filtering to
   show the spectrum shifting toward higher energies — a classic, compelling figure.
4. **Download buttons** (`st.download_button`) for the filtered spectrum and μ(E) curves as
   CSV — "tool others can use" includes getting data *out*.
5. **Mass-attenuation explorer tab**: plot μ/ρ vs E for selected materials on log-log axes
   (absorption edges are visually striking and show you know the physics).
6. **Thickness units** selector (µm / mm / cm) and optional density override for compounds.

---

## 6. Polish

- Remove the local `envs/` conda directory from the working tree (it's gitignored but
  shouldn't live inside the repo folder) and `code/__pycache__/`.
- `poetry.lock` is both gitignored and present on disk while `pyproject.toml` still declares
  poetry — resolve this when consolidating packaging (§2).
- Pin `polars` to a current 1.x release; `^0.20` is pre-1.0 API.
- Add a `CHANGELOG.md` and use tagged releases (`v1.0.0`) — small effort, professional look.
- Keep `CLAUDE.md` updated as the structure changes so it stays accurate.

---

## Suggested execution order

| Phase | Content | Outcome |
|-------|---------|---------|
| 1 | §1 bug fixes + first regression tests | Tool is *correct* |
| 2 | §2 package restructure + §3 CI | Tool is *installable and verified* |
| 3 | §4 README, screenshots, deployment, provenance | Project is *presentable* |
| 4 | §5 scientific features | Project is *impressive* |

Phases 1–2 are the foundation; don't add features (§5) before the double-filtering bug (§1.1)
is fixed and tested, or the new features will build on wrong numbers.
