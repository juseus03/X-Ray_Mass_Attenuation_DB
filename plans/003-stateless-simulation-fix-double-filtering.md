# Plan 003: Make `Simulation` stateless — fix double-filtering and cross-user state corruption

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat b8a2fff..HEAD -- code/XrayTransmission.py code/app.py`
> If either file changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch, treat
> it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/001 (pytest harness). Recommended after plans/002.
- **Category**: bug
- **Planned at**: commit `b8a2fff`, 2026-07-29

## Why this matters

The Streamlit app **produces physically wrong transmission numbers**. Two coupled
defects in `code/XrayTransmission.py`:

1. **Double-filtering.** `Simulation` maintains incremental state
   (`spectrum_list`, `filter_list`). `calculate_transmited_spectrum` applies a
   filter on top of `spectrum_list[-1]` (already-filtered). Because the app's
   `Simulation` is cached with `@st.cache_resource`, `filter_list` survives
   across reruns while `set_base_spectrum` resets `spectrum_list`. On the second
   and later reruns of a 2-filter setup, `add_filter`'s `else` branch recomputes
   the *tail* of the stack on top of an already-filtered spectrum, so the last
   filter's attenuation is applied **twice** (three filters compounds worse).
2. **Shared mutable state across users.** The single cached `Simulation` instance
   is shared by every browser session; its `filter_list`/`spectrum_list` are
   instance state, so concurrent users corrupt each other's results.

The fix is one idea: make attenuation a **pure function** — always recomputed
from the base spectrum for the whole filter stack (`I = I₀·exp(-Σ μᵢ·tᵢ)`), with
no per-request state on `Simulation`. Per-user state lives only in
`st.session_state` (already the case in `app.py`). This is mathematically
order-independent and can never double-apply. It also deletes `remove_filter`,
which had the same re-apply-on-top bug.

While here we also fix the type-hint/docstring lies (functions documented as
returning `List[np.ndarray]` actually return `pl.DataFrame`), rename the
builtin-shadowing `Filter` class to a `@dataclass AttenuationFilter`, replace the
`icecream` debug dump with `logging`, and make the app's log-scale y-range adapt
to the data instead of the hardcoded `[-10, -5]` that clips it.

## Current state

`code/XrayTransmission.py` (154 lines). The stateful core to be replaced:

```python
# lines 9-13 — Filter shadows the builtin `filter`
class Filter:
    def __init__(self, pname, pthickness, p_is_element):
        self.name = pname
        self.thickness = pthickness
        self.is_element = p_is_element
```

```python
# line 6 — icecream is left enabled; ic(df) at line 124 dumps a full DataFrame
# ic.disable()
```

```python
# lines 96-125 — incremental attenuation reading spectrum_list[-1]
def calculate_transmited_spectrum(self, filter):
    spectrum = self.spectrum_list[-1]
    mu = self.get_mass_attenuation(filter.name, filter.is_element)
    ...
    df = spectrum.join(mu, left_on="Energy[keV]", right_on="Energy", validate="1:1")
    df = df.with_columns(
        (pl.col(self.base_spectrum_energy) * np.exp(-1 * pl.col(filter.name) * filter.thickness))
        .alias(self.base_spectrum_energy)
    ).with_columns(
        pl.when(pl.col(self.base_spectrum_energy) < 1e-35).then(0)
        .otherwise(pl.col(self.base_spectrum_energy)).alias(self.base_spectrum_energy)
    ).drop(filter.name)
    ic(df)
    self.spectrum_list.append(df)

# lines 127-150 — add_filter / remove_filter maintain the stateful lists
```

The stateful methods `set_base_spectrum`, `get_base_spectrum`,
`get_current_spectrum`, `calculate_transmited_spectrum`, `add_filter`,
`remove_filter` and the instance attributes `spectrum_list`, `filter_list`,
`base_spectrum_energy` are all part of the broken incremental design and are
replaced below. **Only `app.py` uses these methods** (verified: the CLI has its
own loaders; `organize_spectra.py` is unrelated).

`code/app.py` uses them at:
- line 2: `from XrayTransmission import Simulation`
- lines 132-137: removal loop calls `sim.remove_filter(i)`
- lines 143-148: `sim.set_base_spectrum(...)` then `sim.add_filter(...)` loop
- line 159: `spectrum = sim.get_base_spectrum()` (no-arg)
- line 170: `spectrum2 = sim.get_current_spectrum()`
- line 193: `yaxis_range=[-10, -5] if is_log_y else None`

Verified physics facts (from prototyping against the real data at `b8a2fff`):
- Base spectrum has 1000 energy rows (`Energy[keV]` 0.1 … 100.0 after `round(1)`).
- The mass-attenuation grid (`Energy` 3 … 200) has **no duplicate energies**, so
  an **inner** join with `validate="1:1"` succeeds and yields **971 rows** (the
  3.0 … 100.0 overlap). This matches the original code's default-inner join —
  keep inner so no NaN intensities appear below 3 keV.
- Stacking `[Al 0.1, Al 0.2]` equals `[Al 0.3]`; `[Al, Carbon]` equals
  `[Carbon, Al]`; empty filter list equals the base spectrum. (These become the
  regression tests.)

Conventions to match: the codebase uses Polars throughout and joins element/
compound tables on `Energy`; introduce `logging` (`logger = logging.getLogger(__name__)`,
`logger.debug(...)`) rather than `icecream`, consistent with the direction in
`CLAUDE_SUGGESTIONS.md` §1.5. Tests live under `tests/` with the `conftest.py`
from plan 001 (adds `code/` to `sys.path`, chdirs to repo root).

## Commands you will need

| Purpose        | Command                                          | Expected on success       |
|----------------|--------------------------------------------------|---------------------------|
| Deps present   | `python -c "import polars, numpy, streamlit"`    | exit 0                    |
| Compile app    | `python -m py_compile code/app.py`               | exit 0, no output         |
| Physics tests  | `python -m pytest -q tests/test_physics.py`      | all pass                  |
| Full suite     | `python -m pytest -q`                            | all pass                  |
| Manual smoke   | `streamlit run code/app.py` (reviewer, optional) | app loads, filters update |

> Same environment note as plan 001: ensure the imports above exit 0 (activate
> `./envs` or the env by absolute path). If not, STOP.

## Scope

**In scope** (the only files you should modify/create):
- `code/XrayTransmission.py`
- `code/app.py`
- `tests/test_physics.py` (create)

**Out of scope** (do NOT touch):
- `code/calculate_mass_attenuation.py` — plan 002; it has its own loaders.
- `code/organize_spectra.py` — its `icecream` usage is removed in plan 004.
- The data files and the widget/session-state logic in `app.py` above line 132
  (the sidebar filter widgets) — leave that exactly as-is; only the data-flow
  and plotting sections change.
- Dependency manifests — `icecream` is dropped from them in plan 004, not here.

## Git workflow

- Branch: `advisor/003-stateless-simulation`
- Commit per logical unit is fine (engine, then app, then tests); message e.g.
  `Make Simulation stateless to fix double-filtering`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Rewrite `code/XrayTransmission.py` as a stateless engine

Replace the **entire file** with the following. It keeps the read-only database
loading (safe to cache/share), adds one pure `transmitted_spectrum` method, and
removes all incremental state:

```python
import logging
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class AttenuationFilter:
    """A single attenuating layer: material name, thickness [cm], element flag."""

    name: str
    thickness: float
    is_element: bool


class Simulation:
    """Loads the NIST databases and computes transmitted X-ray spectra.

    The instance is read-only after construction and holds no per-request state,
    so a single object can be shared safely across concurrent sessions.
    """

    def __init__(self) -> None:
        self.fname_spectra = "./data/W-Spectra/Spectra_9-100.csv"
        self.fname_compounds = "./data/compounds.dat"
        self.fname_elements = [
            "./data/1-19.dat",
            "./data/20-69.dat",
            "./data/70-92.dat",
        ]
        self.fname_info_compounds = "./data/names_compounds.txt"
        self.fname_info_elements = "./data/names_elements.txt"

        self.load_data_bases()

    def load_data_bases(self) -> None:
        """Load the spectra, mass-attenuation and material-info tables."""
        self.spectra_db = (
            pl.scan_csv(self.fname_spectra)
            .with_columns(pl.col("Energy[keV]").round(1))
            .collect()
        )
        self.mu_compounds = pl.scan_csv(self.fname_compounds, separator="\t").collect()
        df0 = pl.scan_csv(self.fname_elements[0], separator="\t").collect()
        df1 = pl.scan_csv(self.fname_elements[1], separator="\t").collect()
        df2 = pl.scan_csv(self.fname_elements[2], separator="\t").collect()
        self.mu_elements = df0.join(
            df1, on="Energy", how="left", coalesce=True
        ).join(df2, on="Energy", how="left", coalesce=True)

        self.info_elements = pl.read_csv(self.fname_info_elements, separator="\t")
        self.info_compounds = pl.read_csv(self.fname_info_compounds, separator="\t")

    def get_base_spectrum(self, max_energy: str) -> pl.DataFrame:
        """Return the unfiltered spectrum for the given tube voltage.

        Returns a DataFrame with columns ``Energy[keV]`` and ``max_energy``
        (the intensity column).
        """
        return self.spectra_db.select(pl.col("Energy[keV]"), pl.col(max_energy))

    def get_mass_attenuation(
        self, material_name: str, is_element: bool
    ) -> pl.DataFrame:
        """Return linear attenuation coefficient (cm^-1) vs energy.

        Returns a DataFrame with columns ``Energy`` and ``material_name``.
        """
        table = self.mu_elements if is_element else self.mu_compounds
        return table.select(pl.col("Energy"), pl.col(material_name))

    def transmitted_spectrum(
        self, max_energy: str, filters: list[AttenuationFilter]
    ) -> pl.DataFrame:
        """Apply a stack of filters to the base spectrum (Beer-Lambert law).

        Pure function of its inputs: the result is always recomputed from the
        base spectrum, so filter order is irrelevant and no filter is ever
        applied twice. Returns a DataFrame with columns ``Energy[keV]`` and
        ``max_energy``.
        """
        df = self.get_base_spectrum(max_energy)
        intensity_col = max_energy
        for f in filters:
            mu = self.get_mass_attenuation(f.name, f.is_element)
            df = df.join(
                mu,
                left_on="Energy[keV]",
                right_on="Energy",
                how="inner",
                validate="1:1",
            )
            df = df.with_columns(
                (
                    pl.col(intensity_col) * (-pl.col(f.name) * f.thickness).exp()
                ).alias(intensity_col)
            ).drop(f.name)

        df = df.with_columns(
            pl.when(pl.col(intensity_col) < 1e-35)
            .then(0.0)
            .otherwise(pl.col(intensity_col))
            .alias(intensity_col)
        )
        logger.debug(
            "transmitted spectrum for %s kV with %d filter(s)",
            max_energy,
            len(filters),
        )
        return df
```

**Verify**:
`python -c "import sys; sys.path.insert(0,'code'); from XrayTransmission import Simulation, AttenuationFilter; s=Simulation(); print(s.transmitted_spectrum('30', [AttenuationFilter('Aluminum',0.1,True)]).shape)"`
→ prints `(971, 2)`, exit 0. (Run from the repo root.)

### Step 2: Update `code/app.py` to the stateless API

Make exactly these edits; leave everything else (the sidebar filter widgets)
untouched.

**Edit A — import** (line 2):
- from: `from XrayTransmission import Simulation`
- to:   `from XrayTransmission import Simulation, AttenuationFilter`

**Edit B — removal loop** (lines 132-137): drop the `sim.remove_filter(i)` call.
- from:
  ```python
  if filters_to_remove:
      for i in reversed(filters_to_remove):
          st.session_state.filters.pop(i)
          sim.remove_filter(i)
      st.rerun()
  ```
- to:
  ```python
  if filters_to_remove:
      for i in reversed(filters_to_remove):
          st.session_state.filters.pop(i)
      st.rerun()
  ```

**Edit C — replace the stateful apply block** (lines 143-148):
- from:
  ```python
  # Reinitialize spectrum and apply all filters to simulation
  sim.set_base_spectrum(spectrum_energy)
  for i, filter_data in enumerate(st.session_state.filters):
      sim.add_filter(
          filter_data["material"], filter_data["thickness"], filter_data["is_element"], i
      )
  ```
- to:
  ```python
  # Build the filter stack from per-session state (Simulation holds no state)
  active_filters = [
      AttenuationFilter(f["material"], f["thickness"], f["is_element"])
      for f in st.session_state.filters
  ]
  ```

**Edit D — open-beam data** (line 159): pass the energy argument.
- from: `spectrum = sim.get_base_spectrum()`
- to:   `spectrum = sim.get_base_spectrum(spectrum_energy)`

**Edit E — filtered data** (lines 169-172):
- from:
  ```python
  if len(st.session_state.filters) > 0:
      spectrum2 = sim.get_current_spectrum()
      energy2 = spectrum2["Energy[keV]"].to_numpy().reshape(-1)
      intensity2 = spectrum2[spectrum_energy].to_numpy().reshape(-1)
  ```
- to:
  ```python
  if active_filters:
      spectrum2 = sim.transmitted_spectrum(spectrum_energy, active_filters)
      energy2 = spectrum2["Energy[keV]"].to_numpy().reshape(-1)
      intensity2 = spectrum2[spectrum_energy].to_numpy().reshape(-1)
  ```

**Edit F — adaptive y-range.** Immediately *before* the `fig.update_layout(`
call (currently line 189), insert:
  ```python
  # Adaptive log y-range from the actual data (avoids clipping)
  yaxis_range = None
  if is_log_y:
      ys = intensity
      if active_filters:
          ys = np.concatenate([intensity, intensity2])
      positive = ys[ys > 1e-35]
      if positive.size:
          yaxis_range = [
              float(np.floor(np.log10(positive.min()))),
              float(np.ceil(np.log10(positive.max()))),
          ]
  ```
  Then change the range line inside `fig.update_layout(...)`:
- from: `    yaxis_range=[-10, -5] if is_log_y else None,`
- to:   `    yaxis_range=yaxis_range,`

(`np` is already imported at `app.py:4`.)

**Verify**: `python -m py_compile code/app.py` → exit 0, no output.

### Step 3: Regression tests

Create `tests/test_physics.py`:

```python
import numpy as np
import pytest

from XrayTransmission import AttenuationFilter, Simulation


@pytest.fixture(scope="module")
def sim():
    return Simulation()


def _al(t):
    return AttenuationFilter("Aluminum", t, True)


def test_empty_filters_equals_base(sim):
    base = sim.get_base_spectrum("30")["30"].to_numpy()
    out = sim.transmitted_spectrum("30", [])["30"].to_numpy()
    assert np.allclose(out, base)


def test_stacking_equals_combined_thickness(sim):
    stacked = sim.transmitted_spectrum("30", [_al(0.1), _al(0.2)])["30"].to_numpy()
    single = sim.transmitted_spectrum("30", [_al(0.3)])["30"].to_numpy()
    assert np.allclose(stacked, single)


def test_order_independence(sim):
    ab = sim.transmitted_spectrum(
        "30", [_al(0.1), AttenuationFilter("Carbon", 0.2, True)]
    )["30"].to_numpy()
    ba = sim.transmitted_spectrum(
        "30", [AttenuationFilter("Carbon", 0.2, True), _al(0.1)]
    )["30"].to_numpy()
    assert np.allclose(ab, ba)


def test_single_filter_not_double_applied(sim):
    # Direct regression for the double-filtering bug: one Al filter must equal
    # exactly one manual Beer-Lambert application, never two.
    result = sim.transmitted_spectrum("30", [_al(0.1)])["30"].to_numpy()
    base = sim.get_base_spectrum("30")
    mu = sim.get_mass_attenuation("Aluminum", True)
    joined = base.join(mu, left_on="Energy[keV]", right_on="Energy", how="inner")
    expected = joined["30"].to_numpy() * np.exp(-joined["Aluminum"].to_numpy() * 0.1)
    expected = np.where(expected < 1e-35, 0.0, expected)
    assert np.allclose(result, expected)


def test_thicker_transmits_less(sim):
    thin = sim.transmitted_spectrum("30", [_al(0.1)])["30"].to_numpy()
    thick = sim.transmitted_spectrum("30", [_al(0.5)])["30"].to_numpy()
    assert np.all(thick <= thin + 1e-30)
```

**Verify**: `python -m pytest -q tests/test_physics.py` → 5 passed.

### Step 4: Full run

**Verify**: `python -m pytest -q` → all tests pass (7 from 001, +5 here, +4 if
plan 002 already landed). Confirm no `icecream`/`ic(` remains in the engine:
`grep -n "icecream\|ic(" code/XrayTransmission.py` → no matches.

## Test plan

- New file `tests/test_physics.py`, 5 tests: empty==base, stacking==combined
  thickness, order independence, **single-filter-not-double-applied** (the direct
  regression for the reported bug), and monotonicity in thickness.
- Structural pattern: numpy-array comparisons over the returned polars column,
  matching the style of `tests/test_beer_lambert.py` (plan 001).
- Note: these run against a real `Simulation`, which requires the repo root as
  CWD — `conftest.py` (plan 001) guarantees that.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "spectrum_list\|filter_list\|add_filter\|remove_filter\|set_base_spectrum\|get_current_spectrum" code/XrayTransmission.py` returns no matches
- [ ] `grep -rn "sim.add_filter\|sim.remove_filter\|sim.set_base_spectrum\|sim.get_current_spectrum" code/app.py` returns no matches
- [ ] `grep -n "class Filter" code/XrayTransmission.py` returns no matches (renamed to `AttenuationFilter`)
- [ ] `grep -n "icecream" code/XrayTransmission.py` returns no matches
- [ ] `python -m py_compile code/app.py` exits 0
- [ ] `python -m pytest -q` exits 0; `tests/test_physics.py` has 5 passing tests
- [ ] `git status --porcelain` shows changes only to the three in-scope files
- [ ] `plans/README.md` status row for 003 updated

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `XrayTransmission.py` or `app.py` changed and the
  excerpts above don't match the live code.
- `Simulation().transmitted_spectrum('30', [AttenuationFilter('Aluminum',0.1,True)]).shape`
  is not `(971, 2)` — the join/data assumption is wrong; report the actual shape.
- `validate="1:1"` raises a polars error (would mean the mu grid now has
  duplicate energies) — report it; do not silently switch to `validate="1:m"`.
- `app.py`'s sidebar widget code around lines 57-130 differs materially from a
  standard Streamlit filter loop such that Edits B-F don't apply cleanly.

## Maintenance notes

- `transmitted_spectrum` recomputes from base every call — correct and cheap
  (one vectorized `exp` per filter over ~1000 rows). If very large filter stacks
  ever become a performance concern, sum the `μ·t` exponents in a single pass
  before one `exp`; do not reintroduce incremental caching state.
- The `inner` join drops energies below 3 keV (no attenuation data there); this
  matches prior behavior. If log-log interpolation (`CLAUDE_SUGGESTIONS.md` §5.1)
  is added later, revisit whether to extend the energy range.
- Reviewer should scrutinize: (a) that `Simulation` truly holds no per-request
  mutable attributes anymore, and (b) the app still renders both traces — run
  `streamlit run code/app.py`, add two filters, and confirm the filtered curve
  changes when you reorder or duplicate a filter (it must be order-independent
  and must not attenuate twice).
