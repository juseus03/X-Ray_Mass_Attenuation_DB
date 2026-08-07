# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computes photon transmission through matter (Beer-Lambert, `I = I0 * exp(-mu * t)`) from the NIST
X-ray mass attenuation tables. It ships two front ends over one calculation core: an `argparse`
CLI (`xray-attenuation`) and a Dear ImGui / hello_imgui desktop GUI (`xray-attenuation-gui`).

## Commands

The conda environment from `environment.yml` is a prefix env at `./envs`, so commands below are
either run with that env active or prefixed with `./envs/bin/`.

```bash
conda env create -f environment.yml -p ./envs   # creates ./envs and pip-installs -e ".[dev]"

pytest                                          # whole suite (testpaths=tests)
pytest tests/test_physics.py                    # one file
pytest tests/test_cli_spectrum.py::TestRemoveFilter::test_removing_the_last_filter
pytest -k hvl                                   # by name

ruff check .                                    # CI gate
ruff format --diff                              # CI runs this non-blocking

xray-attenuation -m Al -t 0.1 -e 30             # single energy
xray-attenuation -f -m Al "Cadmium Telluride" -t 0.1 -e 100   # filtered spectrum + plot
xray-attenuation-gui                            # desktop GUI
```

The suite imports the *installed* package. Because of the `src/` layout there is no
`__init__.py` in `tests/` and no path hackery, so `pip install -e .` is required before it runs.

## Architecture

Four layers, each depending only on the one below it:

| Layer | File | Responsibility |
| --- | --- | --- |
| GUI | `src/xray_attenuation/app.py` | imgui_bundle front end. Holds no physics: every action calls into `CLI` |
| Orchestration + state | `src/xray_attenuation/cli.py` | `CLI` class owns `spectrum_df`, the filter stack and `max_kv`; also the argparse entry point |
| Maths | `src/xray_attenuation/physics.py` | Pure NumPy functions, no I/O, no state |
| Data | `src/xray_attenuation/data.py` | `Data` loads and looks up the NIST tables and the tungsten spectra (Polars) |

`AppState.filters` is a *live property* over `cli.filters`, not a copy — `CLI.remove_filter`
rebinds that list rather than mutating it, so it must not be cached. `CLI` is the single source
of truth for the filter stack.

### The spectrum DataFrame contract

`CLI.spectrum_df` is the shared structure between `cli.py` and `app.py`. Its columns are, in order:

1. `Energy[keV]`
2. the unfiltered spectrum, named after the tube voltage as a bare integer string (`"60"`), which
   is also `cli.max_kv`
3. one column per filter, in stack order, named `f"{index}_{material}_{thickness}cm"` with
   `index` starting at 1 (e.g. `2_Copper_0.1cm`)

Column *order* is load-bearing: `get_total_filtered_fraction`, `get_mean_energy_spectrum`,
`_aligned_arrays` and the GUI's `gui_plot` all index positionally (`columns[-1]` is the fully
filtered spectrum, `columns[1:]` are the curves to draw). Anything that adds a column must append
it at the end, or read by name.

Filtering is **cumulative**: `add_filter` applies to the output of the previous filter, so each
column holds the whole stack up to and including that filter, never that filter alone. Both front
ends label curves with a `+ Material thickness` prefix to make the accumulation explicit.

### Units

- Everything from `CLI` down is **cm** — `Filter.thickness`, `add_filter`, `physics`.
- The GUI displays and inputs **mm**, converting at the boundary (`* 1e-1` into `register_filter`,
  `* 10` when rendering the stack). Keep the conversion at that boundary.
- `physics.get_hvl` returns **mm**, the one deliberate exception.
- Energies are keV throughout. In full-spectrum mode the tube voltage snaps to an integer kV
  column via `np.round` (half-to-even: 60.5 -> 60, 61.5 -> 62) and `add_base_spectrum` prints a
  NOTICE when it snaps.

### Modules

**`data.py`** — `Data()` loads everything on construction (expensive; share the instance).
Elements come from three `.dat` files joined on `Energy` and stay lazy (`pl.scan_csv`), compounds
stay lazy, spectra are collected eagerly. `resolve_material_name` maps a symbol, element name or
compound name (any case) to `(canonical_name, is_compound)` or `None`;
`get_linear_attenuation_curve` returns the two-column frame that `filter_spectrum` joins against.
Both getters raise `MaterialNotFoundError` (a `KeyError`) for a material absent from the table.
Data is resolved as `DATA_PATH = Path(__file__).parent / "data"` and is bundled into the wheel —
never use a CWD-relative path.

**`physics.py`** — stateless. `get_transmission` is `@overload`ed and works element-wise, so the
same function serves a scalar mu and a whole attenuation curve. `get_hvl` bisects on thickness
until half the integrated spectrum survives; `get_effective_energy` inverts that HVL back to an
energy through the mu curve.

**`cli.py`** — `matplotlib.pyplot` is imported *lazily*, inside `plot_spectra` and inside the
full-spectrum branch of `main`. Keep it that way: a module-level import costs single-value runs
~320 ms, and an import in `main` alone leaves `plot_spectra` with an undefined `plt`. The
presentation style is applied through `plt.style.context(...)` so it cannot leak into the global
rcParams of an importing process.

**`app.py`** — hello_imgui docking layout with three windows: *Configuration* (`gui_commands`),
*Beam quality* (`gui_info`) and *Plot* (`gui_plot`). The beam-quality panel and the plot's
vertical markers are both driven by the `PhysicsQuantity` table built in `_make_physics_info`, so
a new derived quantity is added there and nowhere else. Fonts and assets are resolved from
`ASSETS_PATH` (package-relative) via `hello_imgui.set_assets_folder` *before* `immapp.run`,
because hello_imgui otherwise looks them up relative to the CWD. `imgui-bundle>=1.92.801` is
pinned for the dynamic-font API (`push_font(font, size)`, `get_font_baked`) and the ImPlot Spec
API (`implot.Spec`, `item_icon`, `get_last_item_color`).

## Intentional Design Decisions

Deliberate. Do **not** report these as defects in a review.

- **Cumulative stacking and its column semantics** — see the DataFrame contract above.
- **`CLI.remove_filter` recomputes every downstream filter.** That is the point of it, and it is
  the GUI's remove button. The CLI is one-shot and never calls it; it is not dead code.
- **Full-spectrum mode (`-f`) only prompts for the energy.** Material x thickness permutations
  are too large to prompt through. Missing `-m`/`-t` warns and plots the unfiltered spectrum.
- **`-m`/`-t` shape contract.** Equal-length lists are zipped into pairs; otherwise every material
  is combined with every thickness. The intended shapes are 1xN and Nx1 — a true cross product
  (3 materials x 2 thicknesses) is tolerated, not designed for.
- **The spectrum/attenuation join drops rows.** `filter_spectrum` inner-joins the 1000-row
  spectrum grid (0.1-100 keV) against the 1971-row NIST grid (3-200 keV), giving 971 rows.
  Losing 0.1-2.9 keV is physically harmless: NIST starts at 3 keV and the tungsten spectrum is
  negligible below it.
- **`1e-35` is the "no data" sentinel.** Shorter spectra in `Spectra_9-100.csv` are padded with
  it, and filtered values below it are clamped to 0 so they vanish from the log plots.
- **Fixed y-limits** (`1e-10` to `1e-5` in both `plot_spectra` and `gui_plot`) are tuned, not
  unconsidered constants.
- **`get_linear_attenuation` requires an exact grid energy** and returns `None` otherwise. There
  is no interpolation yet; that is a known gap, not a bug.

## Tests

`tests/conftest.py` pins matplotlib to Agg before the suite touches the lazy pyplot import, so
plotting tests are headless. `test_cli_spectrum.py` builds one module-scoped `Data` and injects
it by monkeypatching `xray_attenuation.cli.Data`, because construction dominates the runtime —
follow that pattern rather than constructing `Data()` per test. `test_class_data.py` asserts
exact frame shapes (1971x93 elements, 1000x93 spectra, ...), so editing anything under `data/`
is expected to break it.
