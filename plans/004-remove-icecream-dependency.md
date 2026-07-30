# Plan 004: Remove the `icecream` debug dependency

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat b8a2fff..HEAD -- code/organize_spectra.py pyproject.toml requirements.txt`
> If any changed since this plan was written, compare against the live files
> before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/002 and plans/003 (they remove the other two `icecream`
  usages; this plan removes the last one and drops the dependency)
- **Category**: tech-debt
- **Planned at**: commit `b8a2fff`, 2026-07-29

## Why this matters

`icecream` is used only for ad-hoc debug prints. In the library it was left
**enabled** (`ic.disable()` commented out), dumping full DataFrames on every
calculation. Plans 002 and 003 replace its use in `calculate_mass_attenuation.py`
and `XrayTransmission.py` with the standard `logging` module. This plan removes
the **last** usage — in the one-off preprocessing script
`code/organize_spectra.py` — and drops `icecream` from every dependency spec, so
the project no longer ships a debug-print library as a runtime dependency.

## Current state

`code/organize_spectra.py` is a top-level preprocessing script. It uses icecream
for progress prints:

```python
# line 4
from icecream import ic
```
```python
# line 67 — inside the interpolated-file branch
ic(f"e0={e0}, e1={e1}, data shape={data.shape}, assigning to columns {idx0}:{idx1}")
```
```python
# lines 85-88 and 99 — export progress
ic(f"Exporting data to {output_file}")
ic(f"Data shape: {all_data.shape}")
ic(f"Number of labels: {len(data_labels)}")
...
ic(f"Export complete!")
```

Dependency declarations of `icecream`:
- `pyproject.toml:13` → `icecream = "^2.1.3"`
- `requirements.txt` → line `icecream`

> After plans 002 and 003 land, `grep -rln icecream code/` should list **only**
> `code/organize_spectra.py`. If it lists `calculate_mass_attenuation.py` or
> `XrayTransmission.py`, those plans are not yet done — see STOP conditions.

Convention: replace with module `logging`, matching plans 002/003. Because this
is a runnable script (not an imported library), it is acceptable here to call
`logging.basicConfig(level=logging.INFO)` once at the top so the progress
messages remain visible when the script is run.

## Commands you will need

| Purpose            | Command                                       | Expected on success     |
|--------------------|-----------------------------------------------|-------------------------|
| Compile script     | `python -m py_compile code/organize_spectra.py` | exit 0, no output     |
| No icecream left   | `grep -rn "icecream\|ic(" code/`              | no matches              |
| Full test suite    | `python -m pytest -q`                         | all pass                |

## Scope

**In scope** (the only files you should modify):
- `code/organize_spectra.py`
- `pyproject.toml` (remove the `icecream` dependency line)
- `requirements.txt` (remove the `icecream` line)

**Out of scope** (do NOT touch):
- `code/XrayTransmission.py`, `code/calculate_mass_attenuation.py` — their
  icecream removal is owned by plans 003 and 002. Do not edit them here.
- Do **not** run `code/organize_spectra.py` — it **overwrites**
  `data/W-Spectra/Spectra_9-100.csv`, which is committed reference data. This
  plan only edits the script's logging, it does not regenerate data.
- `environment.yml` — it does not list `icecream`; leave it for the packaging plan.

## Git workflow

- Branch: `advisor/004-remove-icecream`
- One commit; message e.g. `Replace icecream with logging and drop the dependency`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Swap icecream for logging in organize_spectra.py

Replace the import (line 4) `from icecream import ic` with:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
```

Replace every `ic(...)` call with `logger.info(...)`, keeping the same f-string
message, e.g.:

```python
logger.info(f"e0={e0}, e1={e1}, data shape={data.shape}, assigning to columns {idx0}:{idx1}")
...
logger.info(f"Exporting data to {output_file}")
logger.info(f"Data shape: {all_data.shape}")
logger.info(f"Number of labels: {len(data_labels)}")
...
logger.info("Export complete!")
```

**Verify**: `grep -n "icecream\|ic(" code/organize_spectra.py` → no matches;
`python -m py_compile code/organize_spectra.py` → exit 0.

### Step 2: Drop the dependency

- `pyproject.toml`: delete the line `icecream = "^2.1.3"` (line 13) from
  `[tool.poetry.dependencies]`.
- `requirements.txt`: delete the `icecream` line.

**Verify**: `grep -rn "icecream" pyproject.toml requirements.txt` → no matches.

### Step 3: Confirm nothing else imports icecream

**Verify**: `grep -rn "icecream" code/` → no matches (all three source files are
now clean).

### Step 4: Full run

**Verify**: `python -m pytest -q` → all tests still pass (the test suite does not
import `organize_spectra.py`, so this confirms no collateral breakage).

## Test plan

No new tests. `organize_spectra.py` is an un-imported one-off script; a
`py_compile` check plus the grep gates are sufficient. The existing suite from
plans 001-003 must remain green.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -rn "icecream" code/ pyproject.toml requirements.txt` returns no matches
- [ ] `python -m py_compile code/organize_spectra.py` exits 0
- [ ] `python -m pytest -q` exits 0 (suite still green)
- [ ] `git status --porcelain` shows changes only to the three in-scope files
- [ ] `plans/README.md` status row for 004 updated

## STOP conditions

Stop and report back (do not improvise) if:

- `grep -rln icecream code/` before you start lists any file **other than**
  `code/organize_spectra.py` — that means plan 002 and/or 003 have not landed;
  do them first (this plan depends on them).
- Removing the `icecream` line from `pyproject.toml` leaves the file
  syntactically broken (it should not — it's a single key in a table).

## Maintenance notes

- `organize_spectra.py` remains a top-level script by design; converting it to a
  proper `main()` under `scripts/` is deferred to the packaging work
  (`CLAUDE_SUGGESTIONS.md` §2).
- If any future code needs debug tracing, use `logging`, not `icecream`.
