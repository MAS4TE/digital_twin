# Load assignment

Household load profiles are assigned to the network in two decoupled stages: a
`load_profiles` table is first filled from a bank of **real weeks**, then the
common assignment step maps those profiles onto the load buses. In the MAS4TE
setting, this is how **we** produce a realistic `load_profiles` from the
Bressoux smart-meter data when the agents do not supply their own — the twin's
entry point [`run_digital_twin`](interfacing.md) still receives `load_profiles`
as an argument and never fetches it itself.

```{note}
The metered population is the project's target group of **social housing**
tenants. Consumption levels are accordingly modest and presence patterns
heterogeneous; the real weeks capture this faithfully, where generic
average-household profiles would not.
```

## Building the bank — `build_consumption_bank`

Materialises the bank of real weekly profiles as a parquet file. Run once; the
result is reused by every simulation.

```{code-block} python
from sourcecode.consumption_bank import build_consumption_bank

bank = build_consumption_bank()   # writes data/consumption_bank_real_weeks.parquet
```

`build_consumption_bank(data_path="data/", out_path="data/consumption_bank_real_weeks.parquet")`
extracts every complete calendar week (672 quarter-hourly steps, Monday→Sunday,
no gaps) from the retained meters (filtered on `total_cons >= 365` kWh) and tags
each week by EAN, month, season and street. See [Data files](data_files.md) for
the output schema.

The companion function `build_irradiance_reference(out_path=..., year=2023)`
fetches the matching PVGIS irradiance series used by the PV bricks.

## Filling `load_profiles` from the bank

Assignment is driven by `config['methode_affectation']`. The default,
`"banque_mensuelle"`, fills `load_profiles` from the bank; the legacy method
`"actuelle"` (averaged profiles) remains available for comparison. The bank
filler is `_build_load_profiles_banque_mensuelle(households, config)`.

| Argument     | Type               | Description                                                        |
|--------------|--------------------|-------------------------------------------------------------------|
| `households` | `pandas.DataFrame` | Must contain `profile_id` and `source_street`.                    |
| `config`     | `dict`             | Must contain `mois_simule`, `seed`, `bank_path`.                  |

For each `profile_id`, it draws (with replacement, using `seed`) a real week
from the sub-bank of the **same street** and of the month `mois_simule`, so the
consumption is both street-local and season-consistent. A small anti-duplicate
jitter (±1–2 quarter-hour steps) is applied when the same profile is drawn more
than once; if a street has no week for that month, a global fallback is used and
a warning is recorded.

**Returns** `(load_profiles, warnings)`, where `load_profiles` has the same
layout as `load_profiles.csv` (index `'Mon 00:00'..'Sun 23:45'`, columns
`profile_{id}`, values in **MW**).

```{note}
`mois_simule` is the single knob that keeps consumption seasonally consistent
with the PV production (which reads the same month). Consumption and irradiance
are matched at the **month** level, not the exact calendar week, since the
smart-meter and PVGIS reference years differ.
```

## Why real weeks

The bank replaces an earlier method based on weekday/time-of-day **averaged**
profiles perturbed by a random time shift and lognormal scaling. Two problems
motivated the change:

- the averaged method was **season-blind** — it produced bit-for-bit identical
  profiles regardless of the simulated month;
- the shift/scale perturbation **flattened the aggregate peak**, because it
  desynchronised profiles as much as it diversified them, underestimating the
  winter peak.

Real weeks carry authentic diversity (each household's real peaks at their real
times), which reproduces a more faithful aggregate peak without any synthetic
perturbation.

```{note}
**Known limitation.** Because profiles are still drawn **independently** per
bus, the aggregate load factor stays around 0.64, above the real ~0.25 — the
independent draw dilutes the natural coincidence of household peaks. Recovering
the true value would require a "simultaneous blocks" scheme (grouping real EANs
from the same calendar weeks). This becomes relevant when off-peak dynamics
matter, e.g. for detailed PV self-consumption studies.
```

## Regenerating the bank

```{code-block} bash
python -m sourcecode.consumption_bank   # bank + irradiance parquets
```