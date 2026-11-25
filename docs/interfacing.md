# Interfacing with MAS4TE

The digital twin is not meant to remain a standalone notebook: it is exposed
as a single function so that it can be called by the agents of the MAS4TE
framework.

## The `run_digital_twin` function

```{code-block} python
from digital_twin import run_digital_twin

out = run_digital_twin(config, metadata, network, households,
                       pv_profiles=None, schedules_BSS=None, equipment=None)
```

| Argument         | Type               | Description                                                       |
|------------------|--------------------|---------------------------------------------------------------------|
| `config`         | `dict`             | Configuration parameters for the digital twin simulation.          |
| `metadata`       | `pandas.DataFrame` | Per-household attributes (`profile_id`, `source_street`, ...).      |
| `network`        | `dict`             | Network bundle (`net`, `buses`, `mv_nodes`, ...) built by a site's `inputs/network.py` -- see [Interfacing / the network input](#the-network-input) below. |
| `households`     | `pandas.DataFrame` | Household consumption timeseries, in **kW**.                        |
| `pv_profiles`    | `pandas.DataFrame`, optional | Photovoltaic production profiles, in **kW**. |
| `schedules_BSS`  | `pandas.DataFrame`, optional | Battery fleet schedule, long format, in **kW** -- see below. |
| `equipment`      | `pandas.DataFrame`, optional | Bus-level equipment register; required as soon as `schedules_BSS` names ANY battery, central (`b0`) or domestic. |

**Returns** a single `pandas.DataFrame` (`out`) containing the resulting
time-series (voltages, loadings, ...) together with any warnings raised
during the simulation (e.g. households that could not be matched to a
street/substation).

**Power-flow time series (opt-in).** With `config['export_powerflow'] = True`,
the call returns a tuple instead:

```{code-block} python
out, pf = run_digital_twin({**config, "export_powerflow": True}, metadata, network,
                           households, pv_profiles=pv_profiles,
                           schedules_BSS=schedules_BSS, equipment=equipment)
```

`pf` is a dict of DataFrames, each indexed by the step `0..n_timesteps-1`
(672 for a week), with one column per pandapower element, named by its index
in `net.bus`, `net.line`, ...:

| Key           | Element      | Unit | Sign                                         |
|---------------|--------------|------|----------------------------------------------|
| `vm`          | bus          | pu   |                                              |
| `line_load`   | line         | %    |                                              |
| `trafo_load`  | transformer  | %    |                                              |
| `line_p_from` | line         | MW   | > 0 = flow from `from_bus` towards `to_bus`  |
| `trafo_p_hv`  | transformer  | MW   | < 0 = reverse flow (towards the HV side)     |
| `ext_grid_p`  | external grid| MW   | < 0 = the network exports upstream           |
| `sgen_p_out`  | PV `sgen`    | MW   | only when `pv_profiles` is given             |
| `storage_p`   | `storage`    | MW   | > 0 = charging; only with `schedules_BSS`    |

`storage_p` is the power actually applied, after clipping when
`bss_enforce_limits` is set. The series are returned beside `out` rather than as
long-format rows: 346 buses × 672 steps alone would be 230k rows.

### The `schedules_BSS` input

The schedule of the **battery fleet**, in **long** format: the index is the time
steps repeated once per battery, and there are exactly two columns,
`battery_id` and `p_kw`.

| index | `battery_id` | `p_kw` |
|---|---|---|
| `Mon 00:00` | `b0` | `+142.7` |
| `Mon 00:15` | `b0` | `+138.2` |
| … | … | … |
| `Mon 00:00` | `b1` | `+1.4` |

`b0` is the **central** battery, `b1`…`bn` are **domestic** batteries -- both
kinds placed and sized once, at data-generation time, as rows of the
`equipment` table (see [BSS modelling](bss_modelling.md)), each on the bus its
own row gives it. A battery exists exactly when the schedule names it, so
central-only, domestic-only and both are all valid. The fleet actually read is
reported back as a `[BSS/read]` warning, so the caller can confirm the file
received is the file expected.

Each battery is **injected** as its own pandapower `storage` element, on its own
bus, driven timestep by timestep, with its own state of charge integrated
alongside. The verdict carries the fleet **aggregate**:

| Metric                                    | Meaning                                          |
|-------------------------------------------|--------------------------------------------------|
| `bss_n_batteries`, `bss_n_domestic`       | Fleet size, and how much of it is domestic       |
| `bss_e_installed_kwh`                     | Total capacity actually installed                |
| `bss_p_agg_max_kw`, `bss_p_agg_min_kw`    | Signed peaks of the applied fleet power          |
| `bss_soc_min_pct`, `bss_soc_max_pct`      | State of charge extremes across the whole fleet  |

Per-battery state of charge is opt-in through `config['export_bss_soc']`, which
appends `category == 'bss_soc'` rows. Off by default: that is
`n_batteries × n_timesteps` rows on every run.

```{note}
The `bss_*` metrics describe the batteries; they do not measure their effect on
the network. Comparing a run with and without `schedules_BSS` is what shows the
latter. Because a domestic battery sits on the LV feeder it supports, that
effect is local and visible bus by bus rather than diluted at one MV node.
```

**Identifier contract.** `b1`…`bn` must be exactly the set of `battery_id`
values the `equipment` table carries with `battery_size_kwh > 0` — compared by
**set**, never by count, and a mismatch in either direction is an error. `b0`
must resolve to an MV node, `b1`…`bn` to LV buses.

**Sign convention.** Positive = **charging** (the battery consumes, seen from
the grid), negative = discharge. This is pandapower's `storage` convention, and
the *opposite* of the `sgen` convention `pv_profiles` follows, where positive
means injection. Both live in the same engine, so the asymmetry is worth keeping
in mind when preparing the two inputs.

**Power ratings.** `b0` uses its own `battery_p_max_kw` column in `equipment`
(a park cannot derive its own power from itself). A domestic battery's rating
is derived: `battery_size_kwh × bss_c_rate(config, equipment)`, where the
C-rate defaults to the central battery's own `battery_p_max_kw /
battery_size_kwh` (0.229 C on the Bressoux scenario) unless
`config['bss_c_rate']` sets it explicitly -- required for a domestic-only
fleet, which has no central battery to derive a rate from. The fleet is then
one technology at two scales.

```{warning}
`config` no longer carries `bss_bus`, `bss_park_e_kwh` or `bss_park_p_kw` --
every battery's bus/capacity/power is now a fact of `equipment.csv`, decided
once at data-generation time (see [BSS modelling](bss_modelling.md)).
`run_digital_twin` **raises** immediately if it sees any of these three keys
in the caller's config, rather than silently ignoring a setting that used to
matter and no longer does.
```

**Feasibility.** Two independent checks warn without rejecting: the
instantaneous power against each battery's **own** rating (`[BSS/schedule]`) and
the cumulative energy against its **own** capacity (`[BSS/soc]`). Both are
clipped rather than applied as-is when `config['bss_enforce_limits']` is set,
which is the default — and the clipped series is what reaches the power flow.

```{note}
`schedules_BSS`, `households` and `pv_profiles` are all in **kW** at this
interface. The conversion to MW (pandapower's own unit) happens once, inside
`digital_twin.py`, right before injection into `net.load`/`net.sgen`/
`net.storage` -- never before. A caller handing over any of the three in MW
would be silently read as 1000x too small.
```

A minimal usage example is given in [main.py](../main.py), which loads a
scenario's inputs from `data/<site>/` via
`digital_twin.data_processing.process_data.process_data` and calls
`run_digital_twin` directly.

## Saving a run report — `save_run_report`

```{code-block} python
from digital_twin import save_run_report

save_run_report(out, "data/bressoux/out/results/run_20260101_120000.md")
```

Renders the long-format `out` DataFrame as a human-readable Markdown report
(network validity, a summary metrics table, a violations table, and a
warnings list) and writes it to `out_path`. `main.py` calls this when
`config['save']` is `True`, writing to `<data_path>/out/results/`.

(the-network-input)=
## The `network` input

Unlike `metadata`/`households`, `network` is not read from a flat CSV: it is
built by a site-specific `inputs/network.py` (e.g.
[data/bressoux/inputs/network.py](../data/bressoux/inputs/network.py)),
exposing a single `build_network(config)` entry point, and loaded dynamically
by `digital_twin.utils.load_site_network(config)` (resolved from
`config['data_path']`). This is what keeps `digital_twin/digital_twin.py`
network-agnostic: reusing the twin on a different LV network only requires
writing a new `inputs/network.py`, not touching the installable package.

```{note}
`network['net']` (the pandapower network) is mutated in place by
`run_digital_twin` (loads/sgens are cleared and repopulated on every call). A
caller that reuses the same `network` object across repeated calls is
responsible for copying it first.
```

## Plugging into MAS4TE

Within the MAS4TE environment, `run_digital_twin` is intended to be called
from `communication_agents.py`, so that:

- MAS4TE agents can query the digital twin as their **environment model**
  (e.g. "what happens to voltages/line loading if this household's battery
  is scheduled this way?");
- inputs (`config`, `metadata`, `network`, `households`, `pv_profiles`,
  `schedules_BSS`) are produced/updated by the agents (or by the scenario
  being simulated) instead of being read from static files as in the
  standalone example;
- the returned time-series/warnings DataFrame is consumed by the agents to
  evaluate the state of the grid and adapt their decisions accordingly.

```{mermaid}
flowchart LR
    A[MAS4TE agents] -->|config, metadata, network, households| B(run_digital_twin)
    B -->|timeseries + warnings| A
    B --> C[(pandapower network)]
```

This keeps the digital twin logic self-contained in `digital_twin/`, with a
single, well-defined entry point (`run_digital_twin`) as the boundary between
the network model and the rest of the MAS4TE agent framework.

## Known limitations

Limitations of the **interface** that callers and partners need to be aware of,
because the twin cannot detect or correct them on its own.

### Temporal alignment of `schedules_BSS` is not verifiable

The twin aligns the schedule to the loads **by position**: row *k* of the
schedule is taken to be the same instant as row *k* of the loads. It warns, but
never rejects:

| Schedule index | Behaviour |
| --- | --- |
| Labelled, matching the load index (`'Mon 00:00'`..`'Sun 23:45'`) | Aligned, accepted silently. |
| Labelled, *differing* from the load index | Accepted, with a `[BSS/align]` warning naming both index ranges. Catches the "labelled but time-shifted" case. |
| Bare positional index (`RangeIndex`) | Accepted silently. **Not verifiable.** |

The third row is the irreducible gap. A positional index carries no time
information, so there is nothing to compare it against: guaranteeing that row
*k* really is the same instant on both sides is the **upstream provider's
responsibility**, and the twin cannot check it. A schedule that is both bare
and time-shifted passes undetected, by construction.

```{warning}
This matters more than a typical off-by-one. Each battery is injected as
**signed** power, so a temporal offset does **not** degrade gracefully: it
inverts the battery against the load peak, turning charging into discharging
at exactly the moments that decide the result. A silently shifted schedule
therefore does not merely add noise — it falsifies the charge/discharge effect
the study exists to measure, while still producing a plausible-looking report.
```

Only the length is enforced (a schedule with a different number of timesteps
than the loads is rejected). Provide the schedule on the same labelled weekly
index as `households` whenever possible: that is what moves the case from
row three to row two, and makes an offset visible.
