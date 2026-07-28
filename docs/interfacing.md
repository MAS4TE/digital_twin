# Interfacing with MAS4TE

The digital twin is not meant to remain a standalone notebook: it is exposed
as a single function so that it can be called by the agents of the MAS4TE
framework.

## The `run_digital_twin` function

```{code-block} python
from digital_twin import run_digital_twin

out = run_digital_twin(config, metadata, network, households, pv_profiles=None, schedules_BSS=None)
```

| Argument         | Type               | Description                                                       |
|------------------|--------------------|---------------------------------------------------------------------|
| `config`         | `dict`             | Configuration parameters for the digital twin simulation.          |
| `metadata`       | `pandas.DataFrame` | Per-household attributes (`profile_id`, `source_street`, ...).      |
| `network`        | `dict`             | Network bundle (`net`, `buses`, `mv_nodes`, ...) built by a site's `inputs/network.py` -- see [Interfacing / the network input](#the-network-input) below. |
| `households`     | `pandas.DataFrame` | Household consumption timeseries.                                   |
| `pv_profiles`    | `pandas.DataFrame`, optional | Photovoltaic production profiles, in **MW**. |
| `schedules_BSS`  | `pandas.DataFrame`, optional | Battery park schedule, in **kW** -- see below. |

**Returns** a single `pandas.DataFrame` (`out`) containing the resulting
time-series (voltages, loadings, ...) together with any warnings raised
during the simulation (e.g. households that could not be matched to a
street/substation).

### The `schedules_BSS` input

The schedule of the **central battery park**. Any number of numeric power
columns is accepted — one per battery, per agent, or per park sub-unit — and
they are summed column-wise into a single aggregate park power series. A
single pre-aggregated column is the same case, not a special one. The number
of columns read is always reported back as a `[BSS/schedule]` warning, so the
caller can confirm the file received is the file expected.

Currently the schedule is **read, validated, summed and reported** (the
`bss_n_schedule_columns`, `bss_peak_abs_kw` and `bss_mean_kw` summary metrics)
but **not injected**: the power flow result is identical with and without it.

```{warning}
`schedules_BSS` is in **kW**, while `households` and `pv_profiles` are in
**MW**. This is the only kW interface of `run_digital_twin`, and the
discrepancy cannot be detected from the values themselves — a schedule handed
over in MW is silently read as 1000x too small.
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
This matters more than a typical off-by-one. The aggregate is injected as
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
