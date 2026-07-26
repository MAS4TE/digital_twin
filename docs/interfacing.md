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
| `pv_profiles`    | `pandas.DataFrame`, optional | Photovoltaic production profiles.                         |
| `schedules_BSS`  | `pandas.DataFrame`, optional | Battery storage system (BSS) schedules (not yet implemented). |

**Returns** a single `pandas.DataFrame` (`out`) containing the resulting
time-series (voltages, loadings, ...) together with any warnings raised
during the simulation (e.g. households that could not be matched to a
street/substation).

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
