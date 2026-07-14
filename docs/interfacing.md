# Interfacing with MAS4TE

The digital twin is not meant to remain a standalone notebook: it is exposed
as a single function so that it can be called by the agents of the MAS4TE
framework.

## The `run_digital_twin` function

```{code-block} python
from sourcecode.digital_twin import run_digital_twin

out = run_digital_twin(config, households, load_profiles, pv_profiles, schedules_BSS)
```

| Argument         | Type               | Description                                                       |
|------------------|--------------------|---------------------------------------------------------------------|
| `config`         | `pandas.DataFrame` | Configuration parameters for the digital twin simulation.          |
| `households`     | `pandas.DataFrame` | Households taking part in the simulation (location, substation...). |
| `load_profiles`  | `pandas.DataFrame` | Household consumption profiles.                                    |
| `pv_profiles`    | `pandas.DataFrame` | Photovoltaic production profiles.                                  |
| `schedules_BSS`  | `pandas.DataFrame` | Battery storage system (BSS) schedules.                             |

**Returns** a single `pandas.DataFrame` (`out`) containing the resulting
time-series (voltages, loadings, ...) together with any warnings raised
during the simulation (e.g. households that could not be matched to a
street/substation).

A minimal usage example is given in [main.py](../main.py), which loads the
CSV inputs from a local `data/` folder and calls `run_digital_twin` directly.

```{note}
`sourcecode/digital_twin.py` currently holds the function signature; the
implementation is being migrated from [power_flow.ipynb](../power_flow.ipynb)
following the steps described in [Pipeline](pipeline.md).
```

## Plugging into MAS4TE

Within the MAS4TE environment, `run_digital_twin` is intended to be called
from `communication_agents.py`, so that:

- MAS4TE agents can query the digital twin as their **environment model**
  (e.g. "what happens to voltages/line loading if this household's battery
  is scheduled this way?");
- inputs (`config`, `households`, `load_profiles`, `pv_profiles`,
  `schedules_BSS`) are produced/updated by the agents (or by the scenario
  being simulated) instead of being read from static CSV files as in the
  standalone example;
- the returned time-series/warnings DataFrame is consumed by the agents to
  evaluate the state of the grid and adapt their decisions accordingly.

```{mermaid}
flowchart LR
    A[MAS4TE agents] -->|config, households, profiles, schedules| B(run_digital_twin)
    B -->|timeseries + warnings| A
    B --> C[(pandapower network\nBressoux)]
```

This keeps the digital twin logic self-contained in `sourcecode/`, with a
single, well-defined entry point (`run_digital_twin`) as the boundary between
the network model and the rest of the MAS4TE agent framework.
