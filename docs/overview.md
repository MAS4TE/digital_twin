# Overview

## The MAS4TE project

This work is carried out within **MAS4TE**, a research project funded by the
European Union. This repository hosts the contribution developed at
**ULiège**: a digital twin (DT) of a real low-voltage distribution network,
used as a test bed for the multi-agent concepts studied in MAS4TE.

## The Bressoux digital twin

The neighbourhood of **Bressoux** (Liège, Belgium) was selected as a
representative case study of a low-income, residential area. Its distribution
network is reconstructed and simulated so that it can be used to test control
and coordination strategies before they are considered for a real grid.

The digital twin is built in two stages: a **data-generation** stage
(`bressoux_setup.ipynb`, run once, from the raw field data) and a **simulation**
stage (`run_digital_twin`, run as many times as needed, on the files the first
stage produced):

```{code-block} text
MAS4TE_ULIEGE/
├── main.py                    -- minimal worked example: loads a scenario, runs the twin
├── digital_twin/              -- the installable, network-agnostic engine
│   ├── digital_twin.py            run_digital_twin, save_run_report
│   ├── utils.py                    config handling, the bus id contract, BSS quantities
│   ├── data_processing/
│   │   ├── process_data.py         loads a scenario's files into run_digital_twin's inputs
│   │   ├── profiles.py             consumption bank + household metadata/load profiles
│   │   ├── equipment.py            PV / domestic battery / central battery placement
│   │   ├── pv.py                   irradiance -> production, multi-scenario PV tooling
│   │   └── bss.py                  reference battery-schedule generator (test fixture)
│   └── plots/                      visual-check and presentation figures (outside the pipeline)
├── data/bressoux/              -- the Bressoux case study (a "scenario")
│   ├── bressoux_setup.ipynb        the FIRST notebook to run -- builds everything below
│   ├── resflex_scenarios.py        electrification-scenario generator (notebook-local)
│   ├── inputs/                     true inputs: network.py, coordinates.csv, raw meter data
│   ├── out/processed/              derived data: equipment.csv, metadata.csv, load_profiles.csv, ...
│   └── scenarios/<name>/           self-contained Low/Medium/High electrification scenarios
└── docs/                       -- this documentation
```

See [Pipeline](pipeline.md) for what each generation step does,
[Data files](data_files.md) for the schema of every file above, and
[Electrification scenarios](electrification_scenarios.md) for how the
`scenarios/<name>/` folders are built.

## Why a digital twin?

A digital twin of the neighbourhood makes it possible to:

- evaluate the impact of household behaviour (consumption, PV production,
  battery storage) on the low-voltage grid without touching the real
  network;
- simulate scenarios (e.g. new PV installations, battery scheduling,
  flexibility strategies) and check for voltage or line-loading violations;
- provide the MAS4TE agents with a fast, repeatable "environment" against
  which decisions can be evaluated.

See [Pipeline](pipeline.md) for how these pieces fit together, and
[Interfacing](interfacing.md) for how the digital twin is exposed to the rest
of the MAS4TE framework.
