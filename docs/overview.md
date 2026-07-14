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

The digital twin is built in two stages:

[INSERT HERE THE PROJECT STRUCTURE]

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
