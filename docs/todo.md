# To do

Day-to-day tasks and priorities for the Bressoux digital twin are tracked on
the project Trello board:

**[MAS4TE Trello board](https://trello.com/b/rfVvtiZ8/mas4te)**

Below is a non-exhaustive list of open items identified while building the
current version of the digital twin (see cards on Trello for up-to-date
status):

- Refine the electrical **line parameters** (length, R/X/C, thermal limit),
  currently set to generic placeholder values.
- Refine the **substation coordinates**, currently taken manually from
  Google Maps rather than actual GIS data.
- Wire `run_digital_twin` into `communication_agents.py` in the MAS4TE
  framework (see [Interfacing](interfacing.md)).
- Measure the network **effect** of the fleet, not just its own metrics: a run
  with and without `schedules_BSS`, comparing voltage envelopes rather than the
  rounded aggregate verdict values. The domestic batteries sit on the LV feeders
  they support, so the effect is local and readable bus by bus.
- Revisit the **1e-4 MW clipping floor** in `_assign_loads`. On the June
  scenario it flattens 29 near-vacant dwellings (whole week under 0.1 kW, at
  most 8 kWh) into one identical constant series, destroying their diversity and
  biasing them upward by ~0.05 kW each. Negligible for the verdict (2.9 kW on a
  221 kW peak) but it pollutes any per-bus statistic. The floor exists for
  power-flow convergence — check whether a lower one, or zero, still converges.
- Recover a realistic **aggregate load factor**: the independent per-bus draw
  leaves it near 0.59 against a real ~0.25 (see
  [Load assignment](load_assignment.md)). Would need a "simultaneous blocks"
  scheme grouping real EANs from the same calendar weeks.
