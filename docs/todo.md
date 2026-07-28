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
- Reduce the number of households **randomly assigned** to a load profile
  when their street could not be matched to a metered EAN.
- Wire `run_digital_twin` into `communication_agents.py` in the MAS4TE
  framework (see [Interfacing](interfacing.md)).
- Wire the two upstream BSS checks
  (`utils.check_bss_config_specs`, `utils.check_bss_energetic_feasibility`).
  Both are written and tested but called from nowhere, so the park specs can
  silently fall back to their defaults and the per-household slices are never
  verified against the park capacity.
- Decide whether the BSS schedule should be driven by the **aggregate** net
  load rather than per-bus net load. The park is a single MV-connected asset,
  so it only ever sees aggregate flows; the current generator
  (`data_processing/bss.py`) sums per-bus signals that would only be meaningful
  for storage behind each meter. See
  [BSS modelling](bss_modelling.md#a-known-modelling-tension).
- Measure the network **effect** of the park, not just its own metrics: a run
  with and without `schedules_BSS`, comparing voltage envelopes rather than the
  rounded aggregate verdict values.
