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
- Finish migrating the network-reconstruction and simulation logic from
  [power_flow.ipynb](../power_flow.ipynb) into
  [digital_twin/digital_twin.py](../digital_twin/digital_twin.py)
  (`run_digital_twin`).
- Wire `run_digital_twin` into `communication_agents.py` in the MAS4TE
  framework (see [Interfacing](interfacing.md)).
- Extend the digital twin to make use of the `pv_profiles` and
  `schedules_BSS` inputs (PV production and battery scheduling), not yet
  reflected in the network simulation.
