import datetime

from digital_twin import run_digital_twin, save_run_report
from digital_twin.data_processing.process_data import process_data


# - - - - TO MODIFY - - - -
config = {
        'data_path': "data/bressoux",              # path to the data (scenario) folder
                                                     # (inputs/network.py, inputs/pv_installations.csv, out/processed/metadata.csv, etc.)
        'simulated_week': 6 * 4,                    # target week of the bank at generation time
        'n_timesteps': 7 * 24 * 4,                  # number of timesteps in the simulation (7*24*4 for a week of 15-min intervals)
        'v_min': 0.95,                              # minimum voltage threshold (per unit)
        'v_max': 1.05,                              # maximum voltage threshold (per unit)
        'loading_max': 90.0,                        # maximum loading threshold (percent)
        'cos_phi': 0.95,                            # power factor (cosine of the phase angle)
        'trafo_sn_mva': 0.22,                       # transformer nominal power (MVA)
        'seed': 42,                                 # exposed for reproducible comparisons
        'plot': False,                              # whether to plot the results
        'save': True,                              # whether to save the summary/violations/warnings to a Markdown report
}
# - - - - - - - - - - - - -

if __name__ == "__main__":

    # Load the data files from the specified path
    print(f"Loading data from {config['data_path']}...")
    metadata, network, households, pv_profiles = process_data(config)

    # Simulating the digital twin for the specified month
    print(f"Simulating digital twin ...")
    out = run_digital_twin(config, metadata, network, households, pv_profiles=pv_profiles)

    if config['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{config['data_path'].rstrip('/')}/out/results/run_{timestamp}.md"
        save_run_report(out, report_path)
        print(f"Report written to {report_path}")

"""
The function `run_digital_twin` is defined in digital_twin/digital_twin.py.
The function is to be introduced within the MAS4TE framework.

args:
    - config: a dict containing configuration parameters for the digital twin simulation.
    - metadata: a pandas DataFrame containing per-household attributes (profile_id,
      source_street, and any other household-level attributes -- position on the
      network, inhabitants, has_bss, has_EV, has_HP, ...).
    - network: a dict (the network bundle produced by a site's inputs/network.py,
      see digital_twin.utils.load_site_network) containing the pandapower network
      and its bus lookups.
    - households: a pandas DataFrame containing the consumption timeseries
      associated to all households in the simulation.
    - pv_profiles: a pandas DataFrame containing PV production profiles, built by
      process_data from inputs/pv_installations.csv (if the scenario has one) --
      None when the scenario has no PV.

Note: `schedules_BSS` (battery schedules) is also accepted by run_digital_twin
as an optional keyword argument, not wired up in this minimal demo.

Returns:
    - out: a pandas DataFrame containing the timeseries and warnings of the digital twin simulation.

`save_run_report(out, out_path)` (also in digital_twin/digital_twin.py) renders
`out` as a human-readable Markdown report (summary/violations/warnings) and
writes it to `out_path` -- used above when config['save'] is True.
"""
