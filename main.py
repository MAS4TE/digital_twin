import datetime

from digital_twin import run_digital_twin, save_run_report
from digital_twin.data_processing.process_data import process_data

# - - - - TO MODIFY - - - -
config = {
        'data_path': "data/bressoux",              # scenario folder (inputs/network.py,
                                                     # out/processed/{equipment,metadata,load_profiles}.csv, ...)
        'mois_simule': 6,                           # target month of the bank (1-12, or a French month name).
                                                     # Drives BOTH the load sub-bank and the PV production month,
                                                     # so load and PV stay on the same season.
        'n_timesteps': 7 * 24 * 4,                  # number of timesteps in the simulation (7*24*4 for a week of 15-min intervals)
        'v_min': 0.90,                              # minimum voltage threshold (per unit)
        'v_max': 1.1,                              # maximum voltage threshold (per unit)
        'loading_max': 90.0,                        # maximum loading threshold (percent)
        'cos_phi': 0.95,                            # power factor (cosine of the phase angle)
        'trafo_sn_mva': 0.22,                       # transformer nominal power (MVA)
        'seed': 42,                                 # exposed for reproducible comparisons
        # --- Site-calibrated equipment placement (Bressoux) ---
        # These describe THIS neighbourhood, so they live in the scenario rather
        # than in data_processing.equipment.DEFAULTS, which stays neutral.
        'street_weights': {"Rue Ernest Malvoz": 3.0},  # field report: PV seen more often there
        # PV carried by MV load facilities, {name -> kWp}. Bressoux has one, the rest home.
        'mv_pv_installations': {"Paradis du Bouhay": 12.0},
        # --- Central battery b0 (FZJ high-energy park) ---
        # Set EXPLICITLY: these describe one specific park, and
        # check_bss_config_specs warns on every run until a scenario confirms
        # them. The domestic batteries b1..bn derive their power from this pair
        # (C-rate = p/e = 0.229), so it sizes the whole fleet.
        'bss_park_e_kwh': 2600.0,
        'bss_park_p_kw': 595.0,
        'plot': False,                              # whether to plot the results
        'save': True,                              # whether to save the summary/violations/warnings to a Markdown report
}
# - - - - - - - - - - - - -

if __name__ == "__main__":

    print(f"Loading data from {config['data_path']}...")
    metadata, network, households, pv_profiles, schedules_BSS, equipment = process_data(config)

    print("Simulating digital twin ...")
    out = run_digital_twin(config, metadata, network, households,
                           pv_profiles=pv_profiles, schedules_BSS=schedules_BSS,
                           equipment=equipment)

    if config['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{config['data_path'].rstrip('/')}/out/results/run_{timestamp}.md"
        save_run_report(out, report_path)
        print(f"Report written to {report_path}")

"""
`run_digital_twin` is the entry point the MAS4TE framework calls; it and
`save_run_report` are defined in digital_twin/digital_twin.py, whose docstrings
document the arguments, units and sign conventions in full.

Key points for a caller: `schedules_BSS` is the only kW interface (households
and pv_profiles are in MW), positive = charging, and `equipment` is what places
the domestic batteries b1..bn and validates the schedule's identifiers.
"""
