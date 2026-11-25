import datetime

from digital_twin import run_digital_twin, save_run_report
from digital_twin.data_processing.process_data import process_data

# - - - - TO CONFIGURE - - - -
config = {
        # Worked example: the "medium" electrification scenario (moderate
        # heat-pump/water-heater/EV adoption + PV growth on top of the base
        # Bressoux data, see data/bressoux/resflex_scenarios.py and
        # docs/electrification_scenarios.md). Alternatives already generated
        # by bressoux_setup.ipynb: "data/bressoux/scenarios/low" (today's
        # baseline, no synthetic assets), ".../high" (heavier electrification),
        # or the base "data/bressoux" scenario itself.
        # Second site: "data/heerlerheide" (Heerlen, NL), built by
        # data/heerlerheide/heerlerheide_setup.ipynb -- use 'trafo_sn_mva': 0.4
        # and 'mois_simule': 1 with it (the month its profiles were drawn for).
        'data_path': "data/bressoux/scenarios/medium",  # scenario folder (inputs/network.py, out/processed/{equipment,metadata,load_profiles}.csv, ...)
        'mois_simule': 6,                               # target month of the bank (1-12).
        'n_timesteps': 7 * 24 * 4,                      # number of timesteps in the simulation (7*24*4 for a week of 15-min intervals)
        'v_min': 0.90,                                  # minimum voltage threshold (per unit)
        'v_max': 1.1,                                   # maximum voltage threshold (per unit)
        'loading_max': 90.0,                            # maximum loading threshold (percent)
        'cos_phi': 0.95,                                # power factor (cosine of the phase angle)
        'trafo_sn_mva': 0.22,                           # transformer nominal power (MVA)
        'seed': 42,                                     # exposed for reproducible comparisons
        'save': False,                                  # whether to save the summary/violations/warnings to a Markdown report
        'verbose': True,                                # whether to print detailed logs during simulation
}
# - - - - - - - - - - - - -

if __name__ == "__main__":

# [MAS4TE] process_data() is not called within MAS4TE framework, outputs are directly passed to run_digital_twin() for simulation. 
    print(f"Loading data from {config['data_path']}...")
    metadata, network, households, pv_profiles, schedules_BSS, equipment = process_data(config)

# [MAS4TE] run_digital_twin() is the entry point for MAS4TE framework, it takes the processed data and runs the simulation.
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
[MAS4TE] run_digital_twin : single entry point MAS4TE agents should call to
evaluate a scenario (loads + optional PV/BSS layers) and get a network verdict.

args:
    config        -- simulation/validation parameters (thresholds, seed, ...)
    metadata      -- per-household attributes (profile_id, source_street, ...)
    network       -- network bundle (net, buses, mv_nodes, ...); mutated in place
    households    -- consumption profiles (n_timesteps x n_profiles), kW
    pv_profiles   -- optional, PV production per bus (kW) -- activates the PV layer
    schedules_BSS -- optional, battery fleet schedule (kW) -- activates the BSS layer
    equipment     -- optional, bus-level equipment register -- required as soon
                     as schedules_BSS names a battery

out:
    out -- long-format DataFrame: summary (verdict) + violations + warnings
"""

