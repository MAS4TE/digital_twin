import os

import pandas as pd

from sourcecode.digital_twin import run_digital_twin

DATA_PATH = "data/"


def _read_csv_if_exists(path, **kwargs):
    """v1: pv_profiles / schedules_BSS / config are optional inputs — if the
    file isn't there yet, fall back to None rather than crash (run_digital_twin
    handles config=None via defaults, and ignores pv_profiles/schedules_BSS)."""
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


if __name__ == "__main__":

    config = _read_csv_if_exists(DATA_PATH + "config.csv")
    households = pd.read_csv(DATA_PATH + "households.csv")
    load_profiles = pd.read_csv(DATA_PATH + "load_profiles.csv", index_col=0)
    pv_profiles = _read_csv_if_exists(DATA_PATH + "pv_profiles.csv")
    schedules_BSS = _read_csv_if_exists(DATA_PATH + "schedules_BSS.csv")

    # --- Stress test : charges nominales vs x3 ---
    # Canal unique de mise a l'echelle : config['load_scale'], applique par
    # run_digital_twin apres construction de load_profiles (quel que soit
    # methode_affectation). Remplace l'ancienne multiplication de
    # load_profiles en amont, qui ne stressait que la methode "actuelle" et
    # laissait "banque_mensuelle" insensible a tout scenario de stress.
    config_x3 = {'load_scale': 3}

    out_normal = run_digital_twin(config, households, load_profiles,
                                  pv_profiles, schedules_BSS)
    out_x3    = run_digital_twin(config_x3, households, load_profiles,
                                  pv_profiles, schedules_BSS)

    # --- Comparaison des verdicts ---
    for name, result in [("x1", out_normal), ("x3", out_x3)]:
        valid = result[(result.category=='summary') &
                       (result.metric=='network_valid')]['value'].iloc[0]
        minv  = result[(result.category=='summary') &
                       (result.metric=='min_voltage_pu')]['value'].iloc[0]
        print(f"{name:<4} -> viable : {valid}, tension min : {minv}")

    # --- Détail complet du cas nominal ---
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print("\n--- Détail cas nominal ---")
    print(out_normal)
    print("\n--- Détail cas x3 ---")
    print(out_x3)

"""
The function `run_digital_twin` is defined in digital_twin.py inside sourcecode folder.
The function is to be introduced in communication_agents.py within the MAS4TE framework.

args: 
    - config: a pandas DataFrame containing configuration parameters for the digital twin simulation.
    - households: a pandas DataFrame containing information about households in the simulation.
    - load_profiles: a pandas DataFrame containing load profiles for the households.
    - pv_profiles: a pandas DataFrame containing photovoltaic (PV) profiles for the households.
    - schedules_BSS: a pandas DataFrame containing schedules for battery storage systems (BSS) in the households.

Returns:
    - out: a pandas DataFrame containing the timeseries and warnings of the digital twin simulation.
"""