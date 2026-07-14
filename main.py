import pandas as pd

from sourcecode.digital_twin import run_digital_twin

DATA_PATH = "data/"

if __name__ == "__main__":

    config = pd.read_csv(DATA_PATH + "config.csv")
    households = pd.read_csv(DATA_PATH + "households.csv")
    load_profiles = pd.read_csv(DATA_PATH + "load_profiles.csv")
    pv_profiles = pd.read_csv(DATA_PATH + "pv_profiles.csv")
    schedules_BSS = pd.read_csv(DATA_PATH + "schedules_BSS.csv")

    out = run_digital_twin(config, households, load_profiles, pv_profiles, schedules_BSS)
    
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