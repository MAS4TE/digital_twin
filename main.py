import os

import pandas as pd

from digital_twin import run_digital_twin
from digital_twin.consumption_bank import generate_load_profiles
from digital_twin.pv_metadata import generate_pv_metadata
from digital_twin.pv_production import build_pv_production

DATA_PATH = "data/"

# Simulated month (configurable): drives profile GENERATION on both the load
# side (bank of real EAN weeks) and the PV side (irradiance week of the same
# month) -- a single knob, seasonal consistency guaranteed.
# 6 = June: season where the PV effect is visible (possible overvoltage, rising
# Vmax); in winter (January) PV would be negligible and invisible on Vmax (the
# slack at 1.03 pu then dominates the network maximum).
MOIS_SIMULE = 6


def _read_csv_if_exists(path, **kwargs):
    """config / schedules_BSS are optional: if the file does not exist, fall back
    to None rather than crash (run_digital_twin handles config=None via its
    defaults, and ignores schedules_BSS with a warning)."""
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


def _verdict(out):
    """Extract (valid, min voltage, max voltage) from the long-format summary."""
    s = out[out.category == 'summary'].set_index('metric')['value']
    return bool(s['network_valid']), float(s['min_voltage_pu']), float(s['max_voltage_pu'])


if __name__ == "__main__":

    households = pd.read_csv(DATA_PATH + "households.csv")
    config = _read_csv_if_exists(DATA_PATH + "config.csv")
    schedules_BSS = _read_csv_if_exists(DATA_PATH + "schedules_BSS.csv")

    # ============================================================
    #  GENERATION (upstream, outside the twin) -- generation / execution split
    # ============================================================
    # (A) LOAD profiles: the monthly bank produces data/load_profiles.csv (real
    #     EAN weeks of the chosen month). The ONLY place they are generated.
    print(f"[GEN A] load_profiles.csv (banque mensuelle, mois={MOIS_SIMULE})...")
    _, gen_warnings = generate_load_profiles(MOIS_SIMULE, households,
                                             out_path=DATA_PATH + "load_profiles.csv")
    for w in gen_warnings:
        print(f"        (warning) {w}")
    load_profiles = pd.read_csv(DATA_PATH + "load_profiles.csv", index_col=0)

    # (B) PV PRODUCTION profiles (for run 3 only): metadata (pv_metadata) ->
    #     week production for the month (pv_production, kW) -> kW -> MW conversion
    #     (CRITICAL JUNCTION: the power flow is in MW, like load_profiles). Same
    #     month as the load side -> seasonal consistency.
    print(f"[GEN B] pv_profiles (metadata -> production, mois={MOIS_SIMULE}, kW -> MW)...")
    pv_installations = generate_pv_metadata()
    irradiance = pd.read_parquet(DATA_PATH + "pvgis_irradiance_quarter_hourly.parquet")
    pv_profiles = build_pv_production(MOIS_SIMULE, pv_installations, irradiance) / 1000.0  # kW -> MW
    peak_kw = pv_profiles.sum(axis=1).max() * 1000.0
    installed_kwp = pv_installations['kwp'].sum()
    print(f"        {len(pv_profiles.columns)} installations, {installed_kwp:.1f} kWp installes, "
          f"pointe injectee {peak_kw:.1f} kW (< nameplate -> unite MW confirmee)")

    # ============================================================
    #  EXECUTION -- 3 demo scenarios (each one parameter further)
    #    Run 1 (normal): load_scale=1, pv=None
    #    Run 2 (stress): load_scale=2, pv=None   (one parameter changed vs Run 1)
    #    Run 3 (PV)    : load_scale=1, pv provided (two parameters vs Run 1)
    # ============================================================
    runs = [
        ("Run 1 normal (scale=1, PV=non)", {'load_scale': 1}, None),
        ("Run 2 stress (scale=2, PV=non)", {'load_scale': 2}, None),
        ("Run 3 PV     (scale=1, PV=oui)", {'load_scale': 1}, pv_profiles),
    ]

    results = {}
    for label, cfg, pv in runs:
        print(f"[RUN] {label} ...")
        # config.csv (if present) provides the base thresholds/parameters; the
        # scenario's load_scale is layered on top. Here config=None -> twin defaults.
        run_cfg = {**cfg} if config is None else {**config.iloc[0].to_dict(), **cfg}
        results[label] = run_digital_twin(run_cfg, households, load_profiles,
                                          pv_profiles=pv, schedules_BSS=schedules_BSS)

    # --- Verdicts side by side ---
    print("\n=== Verdicts des 3 scenarios ===")
    print(f"{'Scenario':<34} {'viable':<7} {'Vmin [pu]':<11} {'Vmax [pu]':<11}")
    for label, _, _ in runs:
        valid, vmin, vmax = _verdict(results[label])
        print(f"{label:<34} {str(valid):<7} {vmin:<11.4f} {vmax:<11.4f}")

    # --- Full detail of the PV case (run 3) ---
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print("\n--- Détail Run 3 (PV) ---")
    print(results["Run 3 PV     (scale=1, PV=oui)"])

"""
The function `run_digital_twin` is defined in digital_twin/digital_twin.py and re-exported
from the `digital_twin` package (`from digital_twin import run_digital_twin`).
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