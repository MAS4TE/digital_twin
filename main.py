import os

import pandas as pd

from sourcecode.consumption_bank import generate_load_profiles
from sourcecode.digital_twin import run_digital_twin
from sourcecode.pv_metadata import generate_pv_metadata
from sourcecode.pv_production import build_pv_production

DATA_PATH = "data/"

# Mois simule (configurable) : pilote la GENERATION des profils, cote conso
# (banque de vraies semaines EAN) ET cote PV (semaine d'irradiance du meme
# mois) -- un seul reglage, coherence saisonniere garantie.
# 6 = juin : saison ou l'effet PV est visible (surtension possible, Vmax qui
# monte) ; l'hiver (janvier) donnerait un PV negligeable, invisible sur Vmax
# (le slack a 1.03 pu domine alors le maximum reseau).
MOIS_SIMULE = 6


def _read_csv_if_exists(path, **kwargs):
    """config / schedules_BSS optionnels : si le fichier n'existe pas, on
    retombe sur None plutot que de crasher (run_digital_twin gere config=None
    via ses defauts, et ignore schedules_BSS avec un warning)."""
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


def _verdict(out):
    """Extrait (viable, tension min, tension max) du summary long-format."""
    s = out[out.category == 'summary'].set_index('metric')['value']
    return bool(s['network_valid']), float(s['min_voltage_pu']), float(s['max_voltage_pu'])


if __name__ == "__main__":

    households = pd.read_csv(DATA_PATH + "households.csv")
    config = _read_csv_if_exists(DATA_PATH + "config.csv")
    schedules_BSS = _read_csv_if_exists(DATA_PATH + "schedules_BSS.csv")

    # ============================================================
    #  GENERATION (en amont, hors du twin) -- separation generation / execution
    # ============================================================
    # (A) Profils de CHARGE : la banque mensuelle produit data/load_profiles.csv
    #     (vraies semaines EAN du mois choisi). SEUL endroit ou ils sont generes.
    print(f"[GEN A] load_profiles.csv (banque mensuelle, mois={MOIS_SIMULE})...")
    _, gen_warnings = generate_load_profiles(MOIS_SIMULE, households,
                                             out_path=DATA_PATH + "load_profiles.csv")
    for w in gen_warnings:
        print(f"        (warning) {w}")
    load_profiles = pd.read_csv(DATA_PATH + "load_profiles.csv", index_col=0)

    # (B) Profils de PRODUCTION PV (pour le run 3 uniquement) : metadonnees
    #     (pv_metadata) -> production semaine du mois (pv_production, kW) ->
    #     conversion kW -> MW (RACCORDEMENT CRITIQUE : le power flow est en MW,
    #     comme load_profiles). Meme mois que la conso -> coherence saisonniere.
    print(f"[GEN B] pv_profiles (metadata -> production, mois={MOIS_SIMULE}, kW -> MW)...")
    pv_installations = generate_pv_metadata()
    irradiance = pd.read_parquet(DATA_PATH + "pvgis_irradiance_quarter_hourly.parquet")
    pv_profiles = build_pv_production(MOIS_SIMULE, pv_installations, irradiance) / 1000.0  # kW -> MW
    peak_kw = pv_profiles.sum(axis=1).max() * 1000.0
    installed_kwp = pv_installations['kwp'].sum()
    print(f"        {len(pv_profiles.columns)} installations, {installed_kwp:.1f} kWp installes, "
          f"pointe injectee {peak_kw:.1f} kW (< nameplate -> unite MW confirmee)")

    # ============================================================
    #  EXECUTION -- 3 scenarios de demo (chacun UN parametre de plus)
    #    Run 1 (normal) : load_scale=1, pv=None
    #    Run 2 (stress) : load_scale=2, pv=None   (un parametre change vs Run 1)
    #    Run 3 (PV)     : load_scale=1, pv fourni (deux parametres vs Run 1)
    # ============================================================
    runs = [
        ("Run 1 normal (scale=1, PV=non)", {'load_scale': 1}, None),
        ("Run 2 stress (scale=2, PV=non)", {'load_scale': 2}, None),
        ("Run 3 PV     (scale=1, PV=oui)", {'load_scale': 1}, pv_profiles),
    ]

    results = {}
    for label, cfg, pv in runs:
        print(f"[RUN] {label} ...")
        # config.csv (si present) fournit les seuils/parametres de base ; on y
        # superpose load_scale du scenario. Ici config=None -> defauts du twin.
        run_cfg = {**cfg} if config is None else {**config.iloc[0].to_dict(), **cfg}
        results[label] = run_digital_twin(run_cfg, households, load_profiles,
                                          pv_profiles=pv, schedules_BSS=schedules_BSS)

    # --- Verdicts cote a cote ---
    print("\n=== Verdicts des 3 scenarios ===")
    print(f"{'Scenario':<34} {'viable':<7} {'Vmin [pu]':<11} {'Vmax [pu]':<11}")
    for label, _, _ in runs:
        valid, vmin, vmax = _verdict(results[label])
        print(f"{label:<34} {str(valid):<7} {vmin:<11.4f} {vmax:<11.4f}")

    # --- Détail complet du cas PV (run 3) ---
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print("\n--- Détail Run 3 (PV) ---")
    print(results["Run 3 PV     (scale=1, PV=oui)"])

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