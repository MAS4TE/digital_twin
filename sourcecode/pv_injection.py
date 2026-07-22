"""
Volet PV -- Brique 3 : injection des installations PV comme sgen sur le
réseau, à partir de pv_installations.csv (brique 1) et des profils de
production build_pv_production() (brique 2).

Architecture "réseau de base pré-construit + couche PV" (cf. brief) :
_build_network() est appelé UNE SEULE FOIS (net_base) -- c'est la lecture du
CSV brut de 335 Mo qui coûte ~14s, PAS la construction mémoire pandapower
elle-même (mesuré dans scripts/check_pv_injection.py : le deepcopy est de
l'ordre de quelques centièmes de seconde, cf. NEXT_STEPS.md). Chaque
scénario travaille sur une copy.deepcopy(net_base), jamais sur net_base
directement, pour ne jamais accumuler de sgen/load entre scénarios.

Inversion de dependance (refonte PV natif) : la logique d'injection PROPREMENT
DITE (injection sgen, verif d'alignement, power flow charges+PV) vit desormais
dans digital_twin.py -- _inject_pv_sgens / _verify_temporal_alignment /
_run_powerflow. Ce module les IMPORTE (source unique, aucune duplication) et
ne garde que l'orchestration multi-scenarios (build_base_network + deepcopy,
verify_bus_ids sur les metadonnees, run_pv_scenario) utile a l'analyse
d'impact (scripts/pv_impact_analysis.py). Reutilise aussi _build_network,
_assign_loads, _build_load_profiles_banque_mensuelle, _config_to_dict.

Convention générateur : p_mw POSITIF = injection (PAS un load négatif) --
comportement natif de pandapower pour les éléments `sgen` (contrairement à
`load`, où p_mw positif = consommation) : aucune inversion de signe à faire
nous-mêmes, juste ne pas utiliser `create_load` avec un p_mw négatif.

q_mvar=0 pour tous les sgen PV en v1 -- hypothèse documentée : les onduleurs
résidentiels/tertiaires injectent essentiellement de l'actif au point de
fonctionnement nominal ; un contrôle Q (cos phi variable, Q(V), ...) serait
un raffinement de brique ultérieure, pas fait ici.
"""
import copy

import pandas as pd

from sourcecode.digital_twin import (
    _build_network, _assign_loads, _build_load_profiles_banque_mensuelle, _config_to_dict,
    _inject_pv_sgens, _verify_temporal_alignment, _run_powerflow,
)
from sourcecode.pv_production import build_pv_production

PV_INSTALLATIONS_PATH = "data/pv_installations.csv"
IRRADIANCE_PATH = "data/pvgis_irradiance_quarter_hourly.parquet"


def build_base_network(config=None):
    """Construit le réseau UNE fois -- à réutiliser via copy.deepcopy() pour
    chaque scénario (jamais modifié directement).

    `buses` (dict (rue,num)->bus_id) et `mv_nodes` (dict nom->bus_id) restent
    valables pour TOUTE copie de net_base : ce sont des indices entiers de
    bus pandapower, inchangés par un deepcopy (le deepcopy clone les
    DataFrames de net, pas les clés/valeurs de ces dicts externes -- pas
    besoin de les recopier ni de les reconstruire par scénario).
    """
    net_cfg = _config_to_dict(config)
    net_base, buses, mv_nodes, validation = _build_network(net_cfg)
    return net_base, buses, mv_nodes, validation


def verify_bus_ids(pv_installations, buses, mv_nodes):
    """Vérifie que chaque bus_id du CSV PV correspond à un bus réel du
    réseau : résidentiel via `buses` (333 bus), OU bus MV de passage via
    `mv_nodes` (cas de la maison de repos, hors coordonnées résidentielles).
    Lève une erreur claire plutôt qu'un échec silencieux ou un crash
    pandapower opaque plus loin dans l'injection.
    """
    valid_bus_ids = set(int(b) for b in buses.values()) | set(int(b) for b in mv_nodes.values())
    pv_bus_ids = set(int(b) for b in pv_installations["bus_id"])
    missing = sorted(pv_bus_ids - valid_bus_ids)
    if missing:
        raise ValueError(f"bus_id du CSV PV introuvables dans le reseau reel : {missing}")

    rest_home_rows = pv_installations[pv_installations["type"] == "rest_home"]
    for _, row in rest_home_rows.iterrows():
        bus_id = int(row["bus_id"])
        if bus_id not in set(int(b) for b in mv_nodes.values()):
            raise ValueError(
                f"Maison de repos : bus_id={bus_id} n'est pas un bus MV de passage connu "
                f"({sorted(mv_nodes.keys())}) -- verifier la coherence avec sourcecode/pv_metadata.py."
            )
    return True


def run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                     config=None, run_powerflow=True):
    """Exécute un scénario complet (charges + PV) sur une COPIE de
    net_base -- net_base n'est JAMAIS modifié.

    `config['mois_simule']` pilote À LA FOIS la sous-banque de conso
    (`_build_load_profiles_banque_mensuelle`) et le mois de production PV
    (`build_pv_production`) -- un seul paramètre, cohérence saisonnière
    garantie par construction (pas deux réglages à synchroniser à la main).

    Parameters
    ----------
    run_powerflow : bool -- si False, prépare tout (loads, sgen, profils,
        vérification d'alignement) mais NE LANCE PAS le power flow --
        utile pour les vérifications d'injection pure (brique 3) sans payer
        le coût du timeseries à chaque test.

    Returns
    -------
    dict : net, sgen_bus, bus_to_profile, df_load_p, df_load_q, df_sgen_p,
           et si run_powerflow=True : vm, line_load, trafo_load, sgen_p_out.
    """
    cfg = _config_to_dict(config)
    net = copy.deepcopy(net_base)

    # Charges via la banque mensuelle (le commutateur methode_affectation a ete
    # retire du twin ; run_digital_twin recoit desormais load_profiles tel quel.
    # run_pv_scenario s'appuie toujours sur la banque pour l'analyse d'impact --
    # passer load_profiles CSV directement a run_digital_twin si necessaire).
    load_profiles_for_assignment, w_bank = _build_load_profiles_banque_mensuelle(households, cfg)
    load_profiles_for_assignment = load_profiles_for_assignment * cfg['load_scale']

    df_load_p, df_load_q, bus_to_profile, warnings = _assign_loads(
        net, buses, mv_nodes, households, load_profiles_for_assignment, cfg)

    prod_kw = build_pv_production(cfg['mois_simule'], pv_installations, irradiance)
    # kW -> MW. build_pv_production() garde l'index texte 'Mon 00:00'.. ; on le
    # ramene sur un RangeIndex positionnel, identique a df_load_p (qui a perdu
    # l'index texte via `.values` dans _assign_loads). C'est cet index
    # positionnel qui fait foi au runtime (DFData/ConstControl indexent par
    # position via time_steps=range(n_timesteps)).
    df_sgen_p = (prod_kw / 1000.0).reset_index(drop=True)
    df_load_p_aligned = df_load_p.reset_index(drop=True)

    _verify_temporal_alignment(df_load_p_aligned, df_sgen_p)   # garde-fou index
    sgen_bus = _inject_pv_sgens(net, df_sgen_p)                # sgen a partir des colonnes 'pv_{bus_id}'

    result = {
        "net": net, "sgen_bus": sgen_bus, "bus_to_profile": bus_to_profile,
        "df_load_p": df_load_p, "df_load_q": df_load_q, "df_sgen_p": df_sgen_p,
        "warnings": warnings,
    }

    if run_powerflow:
        pf_out = _run_powerflow(net, df_load_p, df_load_q, cfg, df_pv_p=df_sgen_p)
        result.update(pf_out)

    return result


if __name__ == "__main__":
    import time

    households = pd.read_csv("data/households.csv")
    pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
    irradiance = pd.read_parquet(IRRADIANCE_PATH)

    t0 = time.time()
    net_base, buses, mv_nodes, validation = build_base_network()
    print(f"net_base construit en {time.time() - t0:.2f}s")

    verify_bus_ids(pv_installations, buses, mv_nodes)
    print("bus_id du CSV PV verifies contre le reseau reel : OK")

    t0 = time.time()
    net_copy = copy.deepcopy(net_base)
    print(f"copy.deepcopy(net_base) : {time.time() - t0:.4f}s")

    result = run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                              config={"mois_simule": "juin"})
    print(f"sgen crees : {len(result['net'].sgen)} (attendu {len(pv_installations)})")
    print(f"vm shape : {result['vm'].shape}")
