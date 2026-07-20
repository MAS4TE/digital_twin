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

Ne modifie PAS digital_twin.py : réutilise _build_network(), _assign_loads(),
_build_load_profiles_banque_mensuelle(), _config_to_dict() en lecture seule.
_run_powerflow() de digital_twin.py n'est PAS réutilisée telle quelle (elle
ne connaît que les loads, pas les sgen) -- run_powerflow_with_pv() ci-dessous
reprend exactement son fonctionnement pour les loads et y ajoute la couche
sgen, sans toucher au fichier original.

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

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

from sourcecode.digital_twin import (
    _build_network, _assign_loads, _build_load_profiles_banque_mensuelle, _config_to_dict,
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


def inject_pv_sgens(net, pv_installations):
    """Crée un sgen par installation du CSV, sur une COPIE du réseau
    (l'appelant doit passer copy.deepcopy(net_base), jamais net_base
    directement -- cette fonction ne fait pas la copie elle-même, pour
    rester composable). p_mw=q_mvar=0 à la création (rempli au pas de temps
    par ConstControl dans run_powerflow_with_pv, même réflexe que
    digital_twin._assign_loads pour les loads).

    Noms de sgen : 'pv_{bus_id}', alignés MÉCANIQUEMENT sur les colonnes de
    build_pv_production() (même convention, aucun renommage/mapping requis).

    Returns
    -------
    sgen_bus : dict {index dans net.sgen -> bus_id}
    """
    net.sgen = net.sgen.iloc[0:0]  # jamais accumuler entre scenarios (meme reflexe que net.load dans _assign_loads)
    sgen_bus = {}
    for _, row in pv_installations.iterrows():
        bus_id = int(row["bus_id"])
        idx = pp.create_sgen(
            net, bus=bus_id, p_mw=0.0, q_mvar=0.0,
            name=f"pv_{bus_id}", type=row["type"],
        )
        sgen_bus[idx] = bus_id
    return sgen_bus


def verify_temporal_alignment(df_load_p, df_sgen_p):
    """Vérifie EXPLICITEMENT (cf. brief) que les profils de charge et de
    production PV partagent le même index (672 pas, même labels, même
    ordre) -- l'alignement au pas t est mécanique SEULEMENT si les deux
    DataFrames sont construits sur la même grille. Lève une erreur claire
    sinon plutôt que de laisser un décalage silencieux fausser le bilan
    net production/consommation.
    """
    if len(df_load_p) != len(df_sgen_p):
        raise ValueError(
            f"Longueurs differentes : load={len(df_load_p)} pas, sgen={len(df_sgen_p)} pas."
        )
    if list(df_load_p.index) != list(df_sgen_p.index):
        raise ValueError(
            "Index different entre profils de charge et de production PV -- "
            "l'alignement pas-a-pas n'est plus garanti mecaniquement."
        )
    return True


def run_powerflow_with_pv(net, df_load_p, df_load_q, df_sgen_p, n_timesteps):
    """Timeseries power flow avec loads ET sgen -- extension du
    fonctionnement de digital_twin._run_powerflow() (NON modifiee) pour
    couvrir aussi les sgen PV. q_mvar des sgen reste a 0 tout le run (v1,
    cf. en-tete du module).

    Returns
    -------
    vm : pd.DataFrame         # tension par bus [pu]
    line_load : pd.DataFrame  # charge de ligne [%]
    trafo_load : pd.DataFrame # charge de transfo [%]
    sgen_p : pd.DataFrame     # p_mw injecte par sgen, releve (verification / brique 4)
    """
    net.controller = net.controller.iloc[0:0]  # jamais accumuler entre scenarios

    data_source_load_p = DFData(df_load_p)
    data_source_load_q = DFData(df_load_q)
    for i, load in net.load.iterrows():
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=data_source_load_p, profile_name=load["name"])
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=data_source_load_q, profile_name=load["name"])

    data_source_sgen_p = DFData(df_sgen_p)
    for i, sgen in net.sgen.iterrows():
        ConstControl(net, element='sgen', variable='p_mw', element_index=[i],
                     data_source=data_source_sgen_p, profile_name=sgen["name"])

    ow = OutputWriter(net, time_steps=range(n_timesteps), output_path=None)
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'p_from_mw')    # signe : sens du flux (brique 4, flux inverses)
    ow.log_variable('res_trafo', 'loading_percent')
    ow.log_variable('res_trafo', 'p_hv_mw')     # signe : <0 = flux HT->BT inverse sur ce trafo (brique 4)
    ow.log_variable('res_ext_grid', 'p_mw')     # signe : <0 = le reseau exporte vers l'amont (brique 4)
    ow.log_variable('res_sgen', 'p_mw')

    run_timeseries(net, time_steps=range(n_timesteps), verbose=False)

    return {
        "vm": ow.output['res_bus.vm_pu'],
        "line_load": ow.output['res_line.loading_percent'],
        "line_p_from": ow.output['res_line.p_from_mw'],
        "trafo_load": ow.output['res_trafo.loading_percent'],
        "trafo_p_hv": ow.output['res_trafo.p_hv_mw'],
        "ext_grid_p": ow.output['res_ext_grid.p_mw'],
        "sgen_p_out": ow.output['res_sgen.p_mw'],
    }


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

    if cfg['methode_affectation'] == 'banque_mensuelle':
        load_profiles_for_assignment, w_bank = _build_load_profiles_banque_mensuelle(households, cfg)
    else:
        raise ValueError(
            "run_pv_scenario ne prend en charge que methode_affectation='banque_mensuelle' "
            "(passer load_profiles CSV directement si 'actuelle' est necessaire -- non requis "
            "pour la brique 3)."
        )
    load_profiles_for_assignment = load_profiles_for_assignment * cfg['load_scale']

    df_load_p, df_load_q, bus_to_profile, warnings = _assign_loads(
        net, buses, mv_nodes, households, load_profiles_for_assignment, cfg)

    sgen_bus = inject_pv_sgens(net, pv_installations)

    prod_kw = build_pv_production(cfg['mois_simule'], pv_installations, irradiance)
    df_sgen_p = prod_kw / 1000.0  # kW -> MW, colonnes 'pv_{bus_id}' deja alignees sur les noms de sgen

    # _assign_loads() construit df_load_p via `load_profiles[...].values`
    # (digital_twin.py, non modifie) : l'index string 'Mon 00:00'.. de
    # load_profiles est donc PERDU au passage, df_load_p se retrouve avec un
    # RangeIndex 0..671 -- c'est CE index (positionnel), pas les labels
    # texte, qui fait foi au runtime (DFData/ConstControl indexent par
    # position via time_steps=range(n_timesteps)). build_pv_production()
    # garde volontairement l'index texte (meme format que load_profiles.csv,
    # cf. brique 2) ; on le remet ici sur un RangeIndex identique a
    # df_load_p AVANT la verification d'alignement, pour comparer ce qui est
    # reellement consomme par le power flow, pas la forme d'origine.
    df_sgen_p = df_sgen_p.reset_index(drop=True)
    df_load_p_aligned = df_load_p.reset_index(drop=True)

    verify_temporal_alignment(df_load_p_aligned, df_sgen_p)

    result = {
        "net": net, "sgen_bus": sgen_bus, "bus_to_profile": bus_to_profile,
        "df_load_p": df_load_p, "df_load_q": df_load_q, "df_sgen_p": df_sgen_p,
        "warnings": warnings,
    }

    if run_powerflow:
        pf_out = run_powerflow_with_pv(net, df_load_p, df_load_q, df_sgen_p, cfg['n_timesteps'])
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
