"""
Digital twin of the Bressoux LV/MV distribution network.

Exposes a single entry point `run_digital_twin` callable by the MAS4TE
framework. Given household load profiles, it reconstructs the network,
assigns loads, runs a time-series power flow, and returns a verdict on
whether the network keeps voltages and line/transformer loadings within
acceptable limits.

Pipeline (see pipeline.md):
    1. Data ingestion / input validation
    2. Network reconstruction  (fixed Bressoux topology)
    3. Load assignment
    4. Time-series power flow
    5. Results evaluation (verdicts)
    6. Output formatting

Scope: LOADS + PV. `pv_profiles` est desormais une capacite native (injection
sgen, cf. _inject_pv_sgens / _run_powerflow) : si fourni, le power flow tourne
charges + PV ensemble ; si None, comportement charges seules inchange.
`schedules_BSS` reste accepte dans la signature (contrat MAS4TE) mais pas encore
implemente -> ignore avec un warning.
"""

import math
import os

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries


# ============================================================
#  FIXED INTERNAL PARAMETERS
#  (part of the frozen Bressoux model — change here for internal
#   scenarios; NOT exposed to the MAS4TE caller via `config`)
# ============================================================
RANDOM_SEED = 369
SHIFT_MAX_STEPS = 8        # load foisonnement: standard : # time shift +/- 8 steps (+/- 2h)
SCALE_SIGMA = 0.2          # load amplitude dispersion: standard 0.2 (~20%)
# (!) v1 : _assign_loads reproduit le comportement actuel de power_flow.ipynb
# (tirage direct dans la banque de profils déjà perturbée), donc ces deux
# constantes ne sont pas encore appliquées. Gardées ici pour une future
# variante qui ré-instancierait le foisonnement à chaque appel.

STEPS_PER_DAY = 96         # 15-min steps
STEPS_PER_WEEK = 672

# Index temporel hebdo 'Mon 00:00'..'Sun 23:45' -- identique a celui produit
# par profil_conso.ipynb pour load_profiles.csv. Reutilise par
# _build_load_profiles_banque_mensuelle pour que le load_profiles construit
# a partir de la banque de vraies semaines ait EXACTEMENT le meme index.
_WEEK_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
_WEEK_TIME_INDEX = [
    f"{_WEEK_DAYS[step // STEPS_PER_DAY]} "
    f"{(step % STEPS_PER_DAY) * 15 // 60:02d}:{(step % STEPS_PER_DAY) * 15 % 60:02d}"
    for step in range(STEPS_PER_WEEK)
]

# Câbles (validés / power_flow.ipynb) — pas dans le squelette d'origine,
# nécessaires pour dimensionner électriquement les lignes.
LV_KV = 0.4
MV_KV = 15

LV_CABLE = dict(r_ohm_per_km=0.642, x_ohm_per_km=0.083, c_nf_per_km=210.0, max_i_ka=0.142)   # NAYY 4x50 SE
MV_CABLE = dict(r_ohm_per_km=0.313, x_ohm_per_km=0.132, c_nf_per_km=216.0, max_i_ka=0.252)   # NA2XS2Y 1x95 RM/25 12/20 kV

# Poste source (slack) — coordonnées réelles + tension calée sur la moyenne
# mesurée par les compteurs réels (measure_ph1/2/3, ~236.9 V -> ~1.026 pu).
SLACK_COORDS = (5.610327685729984, 50.64764449828147)  # (lon, lat)
SLACK_VM_PU = 1.03          #Standard 1.03 pu

# MV network topology (fixed — real Bressoux network).
# Substations = MV nodes WITH transformer + LV secondary; feed the loads.
# Passage nodes = MV nodes WITHOUT transformer/load; structure the MV feeder.
MV_SUBSTATIONS = ["TROULOUETTE", "CIMETIERE BRESSOUX", "MALVOZ BRESSOUX", "GROTTES"]
MV_PASSAGE = ["Paradis du Bouhay", "Plantations", "Omnisport BX", "Groupe Scolaire"]

# Coordinates (lon, lat) — pp geodata convention is (x=lon, y=lat).
SUB_COORDS = {
    "TROULOUETTE":        (5.604538610198152, 50.63729347597862),
    "CIMETIERE BRESSOUX": (5.604262438977876, 50.63560256673082),
    "MALVOZ BRESSOUX":    (5.605263636742981, 50.63987545274783),
    "GROTTES":            (5.60727499545306, 50.64089847793263),  # (!) toujours une approximation Google Maps
}
PASSAGE_COORDS = {
    "Groupe Scolaire": (5.603251650027396, 50.63861456477974),
    # Paradis du Bouhay / Plantations / Omnisport BX sont extraits de
    # df_coords par @id (cf. MV_PASSAGE_IDS) et ajoutés dynamiquement.
}
# Passage nodes present in df_coords, identified by OSM @id
MV_PASSAGE_IDS = {
    757711346:  "Paradis du Bouhay",
    1071048197: "Plantations",
    55139331:   "Omnisport BX",
}

# MV feeder links (radial). Slack -> Paradis -> Plantations -> Troulouette,
# then ramifications. Distances computed from coordinates (not hard-coded).
MV_LINKS = [
    ("SLACK",              "Paradis du Bouhay"),
    ("Paradis du Bouhay",  "Plantations"),
    ("Plantations",        "TROULOUETTE"),
    ("TROULOUETTE",        "CIMETIERE BRESSOUX"),
    ("TROULOUETTE",        "MALVOZ BRESSOUX"),
    ("MALVOZ BRESSOUX",    "GROTTES"),
    ("MALVOZ BRESSOUX",    "Omnisport BX"),
    ("Omnisport BX",       "Groupe Scolaire"),
]

# Reference MV feeder lengths from the DSO one-line diagram [km].
# Used only to VALIDATE that coordinates are placed sensibly.
SCHEMA_LENGTHS = {
    ("Paradis du Bouhay",  "Plantations"):        0.600,
    ("Plantations",        "TROULOUETTE"):        0.540,
    ("TROULOUETTE",        "CIMETIERE BRESSOUX"): 0.217,
    ("TROULOUETTE",        "MALVOZ BRESSOUX"):    0.450,
    ("MALVOZ BRESSOUX",    "GROTTES"):            0.020,
    ("MALVOZ BRESSOUX",    "Omnisport BX"):       0.353,
    ("Omnisport BX",       "Groupe Scolaire"):    0.514,
}

# --- Additions nécessaires à la migration, absentes du squelette d'origine ---
# (validées dans power_flow.ipynb ; signalées ici plutôt qu'ajoutées en silence)

# Liens cross-rue "GUESSING" : sans eux, Rue des Primeroses / Rue Joseph
# Lambremont / Rue Simon Burnet / Rue des Bluets seraient déconnectées du
# reste du réseau (aucune cabine ne leur est directement rattachée côté EAN).
GUESS_LINKS = [
    (775720568, 803128095, "Primeroses30_to_Bluets5"),
    (775720553, 803128093, "Primeroses29_to_Bluets7"),
    (775720540, 803128087, "Malvoz26_to_Bluets19"),
    (775720552, 775720583, "Malvoz2_to_Lambremont40"),
    (775725219, 803128086, "Merlot82_to_Bluets2"),
    (775720602, 775725202, "Burnet42_to_Merlot118"),
]

# Coupures de lignes redondantes (rue, num_a, num_b) : nécessaires pour rester
# radial. La 1ère isole le 146 (Avenue Joseph Merlot, son propre transfo
# Cimetière) de son voisin naturel 142. La 2e coupe le lien 85<->109 qui
# apparaît une fois Plantations (107) retiré de df_charges : sans cette
# coupure, les segments Troulouette et Cimetière se reconnecteraient en
# boucle via le slack.
CHAIN_LINK_DROPS = [
    ("Avenue Joseph Merlot", "142.0", "146.0"),
    ("Avenue Joseph Merlot", "85.0", "109.0"),
]

# Profils déterministes des 4 bus MV "de passage" (maison de repos, centre
# horticole, hall omnisport, école), générés par profil_conso.ipynb. Fichier
# distinct de la banque résidentielle : ces charges sont fixes par bus, pas
# tirées au hasard. Si absent, ces 4 bus restent à charge nulle (warning).
MV_PASSAGE_PROFILES_FILE = "mv_passage_profiles.csv"

# Rattachement rue -> bus secondaire de cabine : (cabine, rue, parité, borne_basse, borne_haute).
# borne = None si le segment couvre toute la rue. La frontière entre segments
# partagés (Avenue Joseph Merlot) coïncide avec le nœud de passage Plantations.
CABINE_STREET_SEGMENTS = [
    ("MALVOZ BRESSOUX",    "Rue Ernest Malvoz",    "odd",  None,  None),
    ("MALVOZ BRESSOUX",    "Rue Ernest Malvoz",    "even", None,  None),
    ("TROULOUETTE",        "Avenue Joseph Merlot", "odd",  None,  85.0),
    ("TROULOUETTE",        "Avenue Joseph Merlot", "even", None,  142.0),
    ("CIMETIERE BRESSOUX", "Avenue Joseph Merlot", "odd",  109.0, None),
    ("CIMETIERE BRESSOUX", "Avenue Joseph Merlot", "even", 146.0, None),
    ("GROTTES",            "Plateau du Bouhay",    "odd",  None,  None),
    ("GROTTES",            "Plateau du Bouhay",    "even", None,  None),
]

# ============================================================
#  Generation des profils "banque mensuelle" — additions (integrate-bank-B)
#  cf. explore_seasonal_pv.py (branche explore-seasonal-pv) : banque de
#  vraies semaines EAN (672 pas), segmentee par mois, materialisee par
#  sourcecode/consumption_bank.py.
#
#  ARCHITECTURE (refonte separation generation/execution) : la logique de
#  generation ci-dessous (_build_load_profiles_banque_mensuelle) N'EST PLUS
#  appelee par run_digital_twin. Elle est desormais consommee EN AMONT :
#    - sourcecode/consumption_bank.generate_load_profiles -> ecrit
#      data/load_profiles.csv, que run_digital_twin relit ensuite tel quel ;
#    - sourcecode/pv_injection.py -> construit ses profils de charge.
#  Elle reste ici (importee par ces deux modules) plutot que deplacee, pour
#  ne pas casser leurs imports.
# ============================================================
CONSUMPTION_BANK_FILE = "consumption_bank_real_weeks.parquet"
BANK_JITTER_STEPS = [-2, -1, 1, 2]  # +/- 15-30 min, uniquement sur doublons

_MONTH_NAMES_FR = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}


def _normalize_month(mois_simule):
    """Accepte un numero (1-12) ou un nom de mois francais ; renvoie 1-12."""
    if isinstance(mois_simule, (int, np.integer)):
        m = int(mois_simule)
    else:
        key = str(mois_simule).strip().lower()
        if key in _MONTH_NAMES_FR:
            m = _MONTH_NAMES_FR[key]
        else:
            try:
                m = int(key)
            except ValueError:
                raise ValueError(
                    f"mois_simule ({mois_simule!r}) non reconnu (attendu un entier "
                    "1-12 ou un nom de mois francais)."
                )
    if not (1 <= m <= 12):
        raise ValueError(f"mois_simule ({mois_simule!r}) doit correspondre a un mois entre 1 et 12.")
    return m


# ============================================================
#  Small geometry / parsing helpers (private)
# ============================================================
def _distance_from_coords(lon1, lat1, lon2, lat2):
    """Chord distance on the unit sphere * Earth radius [km] (great-circle
    approximation, valid at this scale — all distances here are < 1 km)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    x1, y1, z1 = math.cos(lat1) * math.cos(lon1), math.cos(lat1) * math.sin(lon1), math.sin(lat1)
    x2, y2, z2 = math.cos(lat2) * math.cos(lon2), math.cos(lat2) * math.sin(lon2), math.sin(lat2)
    return 6371 * math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def _dist_between_buses(bus_coords, bus1, bus2):
    c1, c2 = bus_coords[int(bus1)], bus_coords[int(bus2)]
    return _distance_from_coords(c1[0], c1[1], c2[0], c2[1])


def _process_housenumber(str_hn):
    if pd.isna(str_hn) or str(str_hn).strip() == "":
        return None
    if str_hn and all(ch.isdigit() for ch in str_hn):
        return int("".join(map(str, str_hn)))
    elif str_hn and any(ch.isdigit() for ch in str_hn):
        base_num = int("".join(filter(str.isdigit, str_hn)))
        if "A" in str_hn:
            base_num += 0.2
        elif "B" in str_hn:
            base_num += 0.4
        elif "C" in str_hn:
            base_num += 0.6
        elif "D" in str_hn:
            base_num += 0.8
        return base_num
    return None


def _process_parity(num):
    if pd.isna(num):
        return "unknown"
    return "even" if int(num) % 2 == 0 else "odd"


# ============================================================
#  1. DATA INGESTION / INPUT VALIDATION
# ============================================================
def _validate_inputs(config, households, load_profiles,
                     pv_profiles, schedules_BSS):
    """Check that injected DataFrames have the expected structure.

    Returns
    -------
    warnings : list[str]
        Non-fatal issues (e.g. optional inputs missing, unsupported columns).

    Raises
    ------
    ValueError
        On fatal problems (missing mandatory columns, empty load_profiles,
        inconsistent lengths).
    """
    warnings = []

    # --- households ---
    if households is None or len(households) == 0:
        raise ValueError("`households` is empty or None: at least one household is required.")
    missing_cols = {'profile_id', 'source_street'} - set(households.columns)
    if missing_cols:
        raise ValueError(f"`households` is missing required column(s): {sorted(missing_cols)}")

    # --- load_profiles ---
    if load_profiles is None or len(load_profiles) == 0:
        raise ValueError("`load_profiles` is empty or None: at least one timestep of data is required.")
    missing_profiles = sorted(
        pid for pid in households['profile_id'].unique()
        if f"profile_{pid}" not in load_profiles.columns
    )
    if missing_profiles:
        preview = missing_profiles[:10]
        raise ValueError(
            "`load_profiles` is missing column(s) for profile_id(s) referenced by "
            f"`households`: {preview}{' ...' if len(missing_profiles) > 10 else ''}"
        )

    # --- config sanity ---
    # (required keys are always present at this point: _config_to_dict merges
    #  the caller's config with defaults before _validate_inputs is called;
    #  here we only check that the VALUES make physical sense.)
    if config['v_min'] >= config['v_max']:
        raise ValueError(f"config['v_min'] ({config['v_min']}) must be < config['v_max'] ({config['v_max']}).")
    if config['loading_max'] <= 0:
        raise ValueError(f"config['loading_max'] ({config['loading_max']}) must be positive.")
    if not (0 < config['cos_phi'] <= 1):
        raise ValueError(f"config['cos_phi'] ({config['cos_phi']}) must be in (0, 1].")
    if config['n_timesteps'] > len(load_profiles):
        raise ValueError(
            f"config['n_timesteps'] ({config['n_timesteps']}) exceeds the number of "
            f"timesteps available in `load_profiles` ({len(load_profiles)})."
        )
    if config['load_scale'] <= 0:
        raise ValueError(f"config['load_scale'] ({config['load_scale']}) must be positive.")

    # --- pv_profiles (optionnel) : capacite native desormais supportee.
    # Format attendu : colonnes 'pv_{bus_id}' (bus_id entier), index temporel
    # aligne sur load_profiles ('Mon 00:00'..'Sun 23:45'), valeurs en MW (meme
    # unite que load_profiles -- cf. run_digital_twin). On valide ici la forme
    # des colonnes et la couverture temporelle ; l'existence des bus et
    # l'alignement d'index sont verifies au moment de l'injection
    # (_inject_pv_sgens / _verify_temporal_alignment).
    if pv_profiles is not None and len(pv_profiles) > 0:
        bad_cols = [c for c in pv_profiles.columns
                    if not (str(c).startswith('pv_') and str(c)[3:].isdigit())]
        if bad_cols:
            raise ValueError(
                "`pv_profiles` a des colonnes hors format 'pv_{bus_id}' (bus_id entier) : "
                f"{bad_cols[:10]}{' ...' if len(bad_cols) > 10 else ''}"
            )
        if config['n_timesteps'] > len(pv_profiles):
            raise ValueError(
                f"config['n_timesteps'] ({config['n_timesteps']}) depasse le nombre de pas "
                f"disponibles dans `pv_profiles` ({len(pv_profiles)})."
            )

    # --- schedules_BSS : pas encore supporte ---
    if schedules_BSS is not None and len(schedules_BSS) > 0:
        warnings.append("`schedules_BSS` was provided but is not yet supported (loads + PV only) — ignored.")

    return warnings


# ============================================================
#  2. NETWORK RECONSTRUCTION  (fixed Bressoux topology)
# ============================================================
def _build_network(config):
    """Reconstruct the fixed Bressoux MV/LV network.

    Loads the building coordinates (df_coords) and meter data (df_eans)
    internally, extracts the MV passage nodes, creates the slack, MV buses,
    transformers (one per substation), LV load buses, and all lines.

    Returns
    -------
    net : pandapowerNet
    buses : dict           # (street, num) -> LV bus id  (charge buses)
    mv_nodes : dict        # name -> MV bus id  (SLACK + substations + passage)
    validation : dict      # coordinate-vs-schema distance check
    """
    data_path = config['data_path']

    # --- Load raw data ---
    df_eans = pd.read_csv(
        data_path + "part-00000-3895d355-7ed8-4cff-a072-bdc8805c8651-c000.csv",
        dtype={13: str, 15: str},
    )
    df_coords = pd.read_csv(data_path + "coordonnées.csv", sep='\t')

    total_cons = df_eans.groupby('ean_anon')['measure_cons'].agg(lambda s: s.max() - s.min())
    keep_eans = total_cons[total_cons >= 365].index
    df_eans = df_eans[df_eans['ean_anon'].isin(keep_eans)].copy()

    # --- Housenumber / parity parsing ---
    df_coords["addr:housenumber"] = df_coords["addr:housenumber"].replace("", None)
    df_coords["num"] = df_coords["addr:housenumber"].apply(_process_housenumber)
    df_coords["num_parity"] = df_coords["num"].apply(_process_parity)
    df_coords = df_coords.dropna(subset=["num"]).copy()
    df_coords["@lat"] = df_coords["@lat"].astype(float)
    df_coords["@lon"] = df_coords["@lon"].astype(float)
    # (!) Aucune correction manuelle de numéro de rue à appliquer ici : le
    # swap 2A/2/2B qui existait dans power_flow.ipynb a été supprimé (devenu
    # un no-op une fois Omnisport/2A extrait comme bus MV de passage).

    # --- Extract MV passage nodes (present in df_coords) by @id ---
    df_mv_passage = df_coords[df_coords["@id"].isin(MV_PASSAGE_IDS)].copy()
    df_mv_passage["mv_name"] = df_mv_passage["@id"].map(MV_PASSAGE_IDS)
    df_charges = df_coords[~df_coords["@id"].isin(MV_PASSAGE_IDS)].copy()

    net = pp.create_empty_network(name="Bressoux")
    bus_coords = {}
    mv_nodes = {}
    buses = {}
    bus_by_id = {}
    cabine_lv_bus = {}

    # --- MV buses: slack, substations, passage nodes ---
    slack_bus = pp.create_bus(net, vn_kv=MV_KV, name="SLACK", geodata=SLACK_COORDS)
    pp.create_ext_grid(net, bus=slack_bus, vn_kv=MV_KV, vm_pu=SLACK_VM_PU, name="Substation_Slack_Bus")
    mv_nodes["SLACK"] = slack_bus
    bus_coords[slack_bus] = SLACK_COORDS

    for name in MV_SUBSTATIONS:
        b = pp.create_bus(net, vn_kv=MV_KV, name=name, geodata=SUB_COORDS[name])
        mv_nodes[name] = b
        bus_coords[b] = SUB_COORDS[name]

    passage_coords = dict(PASSAGE_COORDS)
    for _, row in df_mv_passage.iterrows():
        passage_coords[row["mv_name"]] = (row["@lon"], row["@lat"])

    for name in MV_PASSAGE:
        b = pp.create_bus(net, vn_kv=MV_KV, name=name, geodata=passage_coords[name])
        mv_nodes[name] = b
        bus_coords[b] = passage_coords[name]

    # --- LV load buses (one per house in df_charges) ---
    for _, row in df_charges.iterrows():
        street_name = str(row["addr:street"])
        street_num = str(row["num"])
        b = pp.create_bus(
            net, vn_kv=LV_KV, name=f"{street_name}_{street_num}",
            geodata=(row["@lon"], row["@lat"]),
        )
        buses[(street_name, street_num)] = b
        bus_coords[b] = (row["@lon"], row["@lat"])
        bus_by_id[row["@id"]] = b

    # --- Street chaining (consecutive same-parity housenumbers) ---
    for street, grp in df_charges.groupby("addr:street"):
        for parity in ["even", "odd"]:
            subset = grp[grp["num_parity"] == parity].sort_values("num").reset_index(drop=True)
            if len(subset) < 2:
                continue
            for i in range(len(subset) - 1):
                num_curr, num_next = str(subset.loc[i, "num"]), str(subset.loc[i + 1, "num"])
                from_bus, to_bus = buses[(street, num_curr)], buses[(street, num_next)]
                pp.create_line_from_parameters(
                    net, from_bus=from_bus, to_bus=to_bus,
                    length_km=_dist_between_buses(bus_coords, from_bus, to_bus),
                    **LV_CABLE,
                    name=f"{street}_{num_curr}_to_{num_next}",
                )

    def _drop_line_by_name(name):
        net.line.drop(net.line[net.line["name"] == name].index, inplace=True)

    # --- Cross-street links for streets without a directly known substation ---
    for id_a, id_b, label in GUESS_LINKS:
        bus_a, bus_b = bus_by_id[id_a], bus_by_id[id_b]
        pp.create_line_from_parameters(
            net, from_bus=bus_a, to_bus=bus_b,
            length_km=_dist_between_buses(bus_coords, bus_a, bus_b),
            **LV_CABLE,
            name=f"add_{label}",
        )

    # --- Drop redundant / loop-forming chain links ---
    for street, num_a, num_b in CHAIN_LINK_DROPS:
        _drop_line_by_name(f"{street}_{num_a}_to_{num_b}")

    # --- One transformer + one LV secondary bus per real substation ---
    for name in MV_SUBSTATIONS:
        sub_lon, sub_lat = bus_coords[mv_nodes[name]]
        lv_bus = pp.create_bus(net, vn_kv=LV_KV, name=f"{name}_LV_secondaire", geodata=(sub_lon, sub_lat))
        cabine_lv_bus[name] = lv_bus
        bus_coords[lv_bus] = (sub_lon, sub_lat)
        pp.create_transformer_from_parameters(
            net, hv_bus=mv_nodes[name], lv_bus=lv_bus,
            sn_mva=config['trafo_sn_mva'], vn_hv_kv=MV_KV, vn_lv_kv=LV_KV,
            vk_percent=4.0, vkr_percent=1.0,
            pfe_kw=0.6 * (config['trafo_sn_mva'] / 0.4), i0_percent=0.2,
            name=f"trafo_{name.replace(' ', '_')}",
        )

    # --- MV backbone (radial, per MV_LINKS) ---
    for a, b in MV_LINKS:
        pp.create_line_from_parameters(
            net, from_bus=mv_nodes[a], to_bus=mv_nodes[b],
            length_km=_dist_between_buses(bus_coords, mv_nodes[a], mv_nodes[b]),
            **MV_CABLE,
            name=f"MV_{a}_to_{b}",
        )

    # --- Validate MV distances against the DSO one-line diagram (informative only) ---
    validation = {}
    for (a, b), schema_km in SCHEMA_LENGTHS.items():
        computed_km = _dist_between_buses(bus_coords, mv_nodes[a], mv_nodes[b])
        validation[(a, b)] = {
            "schema_km": schema_km,
            "computed_km": computed_km,
            "diff_km": computed_km - schema_km,
            "ratio": (computed_km / schema_km) if schema_km else None,
        }

    # --- Attach each street segment to the right substation's LV secondary bus ---
    for sub_name, street, parity, lo, hi in CABINE_STREET_SEGMENTS:
        seg = df_charges[(df_charges["addr:street"] == street) & (df_charges["num_parity"] == parity)]
        if lo is not None:
            seg = seg[seg["num"] >= lo]
        if hi is not None:
            seg = seg[seg["num"] <= hi]
        seg = seg.sort_values("num").reset_index(drop=True)

        lv_bus = cabine_lv_bus[sub_name]
        mv_bus = mv_nodes[sub_name]
        seg_buses = [buses[(street, str(n))] for n in seg["num"]]
        dists = [_dist_between_buses(bus_coords, mv_bus, b) for b in seg_buses]
        idx_min = min(range(len(dists)), key=lambda i: dists[i])
        base_name = f"LV_{sub_name.replace(' ', '_')}_to_{street}_{parity}"

        def _connect(entry_bus, label):
            pp.create_line_from_parameters(
                net, from_bus=lv_bus, to_bus=entry_bus,
                length_km=_dist_between_buses(bus_coords, lv_bus, entry_bus),
                **LV_CABLE,
                name=label,
            )

        if idx_min == 0 or idx_min == len(seg_buses) - 1:
            _connect(seg_buses[idx_min], base_name)
        else:
            other_idx = idx_min - 1 if dists[idx_min - 1] < dists[idx_min + 1] else idx_min + 1
            num_a, num_b = sorted((seg.loc[idx_min, "num"], seg.loc[other_idx, "num"]))
            _drop_line_by_name(f"{street}_{num_a}_to_{num_b}")
            for i in (idx_min, other_idx):
                _connect(seg_buses[i], f"{base_name}_n{seg.loc[i, 'num']}")

    return net, buses, mv_nodes, validation


# ============================================================
#  3. LOAD ASSIGNMENT
#  Une seule fonction d'assignation (_assign_loads) : elle consomme le
#  `load_profiles` RECU par run_digital_twin (contrat d'interface), peu
#  importe qui l'a produit et comment. La generation de ce load_profiles se
#  fait en AMONT (hors du twin), cf. sourcecode/consumption_bank.py.
#
#  Ou vit la methode "actuelle" ? C'etait le CSV moyenne+shift+scale de
#  profil_conso.ipynb (data/load_profiles.csv historique), autrefois selectionne
#  via config['methode_affectation']='actuelle'. Ce commutateur est retire du
#  twin (option A) : "actuelle" n'est plus cablee comme methode de generation.
#  Son artefact (le CSV plat) et le notebook qui le produit ne sont pas
#  supprimes -- ils restent disponibles pour une re-comparaison, mais hors du
#  chemin principal (main.py genere desormais load_profiles.csv via la banque
#  mensuelle). cf. integrate-bank-B : _assign_loads_bank_mensuelle (2e fonction
#  d'assignation qui court-circuitait load_profiles) reste supprimee.
# ============================================================
def _assign_loads(net, buses, mv_nodes, households, load_profiles, config):
    """Assign a (perturbed) load profile to each LV charge bus, plus the
    fixed deterministic profiles of the 4 MV passage facilities.

    Uses the household->street mapping (light street-based association;
    global draw for orphan buses). Applies the frozen stochastic
    perturbation (SHIFT_MAX_STEPS, SCALE_SIGMA) and reactive power (cos_phi).

    Inchangee par la refonte integrate-bank-B : `load_profiles` est un
    contrat d'interface, peu importe qui l'a rempli (banque plate CSV ou
    banque mensuelle de vraies semaines) et comment.

    Returns
    -------
    df_profiles_p : pd.DataFrame   # active power per load [MW], n_timesteps rows
    df_profiles_q : pd.DataFrame   # reactive power per load [MVAr]
    bus_to_profile : dict          # load name -> profile_id (households) or
                                    # facility name (MV passage), traceability
    warnings : list[str]           # e.g. households not matched to a street
    """
    warnings = []
    rng = np.random.default_rng(config['seed'])

    # street -> [profile_id, ...] ; global fallback pool for unmatched buses
    profiles_by_street = households.groupby('source_street')['profile_id'].apply(list).to_dict()
    all_ids = households['profile_id'].tolist()

    net.load = net.load.iloc[0:0]

    profiles_dict = {}
    bus_to_profile = {}

    for (street, num), b in buses.items():
        load_name = f"Load_{b}"
        pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)

        if street in profiles_by_street:
            pick = rng.choice(profiles_by_street[street])
        else:
            # Tirage global attendu pour les rues sans profil dédié dans
            # households.csv (comportement normal, pas une anomalie à signaler).
            pick = rng.choice(all_ids)

        profiles_dict[load_name] = load_profiles[f"profile_{pick}"].values
        bus_to_profile[load_name] = pick

    # (!) v1 : pas de perturbation shift/scale ici (SHIFT_MAX_STEPS/SCALE_SIGMA
    # non appliqués) — reproduit le comportement actuel de power_flow.ipynb,
    # où la banque de profils est déjà perturbée en amont (profil_conso.ipynb).

    # --- Profils fixes des 4 bus MV "de passage" (maison de repos, centre
    # horticole, hall omnisport, école) — cf. discussion : ajoutés ici plutôt
    # que laissés à charge nulle, moyennant l'élargissement de signature
    # (net, buses, mv_nodes, ...) déjà validé avec l'utilisateur (Option B).
    mv_profiles_path = config['data_path'] + MV_PASSAGE_PROFILES_FILE
    if os.path.exists(mv_profiles_path):
        mv_passage_profiles = pd.read_csv(mv_profiles_path, index_col=0)
        for name in MV_PASSAGE:
            b = mv_nodes[name]
            load_name = f"Load_{b}"
            pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)
            profiles_dict[load_name] = mv_passage_profiles[name].values
            bus_to_profile[load_name] = name
    else:
        warnings.append(
            f"Fichier de profils MV de passage introuvable ({mv_profiles_path}) — "
            f"{', '.join(MV_PASSAGE)} laissés à charge nulle."
        )

    df_profiles_p = pd.DataFrame(profiles_dict).clip(lower=1e-4)

    tan_phi = np.tan(np.arccos(config['cos_phi']))
    df_profiles_q = df_profiles_p * tan_phi

    return df_profiles_p, df_profiles_q, bus_to_profile, warnings


# ============================================================
#  3bis. INJECTION PV (capacite native ; rapatriee de pv_injection.py)
#  Source UNIQUE de la logique d'injection : pv_injection.py et
#  scripts/pv_impact_analysis.py importent desormais ces fonctions d'ici
#  (inversion de dependance -- le fichier central est autonome, il ne depend
#  plus d'un fichier outil).
# ============================================================
def _inject_pv_sgens(net, pv_profiles):
    """Cree un sgen par colonne 'pv_{bus_id}' de `pv_profiles`, sur `net`.

    Convention pandapower : sgen p_mw POSITIF = injection (PAS un load
    negatif). p_mw=q_mvar=0 a la creation (rempli au pas de temps par
    ConstControl dans _run_powerflow, meme reflexe que _assign_loads pour les
    loads). q_mvar=0 tout le run (v1 : onduleurs en actif pur).

    net.sgen est vide au prealable (jamais accumuler entre appels repetes,
    meme reflexe que net.load dans _assign_loads).

    Returns
    -------
    sgen_bus : dict {index dans net.sgen -> bus_id}
    """
    net.sgen = net.sgen.iloc[0:0]
    valid_bus = set(net.bus.index)
    sgen_bus = {}
    for col in pv_profiles.columns:
        # 'pv_123' -> 123 (format garanti par _validate_inputs cote twin ;
        # revalide ici l'existence du bus car pv_injection appelle aussi
        # directement, sans passer par _validate_inputs).
        bus_id = int(str(col).split('_', 1)[1])
        if bus_id not in valid_bus:
            raise ValueError(
                f"pv_profiles : colonne {col!r} vise un bus_id {bus_id} absent du reseau."
            )
        idx = pp.create_sgen(net, bus=bus_id, p_mw=0.0, q_mvar=0.0, name=str(col))
        sgen_bus[idx] = bus_id
    return sgen_bus


def _verify_temporal_alignment(df_load_p, df_pv_p):
    """Garde-fou temporel (bug d'index deja rencontre) : charges et PV doivent
    partager le MEME index (meme longueur, memes labels, meme ordre) --
    l'alignement au pas t n'est mecanique que si c'est le cas. Leve une erreur
    claire plutot que de laisser un decalage silencieux fausser le bilan net
    production/consommation."""
    if len(df_load_p) != len(df_pv_p):
        raise ValueError(
            f"Longueurs differentes : load={len(df_load_p)} pas, PV={len(df_pv_p)} pas."
        )
    if list(df_load_p.index) != list(df_pv_p.index):
        raise ValueError(
            "Index different entre profils de charge et de production PV -- "
            "l'alignement pas-a-pas n'est plus garanti mecaniquement."
        )
    return True


def _load_consumption_bank(bank_path):
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            f"Banque de semaines reelles introuvable ({bank_path}). Executer "
            "`python sourcecode/consumption_bank.py` pour la materialiser avant "
            "de generer load_profiles via la banque mensuelle "
            "(consumption_bank.generate_load_profiles)."
        )
    return pd.read_parquet(bank_path)


def _build_load_profiles_banque_mensuelle(households, config):
    """Remplit un `load_profiles` dans EXACTEMENT le meme format que celui
    produit par profil_conso.ipynb pour la methode "actuelle" (index
    'Mon 00:00'..'Sun 23:45', colonnes profile_{id}, valeurs en MW) mais dont
    le CONTENU vient de vraies semaines EAN du mois `config['mois_simule']`
    (cf. sourcecode/consumption_bank.py) au lieu de la moyenne+shift+scale.

    Pour chaque profile_id de `households`, tire (avec remise, seed=
    config['seed']) une vraie semaine dans la sous-banque de la MEME rue
    (colonne 'street' de la banque, alignee sur households['source_street'] --
    meme champ source addressStreet, meme correction de nom, cf.
    consumption_bank.py). Repli global (toutes rues confondues) si la rue n'a
    aucune semaine ce mois-la. Micro-jitter anti-doublon (+/-1-2 pas, 15-30
    min) sur les occurrences repetees d'une meme ligne de banque tiree
    plusieurs fois. Aucun shift/scale sinon.

    `_assign_loads` ne voit ensuite qu'un `load_profiles` different rempli :
    l'association par rue (households -> profile_id -> load_profiles) qu'il
    fait en aval fonctionne a l'identique, sans aucune modification de son
    code.

    Returns
    -------
    load_profiles : pd.DataFrame   # meme forme que load_profiles.csv
    warnings : list[str]
    """
    warnings = []
    month = _normalize_month(config['mois_simule'])
    bank = _load_consumption_bank(config['bank_path'])
    if 'street' not in bank.columns:
        raise ValueError(
            f"{config['bank_path']} ne contient pas de colonne 'street' -- "
            "regenerer la banque avec `python sourcecode/consumption_bank.py` "
            "(version a jour de consumption_bank.py, qui l'ajoute)."
        )

    pool_month = bank[bank['month'] == month].reset_index(drop=True)
    if len(pool_month) == 0:
        raise ValueError(f"Aucune semaine reelle disponible pour le mois {month} dans {config['bank_path']}.")

    value_cols = [c for c in bank.columns if c.startswith('t') and c[1:].isdigit()]
    # positions (dans pool_month, indexation partagee) des semaines de chaque rue
    pool_positions_by_street = {
        street: idx.to_numpy() for street, idx in pool_month.groupby('street').groups.items()
    }
    all_positions = pool_month.index.to_numpy()

    rng = np.random.default_rng(config['seed'])

    street_by_profile = households.drop_duplicates('profile_id').set_index('profile_id')['source_street']
    profile_ids = sorted(households['profile_id'].unique())

    data = {}
    draw_counts = {}
    streets_missing_this_month = set()
    for pid in profile_ids:
        street = street_by_profile.loc[pid]
        candidates = pool_positions_by_street.get(street)
        if candidates is None or len(candidates) == 0:
            streets_missing_this_month.add(street)
            candidates = all_positions

        j = int(candidates[rng.integers(len(candidates))])
        values = pool_month.loc[j, value_cols].to_numpy(dtype=float)  # deja en MW

        draw_counts[j] = draw_counts.get(j, 0) + 1
        if draw_counts[j] > 1:
            step = rng.choice(BANK_JITTER_STEPS)
            values = np.roll(values, step)

        data[f"profile_{pid}"] = values

    if streets_missing_this_month:
        warnings.append(
            f"Aucune semaine reelle pour la/les rue(s) {sorted(streets_missing_this_month)} "
            f"au mois {month} -- repli sur le pool global pour les profils concernes."
        )

    load_profiles = pd.DataFrame(data, index=_WEEK_TIME_INDEX)
    load_profiles.index.name = "week_step"
    return load_profiles, warnings


# ============================================================
#  4. TIME-SERIES POWER FLOW
# ============================================================
def _run_powerflow(net, df_profiles_p, df_profiles_q, config, df_pv_p=None):
    """Attach controllers and run the time-series power flow.

    Charges TOUJOURS. Si `df_pv_p` est fourni (colonnes 'pv_{bus_id}', MW ;
    sgen deja poses par _inject_pv_sgens), ajoute la couche PV et releve en
    plus les signaux de flux (sens/inversion) utiles a l'analyse d'impact PV
    (surtension, flux inverses). Si `df_pv_p` est None, power flow charges
    seules -- comportement strictement inchange (non-regression).

    Rapatrie de pv_injection.run_powerflow_with_pv : une seule copie de la
    logique charges+PV existe desormais (ici).

    Returns
    -------
    dict : 'vm', 'line_load', 'trafo_load' (toujours) et, si PV,
    'line_p_from', 'trafo_p_hv', 'ext_grid_p', 'sgen_p_out'.
    """
    net.controller = net.controller.iloc[0:0]  # avoid stacking controllers on repeated calls

    data_source_p = DFData(df_profiles_p)
    data_source_q = DFData(df_profiles_q)

    for i, load in net.load.iterrows():
        load_column = load["name"]
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=data_source_p, profile_name=load_column)
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=data_source_q, profile_name=load_column)

    with_pv = df_pv_p is not None
    if with_pv:
        data_source_pv = DFData(df_pv_p)
        for i, sgen in net.sgen.iterrows():
            ConstControl(net, element='sgen', variable='p_mw', element_index=[i],
                         data_source=data_source_pv, profile_name=sgen["name"])

    n_timesteps = config['n_timesteps']
    # output_path=None -> results collected in memory only (ow.output), no CSV
    # round-trip through disk as in the notebook: cleaner for a callable module.
    ow = OutputWriter(net, time_steps=range(n_timesteps), output_path=None)
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')
    if with_pv:
        ow.log_variable('res_line', 'p_from_mw')     # signe : sens du flux (flux inverses)
        ow.log_variable('res_trafo', 'p_hv_mw')      # signe : <0 = flux HT->BT inverse sur ce trafo
        ow.log_variable('res_ext_grid', 'p_mw')      # signe : <0 = le reseau exporte vers l'amont
        ow.log_variable('res_sgen', 'p_mw')

    run_timeseries(net, time_steps=range(n_timesteps), verbose=False)

    out = {
        'vm': ow.output['res_bus.vm_pu'],
        'line_load': ow.output['res_line.loading_percent'],
        'trafo_load': ow.output['res_trafo.loading_percent'],
    }
    if with_pv:
        out['line_p_from'] = ow.output['res_line.p_from_mw']
        out['trafo_p_hv'] = ow.output['res_trafo.p_hv_mw']
        out['ext_grid_p'] = ow.output['res_ext_grid.p_mw']
        out['sgen_p_out'] = ow.output['res_sgen.p_mw']
    return out


# ============================================================
#  5. RESULTS EVALUATION (verdicts)
# ============================================================
def _collect_violations(df, metric, threshold, mode):
    """Scan a (n_timesteps x n_element) DataFrame for out-of-bounds values.

    mode: 'below' -> df < threshold ; 'above' -> df > threshold.
    Returns a list of violation dicts, one per (timestep, element) breach.
    """
    mask = (df < threshold) if mode == "below" else (df > threshold)
    if not mask.values.any():
        return []
    detail = f"{'<' if mode == 'below' else '>'} {threshold}"
    violations = []
    for (timestep, element), value in df.where(mask).stack().items():
        violations.append({
            'element': element,
            'metric': metric,
            'value': float(value),
            'timestep': int(timestep) if isinstance(timestep, (int, np.integer)) else timestep,
            'detail': detail,
        })
    return violations


def _aggregate_profile_summary(df_profiles_p):
    """Metriques agregees (integrate-bank-B) : somme des puissances actives
    de toutes les charges a chaque pas, ramenee en kW pour la lisibilite.
    Additif au summary existant -- ne remplace aucune metrique deja
    presente."""
    agg_mw = df_profiles_p.sum(axis=1)
    peak_mw = float(agg_mw.max())
    mean_mw = float(agg_mw.mean())
    summary = {
        'peak_agg_kw': peak_mw * 1000.0,
        'mean_agg_kw': mean_mw * 1000.0,
        'load_factor_agg': (mean_mw / peak_mw) if peak_mw > 0 else None,
    }
    return summary, agg_mw


def _evaluate(vm, line_load, trafo_load, config):
    """Turn raw results into a network verdict + list of violations.

    Returns
    -------
    summary : dict           # network_valid, min_voltage, max_loading, counts
    violations : list[dict]  # one entry per (element, timestep) out of bounds
    """
    v_min, v_max = config['v_min'], config['v_max']
    load_max = config['loading_max']

    violations = []
    if vm is not None:
        violations += _collect_violations(vm, 'vm_pu', v_min, 'below')
        violations += _collect_violations(vm, 'vm_pu', v_max, 'above')
    if line_load is not None:
        violations += _collect_violations(line_load, 'line_loading_pct', load_max, 'above')
    if trafo_load is not None:
        violations += _collect_violations(trafo_load, 'trafo_loading_pct', load_max, 'above')

    n_voltage_violations = sum(1 for v in violations if v['metric'] == 'vm_pu')
    n_loading_violations = len(violations) - n_voltage_violations

    summary = {
        'network_valid': len(violations) == 0,
        'min_voltage_pu': float(vm.values.min()) if vm is not None else None,
        'max_voltage_pu': float(vm.values.max()) if vm is not None else None,
        'max_line_loading_pct': float(line_load.values.max()) if line_load is not None else None,
        'max_trafo_loading_pct': float(trafo_load.values.max()) if trafo_load is not None else None,
        'n_voltage_violations': n_voltage_violations,
        'n_loading_violations': n_loading_violations,
    }
    return summary, violations


# ============================================================
#  6. OUTPUT FORMATTING
# ============================================================
def _format_output(summary, violations, warnings):
    """Assemble the single output DataFrame (verdict + violations + warnings).

    Long format, one row per record, so a MAS4TE agent can filter by
    `category`:
        - category == 'summary'   -> the verdict (is it viable?)
        - category == 'violation' -> where/when limits were breached
        - category == 'warning'   -> issues during the run (unmatched households...)
    """
    rows = []

    # Verdict lines
    for metric, value in summary.items():
        rows.append({'category': 'summary', 'element': 'network',
                     'metric': metric, 'value': value,
                     'timestep': None, 'detail': None})

    # Violation lines
    for v in violations:
        rows.append({'category': 'violation', 'element': v.get('element'),
                     'metric': v.get('metric'), 'value': v.get('value'),
                     'timestep': v.get('timestep'), 'detail': v.get('detail')})

    # Warning lines
    for w in warnings:
        rows.append({'category': 'warning', 'element': None,
                     'metric': None, 'value': None,
                     'timestep': None, 'detail': w})

    return pd.DataFrame(rows, columns=[
        'category', 'element', 'metric', 'value', 'timestep', 'detail'])


# ============================================================
#  ENTRY POINT (MAS4TE contract)
# ============================================================
def run_digital_twin(config, households, load_profiles,
                     pv_profiles=None, schedules_BSS=None):
    """Run the Bressoux digital twin and return a network verdict.

    Parameters
    ----------
    config : pd.DataFrame        # tunable simulation/validation parameters
    households : pd.DataFrame    # households (profile_id, location, substation...)
    load_profiles : pd.DataFrame # consumption profiles (n_timesteps x n_profiles), MW
    pv_profiles : pd.DataFrame, optional
        Production PV. Colonnes 'pv_{bus_id}' (bus_id entier du reseau), index
        temporel aligne sur load_profiles ('Mon 00:00'..'Sun 23:45'), valeurs
        en MW (MEME unite que load_profiles -- si la production vient de
        pv_production.build_pv_production, qui sort des kW, convertir /1000 en
        amont, cf. main.py). Si fourni : injecte comme sgen, power flow charges
        + PV. Si None : aucun PV (comportement charges seules inchange).
    schedules_BSS : pd.DataFrame, optional   # pas encore supporte (warning if provided)

    Returns
    -------
    out : pd.DataFrame
        Long-format result: summary (verdict) + violations + warnings.
    """
    # config may arrive as a DataFrame -> convert to a dict for convenience
    cfg = _config_to_dict(config)

    warnings = _validate_inputs(cfg, households, load_profiles,
                                pv_profiles, schedules_BSS)

    net, buses, mv_nodes, validation = _build_network(cfg)

    # Separation generation / execution : run_digital_twin ne GENERE plus de
    # profils en interne -- il RECOIT `load_profiles` (produit en amont par la
    # banque mensuelle, cf. sourcecode/consumption_bank.generate_load_profiles)
    # et l'utilise tel quel. Le choix de la methode de generation n'appartient
    # donc plus au twin (methode_affectation retire -- option A). L'assignation
    # (_assign_loads) reste inchangee : load_profiles est un contrat d'interface,
    # peu importe qui l'a rempli.
    load_profiles_for_assignment = load_profiles

    # Canal UNIQUE de mise a l'echelle (stress test) : parametre d'EXECUTION
    # (stress, pas de generation -- sa place est ici), applique sur le
    # load_profiles RECU, avant l'assignation. cf. integrate-bank-B : le x3 de
    # main.py est config['load_scale']=3, qui s'applique desormais sur le
    # load_profiles genere par la banque.
    load_profiles_for_assignment = load_profiles_for_assignment * cfg['load_scale']

    df_p, df_q, bus_to_profile, w_assign = _assign_loads(
        net, buses, mv_nodes, households, load_profiles_for_assignment, cfg)
    warnings += w_assign

    # --- Injection PV (capacite native, pilotee par l'argument pv_profiles) ---
    # pv_profiles fourni : pose un sgen par colonne 'pv_{bus_id}' et lance le
    # power flow charges + PV ensemble. pv_profiles=None : aucun PV, power flow
    # charges seules -- comportement strictement inchange (non-regression).
    df_pv_p = None
    if pv_profiles is not None and len(pv_profiles) > 0:
        # Alignement positionnel : _assign_loads construit df_p via .values
        # (index texte 'Mon 00:00'.. perdu -> RangeIndex 0..N-1). On remet
        # pv_profiles sur ce meme RangeIndex AVANT la verif d'alignement et le
        # power flow (DFData/ConstControl indexent par position).
        df_pv_p = pv_profiles.reset_index(drop=True)
        _verify_temporal_alignment(df_p, df_pv_p)   # garde-fou index (bug deja rencontre)
        _inject_pv_sgens(net, df_pv_p)

    pf = _run_powerflow(net, df_p, df_q, cfg, df_pv_p=df_pv_p)
    vm, line_load, trafo_load = pf['vm'], pf['line_load'], pf['trafo_load']

    summary, violations = _evaluate(vm, line_load, trafo_load, cfg)

    agg_summary, agg_mw = _aggregate_profile_summary(df_p)
    summary.update(agg_summary)

    out = _format_output(summary, violations, warnings)

    # (!) integrate-bank-B : export optionnel du profil agrege complet
    # (672 pas), pour comparaison actuelle/banque_mensuelle -- desactive par
    # defaut pour ne pas alourdir la sortie standard des consommateurs MAS4TE.
    if cfg.get('export_aggregate_profile'):
        profile_rows = pd.DataFrame({
            'category': 'profile_agg', 'element': None, 'metric': 'p_agg_mw',
            'value': agg_mw.values, 'timestep': range(len(agg_mw)), 'detail': None,
        })
        out = pd.concat([out, profile_rows], ignore_index=True)

    return out


def _config_to_dict(config):
    """Accept config as DataFrame or dict; return a plain dict with defaults."""
    defaults = {
        'n_timesteps': STEPS_PER_WEEK,
        'v_min': 0.95,
        'v_max': 1.05,
        'loading_max': 90.0,
        'cos_phi': 0.95,
        'trafo_sn_mva': 0.22,
        'data_path': 'data/',
        # --- integrate-bank-B additions ---
        'seed': RANDOM_SEED,                  # exposed for reproducible comparisons
        # mois_simule / bank_path : parametres de GENERATION (banque mensuelle).
        # run_digital_twin ne genere plus (option A -- methode_affectation retire) ;
        # ils ne sont donc plus lus par le twin lui-meme, mais restent ici comme
        # defauts pour _build_load_profiles_banque_mensuelle, appele en AMONT par
        # sourcecode/consumption_bank.generate_load_profiles (generateur du CSV)
        # et par sourcecode/pv_injection.py (qui construit ses profils via cette
        # meme logique et _config_to_dict).
        'mois_simule': 1,                     # janvier ; mois cible de la banque a la generation
        'bank_path': 'data/' + CONSUMPTION_BANK_FILE,  # banque de vraies semaines -- cf. consumption_bank.py
        'export_aggregate_profile': False,    # opt-in 672-row profile_agg export
        # Canal unique de stress test : parametre d'EXECUTION applique par
        # run_digital_twin sur le load_profiles RECU (remplace l'ancienne
        # multiplication de load_profiles en amont dans main.py).
        'load_scale': 1.0,
    }
    if config is None:
        return defaults
    if isinstance(config, pd.DataFrame):
        # TODO: define the expected DataFrame layout (key/value rows?)
        #       and parse it here. For now, fall back to defaults.
        parsed = {}  # TODO
        return {**defaults, **parsed}
    if isinstance(config, dict):
        return {**defaults, **config}
    return defaults
