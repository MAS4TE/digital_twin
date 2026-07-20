"""
Volet PV -- Brique 1 : generation des metadonnees d'installations PV.

Genere data/pv_installations.csv : quel bus residentiel porte une
installation PV et de quelle taille, plus une installation fixe sur la
maison de repos. Ne fait AUCUNE injection dans pandapower ni calcul de
production -- uniquement les metadonnees, consommees par les briques
suivantes (production, injection).

Ne modifie pas digital_twin.py : reutilise _build_network() et
_config_to_dict() en LECTURE SEULE pour obtenir la topologie reelle
(bus_id <-> rue), garantissant que les bus_id generes ici correspondent
exactement a ceux du reseau reel (pas de reimplementation risquant un ecart
avec le vrai bus_id pandapower).

Prerequis verifie avant tout code (cf. brief) : le lien bus -> rue existe-t-il
et est-il exploitable pour les 333 bus residentiels ?
  - households.csv : NON exploitable pour ce besoin -- ne connait que 3 rues
    (Avenue Joseph Merlot / Rue Ernest Malvoz / Plateau du Bouhay), un
    decoupage simplifie utilise pour l'assignation de profils de conso, sans
    rapport avec la topologie physique complete.
  - colonne 'street' de consumption_bank_real_weeks.parquet : NON exploitable
    non plus -- limitee aux rues des 33 EAN mesures (meme sous-ensemble que
    ci-dessus, cf. sourcecode/consumption_bank.py).
  - _build_network()['buses'] (digital_twin.py) : EXPLOITABLE et faisant
    autorite -- dict (street, num) -> bus_id, verifie couvrant exactement
    les 333 bus residentiels reels sur 7 vraies rues (Avenue Joseph Merlot
    106, Rue Ernest Malvoz 83, Plateau du Bouhay 54, Rue des Primeroses 30,
    Rue des Bluets 21, Rue Simon Burnet 20, Rue Joseph Lambremont 19).
    C'est la source de verite car c'est litteralement ce que la
    reconstruction du reseau utilise (derive de coordonnées.csv).
Aucun repli aleatoire uniforme necessaire : le lien exact existe.

Calibrage terrain : un retour utilisateur (visite de site) indique des PV
observes plus souvent Rue Ernest Malvoz que sur Plateau du Bouhay. Reflete
via `street_weights` (poids de tirage par rue, defaut 1.0 partout sauf Rue
Ernest Malvoz = 3.0) -- un biais probabiliste, pas une garantie deterministe
(cf. DEFAULTS et generate_pv_metadata).
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from sourcecode.digital_twin import _build_network, _config_to_dict

OUT_PATH = "data/pv_installations.csv"

DEFAULTS = {
    "n_installations": 30,
    "kwp_residential": 3.0,
    "kwp_rest_home": 12.0,
    "seed": 42,
    "n_streets_range": (2, 4),     # nombre de rues "grappe" equipees, tire dans cet intervalle
    "min_street_size": 5,          # exclut du tirage les rues trop petites (garde-fou ;
                                    # jamais declenche avec les 7 rues reelles, 19-106 bus chacune)
    # Poids relatif de tirage par rue (defaut 1.0 pour toute rue absente de
    # ce dict). Calibrage QUALITATIF sur un retour terrain (visite de site,
    # utilisateur) : PV observes plus souvent Rue Ernest Malvoz que sur
    # Plateau du Bouhay -- pas une statistique precise (aucun comptage
    # exhaustif fourni), juste une tendance a refleter. Poids=3.0 rend cette
    # rue ~3x plus susceptible d'etre tiree qu'une rue neutre, sans jamais la
    # forcer (le tirage reste probabiliste, seed pilote) -- ajuster ici si
    # une meilleure estimation (comptage reel, etc.) devient disponible.
    "street_weights": {"Rue Ernest Malvoz": 3.0},
    "rest_home_name": "Paradis du Bouhay",  # cf. digital_twin.MV_PASSAGE
    "out_path": OUT_PATH,
}


def _allocate_across_streets(sizes, total, rng):
    """Repartit `total` installations entre des rues de tailles `sizes`, au
    prorata de leur taille, sans jamais depasser la taille d'une rue (evite
    l'absurdite "3 installations sur une rue de 2 maisons"). Le reste
    (arrondi) est distribue 1 par 1, dans un ordre permute aleatoirement,
    aux rues qui ont encore de la place.
    """
    sizes = np.asarray(sizes, dtype=float)
    if total > sizes.sum():
        raise ValueError(
            f"n_installations={int(total)} depasse la capacite totale des rues "
            f"choisies ({int(sizes.sum())} bus) -- augmenter n_streets_range."
        )
    weights = sizes / sizes.sum()
    alloc = np.minimum(np.floor(weights * total).astype(int), sizes.astype(int))
    remaining = int(total - alloc.sum())
    order = rng.permutation(len(sizes))
    guard = 0
    while remaining > 0:
        g = order[guard % len(order)]
        if alloc[g] < sizes[g]:
            alloc[g] += 1
            remaining -= 1
        guard += 1
        if guard > 100 * len(sizes) + 1000:
            raise RuntimeError(
                "Repartition des installations par rue : boucle anormale, "
                "capacite insuffisante malgre la verification initiale."
            )
    return alloc


def generate_pv_metadata(config=None):
    """Genere data/pv_installations.csv (ou cfg['out_path']).

    Placement residentiel : GROUPE PAR RUE (grappes), pas uniforme -- tire
    `n_streets` rues (nombre tire dans n_streets_range, seed pilote), puis
    repartit n_installations entre elles au prorata de leur taille reelle
    (cf. _allocate_across_streets), avant de tirer, DANS chaque rue choisie,
    les bus precis a equiper (sans remise, seed pilote). Intention (cf.
    brief) : quelques rues fortement equipees plutot qu'une dispersion
    homogene -- effet voisinage/installateur, stress reseau localise sur un
    meme depart BT.

    Si les rues tirees en premier n'ont pas assez de capacite cumulee pour
    n_installations, la selection s'etend a d'autres rues eligibles (ordre
    permute) jusqu'a couvrir la demande -- jamais declenche avec les 7 rues
    reelles et les valeurs par defaut, mais evite un crash silencieux pour
    des parametres extremes.

    La maison de repos (`rest_home_name`) est ajoutee en dur, HORS du tirage
    residentiel, avec kwp_rest_home et type='rest_home'. Son bus_id est
    celui de son bus MV de passage existant (le meme bus que sa charge de
    consommation dans _assign_loads) -- pas de rue associee (colonne 'street'
    vide), puisqu'elle n'appartient pas au systeme de rues residentielles.

    Parameters (cfg, tous optionnels -- defauts dans DEFAULTS)
    ----------
    n_installations : int    -- nombre d'installations residentielles (30)
    kwp_residential : float  -- taille residentielle fixe [kWp] (3.0)
    kwp_rest_home : float    -- taille maison de repos [kWp] (12.0)
    seed : int               -- reproductibilite du tirage (42)

    Returns
    -------
    pd.DataFrame : le contenu ecrit dans le CSV (bus_id, kwp, type, street).
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    net_cfg = _config_to_dict(None)  # topologie fixe, independante des parametres PV
    net, buses, mv_nodes, validation = _build_network(net_cfg)

    buses_by_street = defaultdict(list)
    for (street, num), b in buses.items():
        buses_by_street[street].append(b)

    rng = np.random.default_rng(cfg["seed"])

    eligible_streets = [s for s, bs in buses_by_street.items() if len(bs) >= cfg["min_street_size"]]
    lo, hi = cfg["n_streets_range"]
    n_streets = int(rng.integers(lo, hi + 1))
    n_streets = min(n_streets, len(eligible_streets))

    # Ordre de tirage PONDERE (sans remise) plutot qu'une permutation
    # uniforme : les rues avec un poids > 1.0 (cf. street_weights) ont plus
    # de chances d'apparaitre tot dans l'ordre, donc d'etre choisies comme
    # rue "grappe" -- sans jamais garantir leur presence (reste probabiliste).
    street_weights = cfg.get("street_weights") or {}
    weights = np.array([street_weights.get(s, 1.0) for s in eligible_streets], dtype=float)
    probs = weights / weights.sum()
    shuffled = rng.choice(eligible_streets, size=len(eligible_streets), replace=False, p=probs).tolist()

    chosen_streets = shuffled[:n_streets]
    idx_next = n_streets
    while (sum(len(buses_by_street[s]) for s in chosen_streets) < cfg["n_installations"]
           and idx_next < len(shuffled)):
        chosen_streets.append(shuffled[idx_next])
        idx_next += 1

    sizes = [len(buses_by_street[s]) for s in chosen_streets]
    alloc = _allocate_across_streets(sizes, cfg["n_installations"], rng)

    rows = []
    for street, n_take in zip(chosen_streets, alloc):
        if n_take == 0:
            continue
        chosen_buses = rng.choice(buses_by_street[street], size=int(n_take), replace=False)
        for b in chosen_buses:
            rows.append({"bus_id": int(b), "kwp": cfg["kwp_residential"],
                         "type": "residential", "street": street})

    rest_home_bus = mv_nodes[cfg["rest_home_name"]]
    rows.append({"bus_id": int(rest_home_bus), "kwp": cfg["kwp_rest_home"],
                 "type": "rest_home", "street": None})

    out = pd.DataFrame(rows, columns=["bus_id", "kwp", "type", "street"])
    os.makedirs(os.path.dirname(cfg["out_path"]), exist_ok=True)
    out.to_csv(cfg["out_path"], index=False)
    return out


def extend_pv_metadata(base_installations, n_additional, config=None, out_path=None):
    """Volet PV -- Brique 4 : étend un parc PV existant de `n_additional`
    installations résidentielles supplémentaires, en PRÉSERVANT les
    installations existantes À L'IDENTIQUE (mêmes bus_id) -- PAS un tirage
    indépendant. Nécessaire pour comparer des taux de pénétration (30 vs 50)
    sans mélanger effet de taux et effet de placement (cf. brief brique 4).

    Logique (même esprit "grappes par rue" que generate_pv_metadata) :
    étend EN PRIORITÉ les rues déjà équipées (au prorata de leur capacité
    RESTANTE, pas de leur taille totale), pour renforcer les clusters
    existants plutôt que d'en semer de nouveaux. Si leur capacité résiduelle
    ne suffit pas, pioche de nouvelles rues via le même mécanisme pondéré
    (`street_weights`) que generate_pv_metadata -- jamais déclenché avec les
    valeurs par défaut (30->50 sur 2 rues à 106+30 bus, largement assez de
    marge), mais garde-fou pour des extensions plus importantes (80, 100).

    Parameters
    ----------
    base_installations : pd.DataFrame -- parc existant (ex. pv_installations.csv)
    n_additional : int                -- nombre d'installations residentielles a ajouter
    config : dict, optionnel          -- memes cles que DEFAULTS (seed, street_weights, ...)
    out_path : str, optionnel         -- si fourni, ecrit le CSV etendu

    Returns
    -------
    pd.DataFrame : parc complet (existant + nouvelles + maison de repos), meme schema.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    net_cfg = _config_to_dict(None)
    net, buses, mv_nodes, validation = _build_network(net_cfg)
    buses_by_street = defaultdict(list)
    for (street, num), b in buses.items():
        buses_by_street[street].append(b)

    base_res = base_installations[base_installations["type"] == "residential"].copy()
    rest_home_rows = base_installations[base_installations["type"] == "rest_home"].copy()
    existing_bus_ids = set(int(b) for b in base_res["bus_id"])
    existing_streets = [s for s in base_res["street"].dropna().unique()]

    rng = np.random.default_rng(cfg["seed"])

    # 1) Etendre en priorite les rues deja equipees, au prorata de leur
    #    capacite RESTANTE (bus pas encore equipes).
    remaining_in_existing = {
        s: [b for b in buses_by_street[s] if int(b) not in existing_bus_ids]
        for s in existing_streets
    }
    total_remaining_existing = sum(len(v) for v in remaining_in_existing.values())

    n_from_existing = min(n_additional, total_remaining_existing)
    new_rows = []
    if n_from_existing > 0:
        sizes = [len(remaining_in_existing[s]) for s in existing_streets]
        alloc = _allocate_across_streets(sizes, n_from_existing, rng)
        for street, n_take in zip(existing_streets, alloc):
            if n_take == 0:
                continue
            chosen = rng.choice(remaining_in_existing[street], size=int(n_take), replace=False)
            for b in chosen:
                new_rows.append({"bus_id": int(b), "kwp": cfg["kwp_residential"],
                                 "type": "residential", "street": street})

    # 2) Si insuffisant (pas declenche avec 30->50/80/100 sur les rues
    #    reelles), pioche de nouvelles rues via le meme mecanisme pondere.
    still_needed = n_additional - n_from_existing
    if still_needed > 0:
        eligible_streets = [s for s, bs in buses_by_street.items()
                            if len(bs) >= cfg["min_street_size"] and s not in existing_streets]
        street_weights = cfg.get("street_weights") or {}
        weights = np.array([street_weights.get(s, 1.0) for s in eligible_streets], dtype=float)
        probs = weights / weights.sum()
        shuffled = rng.choice(eligible_streets, size=len(eligible_streets), replace=False, p=probs).tolist()

        chosen_new_streets = []
        idx_next = 0
        while (sum(len(buses_by_street[s]) for s in chosen_new_streets) < still_needed
               and idx_next < len(shuffled)):
            chosen_new_streets.append(shuffled[idx_next])
            idx_next += 1

        sizes = [len(buses_by_street[s]) for s in chosen_new_streets]
        alloc = _allocate_across_streets(sizes, still_needed, rng)
        for street, n_take in zip(chosen_new_streets, alloc):
            if n_take == 0:
                continue
            chosen = rng.choice(buses_by_street[street], size=int(n_take), replace=False)
            for b in chosen:
                new_rows.append({"bus_id": int(b), "kwp": cfg["kwp_residential"],
                                 "type": "residential", "street": street})

    new_res = pd.DataFrame(new_rows, columns=["bus_id", "kwp", "type", "street"])
    out = pd.concat([base_res, new_res, rest_home_rows], ignore_index=True)[
        ["bus_id", "kwp", "type", "street"]
    ]

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    df = generate_pv_metadata()

    n_res = int((df["type"] == "residential").sum())
    n_rest = int((df["type"] == "rest_home").sum())
    print(f"{OUT_PATH} genere : {len(df)} lignes "
          f"({n_res} residentielles + {n_rest} maison de repos)")

    print("\nRepartition par rue (residentiel) :")
    print(df[df["type"] == "residential"].groupby("street").size()
          .sort_values(ascending=False).to_string())

    kwp_res = df.loc[df["type"] == "residential", "kwp"].sum()
    kwp_rest = df.loc[df["type"] == "rest_home", "kwp"].sum()
    print(f"\nPuissance PV totale installee : {df['kwp'].sum():.1f} kWp "
          f"(residentiel : {kwp_res:.1f} kWp, maison de repos : {kwp_rest:.1f} kWp)")

    print("\nReproductibilite (2 appels, meme seed) :", end=" ")
    df2 = generate_pv_metadata()
    print("OK, CSV identique" if df.equals(df2) else "ECHEC, CSV different !")
