"""
Figure de présentation des profils employés dans le réseau.
Deux panneaux empilés, une journée type (96 pas quart-horaires, 24 h),
pas de double axe.

  Panneau 1 : 4 charges MV (maison de repos, centre horticole, hall
              omnisport, école)
  Panneau 2 : 3 habitations (EAN idx 3, 9, 16)

Style cohérent avec les autres figures du projet (serif, base 24,
pas de spines top/right). Sauvegarde PNG + PDF.

À LANCER DANS pandapower_env (pandas + matplotlib déjà présents).

Script de présentation, hors pipeline de production : ne modifie ni
n'importe sourcecode/digital_twin.py, lit uniquement les fichiers de
données déjà matérialisés.

>>> Points de raccordement résolus (vérifiés contre le schéma réel) :
    1. load_habitation_day() -> data/consumption_bank_real_weeks.parquet
       Schéma réel : colonnes meta (ean_idx, ean_anon, street, month [int
       1-12], season, week_start) + 672 colonnes t0000..t0671 (PAS
       step_0..step_671, pas de colonne 'values'). Valeurs en MW (comme
       load_profiles.csv) -> converties en kW ici (x1000).
       HAB_IDX désigne des `ean_idx` (identifiants EAN réels, 0-32,
       plusieurs semaines par EAN dans la banque) et NON des positions de
       ligne (iloc).

       Selection des 3 habitations : PLUS de choix manuel d'idx (une version
       precedente pointait idx=3, choisi pour son pic record de 13.9 kW --
       un "geant" non representatif). Desormais _select_percentile_households()
       choisit automatiquement 3 EAN dont l'energie hebdo mediane (calculee
       sur TOUTES leurs semaines disponibles, toute l'annee) est la plus
       proche des percentiles 25/50/75 de la population, en excluant les
       menages dont le ratio pic/moyenne MEDIAN dessine un comportement
       atypique recurrent.

       Seuil d'exclusion calibre sur la distribution reelle, pas au hasard :
       un seuil naif "pic < 3-4x moyenne" (suggestion initiale) exclurait
       97% des semaines (mediane reelle du ratio ~10x sur les 1346 semaines
       de la banque -- les compteurs residentiels individuels ont un facteur
       de charge intrinsequement bas). Le ratio MEDIAN par EAN (annuel) va
       de 1.8x (idx=28, quasi-vacant) a 21.0x (idx=3) ; PEAK_TO_MEAN_MAX=15.0
       n'exclut que idx=3 (21.0x), idx=4 (16.2x) et idx=29 (15.0x), qui se
       detachent nettement du reste de la population (<=13.9x). Choix valide
       avec l'utilisateur avant implementation (seuil 3-4x juge infaisable,
       "exclusion par ratio typique du menage" retenue).

       Alignement temporel : _get_common_week_rows() cherche, PARMI LES
       SEMAINES OU LES 3 ean_idx DE HAB_IDX ONT TOUS DES DONNEES, celle qui
       maximise le MINIMUM des pics quotidiens des 2 premiers idx (haut et
       moyen) -- garantit que les profils compares proviennent bien de la
       MEME semaine ET du MEME jour calendaires reels, et que les deux
       niveaux les plus significatifs montrent tous les deux une activite
       visible ce jour-la (maximiser seulement le pic du 1er idx laissait
       parfois le 2eme quasi plat ce jour precis). Pas de moyenne
       inter-semaines (ecraserait les pics).
    2. load_mv_day() -> data/mv_passage_profiles.csv
       Localisé dans sourcecode/digital_twin.py (MV_PASSAGE_PROFILES_FILE,
       MV_PASSAGE). C'est un fichier CSV dédié, séparé de la banque
       résidentielle : 672 lignes ('Mon 00:00'..'Sun 23:45'), colonnes =
       les 4 noms réels ('Paradis du Bouhay', 'Plantations', 'Omnisport
       BX', 'Groupe Scolaire'), un SEUL profil hebdomadaire fixe (pas de
       segmentation par mois -- contrairement aux habitations, ces 4
       facilities n'ont qu'une version, utilisée toute l'année dans le
       pipeline actuel). Valeurs en MW -> converties en kW ici (x1000).
       Les 4 profils sont bien temporels (672 valeurs distinctes chacun,
       pas de profil constant) : rien à signaler de ce côté.
"""

import calendar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sûr (savefig fiable, cf. soucis machine)
import matplotlib.pyplot as plt

# --- Style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 24,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
})

FIGSIZE = (12, 11)
N_STEPS_DAY = 96               # 24 h * 4 (quart-horaire)
DT_H = 0.25
MW_TO_KW = 1000.0

# --- Paramètres -----------------------------------------------------------
BANK_PATH = "data/consumption_bank_real_weeks.parquet"
MV_PROFILES_PATH = "data/mv_passage_profiles.csv"
MOIS = "juin"                 # mois d'affichage (n'affecte pas la selection
                               # des habitations, qui se base sur l'annee
                               # entiere -- cf. _select_percentile_households)
# Pas de JOUR_INDEX fixe : le jour de la semaine affiche est determine
# dynamiquement (cf. _get_common_week_rows).

PEAK_TO_MEAN_MAX = 15.0       # exclut les menages dont le ratio pic/moyenne
                              # MEDIAN (toutes semaines, toute l'annee)
                              # depasse ce seuil -- cf. en-tete pour la
                              # calibration sur la distribution reelle
HOUSEHOLD_PERCENTILES = (25, 50, 75)  # bas / median / haut

MV_NAMES = ["Paradis du Bouhay", "Plantations", "Omnisport BX", "Groupe Scolaire"]
# noms reels des colonnes de data/mv_passage_profiles.csv (cf.
# digital_twin.MV_PASSAGE) ; correspondent respectivement a maison de
# repos, centre horticole, hall omnisport, ecole.
MV_LABELS = {
    "Paradis du Bouhay": "Paradis du Bouhay (rest home)",
    "Plantations": "Plantations (horticultural center)",
    "Omnisport BX": "Omnisport BX (sports hall)",
    "Groupe Scolaire": "Groupe Scolaire (school)",
}

COLORS_HAB = ["#6DB3D9", "#F5A623", "#DC143C"]
COLORS_MV = ["#444444", "#2E7D5B", "#7B5EA7", "#B0711F"]

_MONTH_NAMES_FR = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}


def _month_number(mois):
    return _MONTH_NAMES_FR[str(mois).strip().lower()]


def _select_percentile_households(ratio_threshold=PEAK_TO_MEAN_MAX,
                                   percentiles=HOUSEHOLD_PERCENTILES):
    """Selectionne 3 EAN representatifs (bas/median/haut) plutot que de
    pointer des idx a la main -- cf. en-tete pour la justification complete
    et la calibration du seuil.

    Pour chaque EAN, calcule sur TOUTES ses semaines disponibles (annee
    entiere, independant du mois d'affichage MOIS) :
      - median_energy : energie hebdo mediane [kWh] (taille typique du menage)
      - median_ratio  : ratio pic/moyenne median [-] (typicalite du profil)
    Exclut les EAN dont median_ratio > ratio_threshold (comportement
    atypique recurrent), puis choisit, pour chaque percentile demande,
    l'EAN eligible dont median_energy est la plus proche de ce percentile
    (calcule parmi les seuls EAN eligibles).

    Returns
    -------
    list[int] : 3 ean_idx, dans l'ordre de `percentiles` (bas -> haut).
    """
    df = pd.read_parquet(BANK_PATH)
    value_cols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]
    vals = df[value_cols].to_numpy(dtype=float) * MW_TO_KW

    stats = df[["ean_idx"]].copy()
    with np.errstate(invalid="ignore"):  # semaine(s) a moyenne nulle (compteur mort) -> NaN, ignore par median()
        stats["peak_to_mean"] = vals.max(axis=1) / vals.mean(axis=1)
    stats["energy_kwh"] = vals.sum(axis=1) * DT_H

    per_ean = stats.groupby("ean_idx").agg(
        median_ratio=("peak_to_mean", "median"),
        median_energy=("energy_kwh", "median"),
    )
    eligible = per_ean[per_ean["median_ratio"] <= ratio_threshold].copy()
    if eligible.empty:
        raise ValueError(f"Aucun menage eligible sous le seuil pic/moyenne={ratio_threshold}.")

    targets = eligible["median_energy"].quantile([p / 100 for p in percentiles])
    chosen = []
    for target in targets:
        remaining = eligible.drop(index=chosen, errors="ignore")
        closest = int((remaining["median_energy"] - target).abs().idxmin())
        chosen.append(closest)

    excluded = sorted(set(per_ean.index) - set(eligible.index))
    print(f"Habitations selectionnees (percentiles {percentiles}, seuil pic/moyenne <= {ratio_threshold}x) :")
    for p, idx in zip(percentiles, chosen):
        print(f"  p{p:<3} -> ean_idx={idx:<3} energie mediane={per_ean.loc[idx, 'median_energy']:.1f} kWh/sem, "
              f"ratio pic/moyenne median={per_ean.loc[idx, 'median_ratio']:.2f}x")
    print(f"  menages exclus (ratio > {ratio_threshold}x) : {excluded}")

    return chosen


HAB_IDX = _select_percentile_households()


# --- LECTURE DES DONNÉES --------------------------------------------------
def _day_slice(week_values, day_index):
    """Extrait la journée `day_index` (96 pts) d'un profil hebdo de 672 pts."""
    week = np.asarray(week_values, dtype=float)
    start = day_index * N_STEPS_DAY
    return week[start:start + N_STEPS_DAY]


_COMMON_WEEK_CACHE = {}


def _get_common_week_rows(idxs, mois):
    """Trouve, PARMI LES SEMAINES OU TOUS LES `idxs` ONT DES DONNEES pour le
    mois donne, la (semaine, jour) qui maximise le MINIMUM des pics
    quotidiens de idxs[0] (haut) ET idxs[1] (moyen, si present) -- garantit
    que les profils compares proviennent bien de la MEME semaine calendaire
    ET que les deux niveaux les plus significatifs montrent tous les deux un
    pic visible ce jour-la.

    Ne maximiser QUE le pic de idxs[0] ne suffit pas : sur certaines
    semaines communes, idx[0] culmine fort mais idxs[1] reste quasi plat ce
    jour-la (ex. semaine du 30/06 : idx=3 pic a 12.4 kW le lundi, mais
    idx=30 n'y monte qu'a ~2 kW, alors qu'il atteint 5.1 kW un autre lundi
    disponible, celui du 25/08 -- retenu par ce critere).

    Returns
    -------
    rows : dict[idx -> pd.Series]  (une ligne de banque par idx, meme semaine)
    best_week : str                (week_start ISO de la semaine retenue)
    best_day : int                 (0=lundi..6=dimanche)
    """
    key = (tuple(idxs), mois)
    if key in _COMMON_WEEK_CACHE:
        return _COMMON_WEEK_CACHE[key]

    df = pd.read_parquet(BANK_PATH)
    month_num = _month_number(mois)
    sub = df[df["month"] == month_num]
    value_cols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]

    weeks_by_idx = {i: set(sub.loc[sub["ean_idx"] == i, "week_start"]) for i in idxs}
    common_weeks = sorted(set.intersection(*weeks_by_idx.values()))
    if not common_weeks:
        raise ValueError(
            f"Aucune semaine calendaire commune aux ean_idx {idxs} pour le mois {mois!r}."
        )

    ref_idxs = idxs[:2] if len(idxs) > 1 else idxs
    best_week, best_day, best_score = None, None, -np.inf
    for wk in common_weeks:
        per_day_by_ref = []
        for i in ref_idxs:
            row = sub[(sub["ean_idx"] == i) & (sub["week_start"] == wk)].iloc[0]
            week_vals = row[value_cols].to_numpy(dtype=float)
            per_day_by_ref.append(week_vals.reshape(7, N_STEPS_DAY).max(axis=1))
        combined = np.minimum.reduce(per_day_by_ref)
        day = int(np.argmax(combined))
        if combined[day] > best_score:
            best_week, best_day, best_score = wk, day, combined[day]

    rows = {i: sub[(sub["ean_idx"] == i) & (sub["week_start"] == best_week)].iloc[0] for i in idxs}
    result = (rows, best_week, best_day)
    _COMMON_WEEK_CACHE[key] = result
    return result


def load_habitation_day(idx, mois):
    """Renvoie 96 valeurs (kW) -- une vraie journee de l'habitation `idx`
    (ean_idx), tiree de la semaine calendaire commune a tout HAB_IDX, le
    jour ou le niveau "haut" atteint son pic (cf. _get_common_week_rows)."""
    rows, _, best_day = _get_common_week_rows(HAB_IDX, mois)
    row = rows[idx]
    value_cols = [c for c in row.index if c.startswith("t") and c[1:].isdigit()]
    week_mw = row[value_cols].to_numpy(dtype=float)
    return _day_slice(week_mw, best_day) * MW_TO_KW


def load_mv_day(mv_name, mois):
    """Renvoie 96 valeurs (kW) -- la journee du profil MV `mv_name`, le meme
    jour de la semaine que le panneau habitations (pour une "journee type"
    coherente entre les deux panneaux).

    Les 4 profils MV sont un fichier dedie (data/mv_passage_profiles.csv),
    un seul profil hebdomadaire fixe (pas de segmentation par mois) : le
    parametre `mois` ne sert qu'a determiner, via HAB_IDX, quel jour de la
    semaine aligner sur le panneau habitations.
    """
    _, _, best_day = _get_common_week_rows(HAB_IDX, mois)
    mv = pd.read_csv(MV_PROFILES_PATH, index_col=0)
    week_mw = mv[mv_name].to_numpy(dtype=float)
    return _day_slice(week_mw, best_day) * MW_TO_KW


def week_energy_kwh_hab(idx, mois):
    """Énergie hebdo (kWh) d'une habitation, pour annotation légende (meme
    semaine commune que load_habitation_day)."""
    rows, _, _ = _get_common_week_rows(HAB_IDX, mois)
    row = rows[idx]
    value_cols = [c for c in row.index if c.startswith("t") and c[1:].isdigit()]
    week_mw = row[value_cols].to_numpy(dtype=float)
    return float(np.sum(week_mw) * MW_TO_KW * DT_H)


# --- FIGURE ---------------------------------------------------------------
def build_figure():
    t = np.arange(N_STEPS_DAY) * DT_H
    fig, (ax_mv, ax_hab) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # Panneau 1 : 4 charges MV
    for name, c in zip(MV_NAMES, COLORS_MV):
        y = load_mv_day(name, MOIS)
        ax_mv.plot(t, y, color=c, lw=2.0, label=MV_LABELS.get(name, name))
    ax_mv.set_ylabel("Power [kW]")
    ax_mv.set_title("MV loads", loc="left", fontsize=22)
    ax_mv.set_ylim(bottom=0)
    ax_mv.legend(frameon=True, fontsize=15)

    # Panneau 2 : 3 habitations (percentiles bas/median/haut, cf.
    # _select_percentile_households)
    for idx, pct, c in zip(HAB_IDX, HOUSEHOLD_PERCENTILES, COLORS_HAB):
        y = load_habitation_day(idx, MOIS)
        e = week_energy_kwh_hab(idx, MOIS)
        ax_hab.plot(t, y, color=c, lw=2.0,
                    label=f"Household {idx} (p{pct}) — {e:.0f} kWh/week")
    ax_hab.set_ylabel("Power [kW]")
    ax_hab.set_xlabel("Hour of day")
    ax_hab.set_title("Households (25th/50th/75th percentile)", loc="left", fontsize=22)
    ax_hab.set_ylim(bottom=0)
    ax_hab.legend(frameon=True, fontsize=15)

    ax_hab.set_xlim(0, 24)
    ax_hab.set_xticks(np.arange(0, 25, 4))
    ax_hab.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 4)])

    _, best_week, best_day = _get_common_week_rows(HAB_IDX, MOIS)
    week_label = pd.Timestamp(best_week).strftime("%d/%m/%Y")
    day_label_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][best_day]
    month_label_en = calendar.month_name[_month_number(MOIS)]
    fig.suptitle(f"Profiles used — typical {day_label_en} ({month_label_en}, week of {week_label})", fontsize=24)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = build_figure()
    for ext in ("png", "pdf"):
        fig.savefig(f"scripts/viz/profils_reseau.{ext}", dpi=200, bbox_inches="tight")
    print("Figures écrites : scripts/viz/profils_reseau.png / .pdf")
