"""
Volet PV -- Brique 2 : conversion irradiance -> production PV.

Calcul PUR (aucun réseau, aucun pandapower, aucun create_sgen) : convertit la
série d'irradiance PVGIS (data/pvgis_irradiance_quarter_hourly.parquet) en
profils de production par installation PV (data/pv_installations.csv,
brique 1), pour un mois donné. Livrable : build_pv_production().

Prérequis vérifié (cohérence temporelle irradiance <-> conso) :
  - Résolution : irradiance bien quart-horaire (35040 lignes = 365 jours x
    96 pas/jour, aucune agrégation horaire résiduelle) -- alignable pas à
    pas sur les profils de conso (672 pas/semaine, même convention d'index
    "Mon 00:00".."Sun 23:45").
  - Sélection par mois : même paramètre mois_simule que la banque mensuelle
    (nom de mois français ou numéro 1-12). L'irradiance (année de référence
    2023, PVGIS-SARAH3) et la banque de conso (semaines réelles
    Nov2024-Oct2025) n'ont PAS la même année calendaire -- la cohérence
    recherchée est au niveau du MOIS (saisonnalité), pas de la semaine
    calendaire exacte (impossible, années différentes), cf. brief.

Modèle de conversion (v1) : LINÉAIRE, isolé dans irradiance_to_power(),
remplaçable plus tard par un modèle pvlib complet (température, orientation,
onduleur) sans toucher au reste de cette brique.

    puissance_kW = irradiance_W_m2 / 1000 * kWp * system_factor

`system_factor` (défaut 0.8) encapsule, EN UN SEUL NOMBRE pour la v1, à la
fois le rendement système réel (onduleur ~97%, câblage ~98%, salissure/
mismatch ~95%, soit ~0.90 combiné) ET le fait qu'on utilise l'irradiance
GLOBALE HORIZONTALE (GHI, tilt=0 -- cf. sourcecode/consumption_bank.py)
plutôt que l'irradiance dans le plan des panneaux (POA, tilt ~30-35° typique
en Belgique) : un panneau incliné collecte relativement MOINS à midi en été
qu'une surface horizontale (soleil quasi au zénith) -- effet qu'un simple
facteur constant ne peut pas capturer. Conséquence mesurée (cf. brique 2,
vérifications) : avec system_factor=0.8, l'énergie journalière de juin
ressort un peu AU-DESSUS de la fourchette indicative du brief (8-12 kWh/jour
pour 3 kWp) -- environ 16 kWh/jour -- même ordre de grandeur (pas 10x), mais
hors fourchette stricte. La baisser (~0.5) collerait mieux à l'énergie mais
ferait chuter le pic sous ~1.5 kW au lieu de ~3 kW visé pour "plein soleil".
C'est une tension inhérente à l'usage de GHI plutôt que POA, pas un bug --
`system_factor` reste LE point d'ajustement exposé, et toute la conversion
est isolée pour permettre un vrai calcul POA (pvlib) plus tard sans
réécriture.
"""
import os

import numpy as np
import pandas as pd

IRRADIANCE_PATH = "data/pvgis_irradiance_quarter_hourly.parquet"
PV_INSTALLATIONS_PATH = "data/pv_installations.csv"

STEPS_PER_DAY = 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY
DEFAULT_SYSTEM_FACTOR = 0.8

_MONTH_NAMES_FR = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}
_WEEK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_WEEK_TIME_INDEX = [
    f"{_WEEK_DAYS[step // STEPS_PER_DAY]} "
    f"{(step % STEPS_PER_DAY) * 15 // 60:02d}:{(step % STEPS_PER_DAY) * 15 % 60:02d}"
    for step in range(STEPS_PER_WEEK)
]


def _month_number(mois):
    """Accepte un numero (1-12) ou un nom de mois francais ; renvoie 1-12."""
    if isinstance(mois, (int, np.integer)):
        m = int(mois)
    else:
        key = str(mois).strip().lower()
        m = _MONTH_NAMES_FR.get(key)
        if m is None:
            try:
                m = int(key)
            except ValueError:
                raise ValueError(
                    f"mois ({mois!r}) non reconnu (attendu un entier 1-12 ou un nom de mois francais)."
                )
    if not (1 <= m <= 12):
        raise ValueError(f"mois ({mois!r}) doit correspondre a un mois entre 1 et 12.")
    return m


def _select_representative_week(irradiance, mois):
    """Semaine calendaire reelle (672 pas quart-horaires) representative du
    mois demande : le premier lundi 00:00 dont le mois correspond a `mois`
    ET qui dispose d'une semaine complete de donnees a la suite (peut
    deborder sur le mois suivant, comme les semaines reelles de la banque de
    conso -- pas de moyenne, une vraie semaine).

    Returns
    -------
    irr_week : np.ndarray (672,)  -- irradiance [W/m2]
    monday : pd.Timestamp         -- debut de la semaine retenue
    """
    month_num = _month_number(mois)
    ts = pd.to_datetime(irradiance["timestamp"])
    s = pd.Series(irradiance["ghi_w_m2"].to_numpy(dtype=float), index=ts).sort_index()

    mondays = s.index[(s.index.weekday == 0) & (s.index.hour == 0)
                      & (s.index.minute == 0) & (s.index.month == month_num)]
    for monday in mondays:
        week = s.loc[monday: monday + pd.Timedelta(minutes=15 * (STEPS_PER_WEEK - 1))]
        if len(week) == STEPS_PER_WEEK:
            return week.to_numpy(dtype=float), monday
    raise ValueError(
        f"Aucune semaine complete de {STEPS_PER_WEEK} pas trouvee pour le mois {month_num} "
        f"dans l'irradiance -- verifier la couverture de {IRRADIANCE_PATH}."
    )


def irradiance_to_power(irradiance_w_m2, kwp, system_factor=DEFAULT_SYSTEM_FACTOR):
    """Modele lineaire v1 : puissance [kW] = irradiance [W/m2] / 1000 * kWp *
    system_factor. Isolee ici pour etre remplacable par un modele pvlib
    complet (temperature, orientation, onduleur) sans toucher au reste de la
    brique -- cf. en-tete pour la justification du facteur par defaut.

    Clippe l'irradiance a 0 en entree (defensif : garantit une production
    strictement nulle et jamais negative la nuit, meme si l'irradiance
    source contenait un residu numerique negatif).
    """
    irr = np.maximum(np.asarray(irradiance_w_m2, dtype=float), 0.0)
    return irr / 1000.0 * kwp * system_factor


def build_pv_production(mois, pv_installations, irradiance, system_factor=DEFAULT_SYSTEM_FACTOR):
    """Construit les profils de production PV (kW) de toutes les
    installations de `pv_installations` (résidentielles + maison de repos),
    pour la semaine représentative du mois `mois`.

    Parameters
    ----------
    mois : str | int              -- mois cible (nom francais ou 1-12)
    pv_installations : pd.DataFrame -- data/pv_installations.csv (bus_id, kwp, type, street)
    irradiance : pd.DataFrame       -- data/pvgis_irradiance_quarter_hourly.parquet (timestamp, ghi_w_m2, year)
    system_factor : float           -- cf. irradiance_to_power

    Returns
    -------
    pd.DataFrame : index 'Mon 00:00'..'Sun 23:45' (672 pas, MEME format que
    load_profiles), colonnes 'pv_{bus_id}' (kW), une par installation.
    """
    irr_week, monday = _select_representative_week(irradiance, mois)

    data = {}
    for _, row in pv_installations.iterrows():
        col = f"pv_{int(row['bus_id'])}"
        data[col] = irradiance_to_power(irr_week, row["kwp"], system_factor)

    out = pd.DataFrame(data, index=_WEEK_TIME_INDEX)
    out.index.name = "week_step"
    out.attrs["source_week_start"] = monday.isoformat()
    return out


if __name__ == "__main__":
    pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
    irradiance = pd.read_parquet(IRRADIANCE_PATH)

    for mois in ("juin", "janvier"):
        prod = build_pv_production(mois, pv_installations, irradiance)
        print(f"{mois}: semaine representative debutant {prod.attrs['source_week_start']}, "
              f"shape={prod.shape}")
