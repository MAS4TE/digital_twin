"""
Materialisation de la banque de vraies semaines EAN (approche B validee dans
explore_seasonal_pv.py, branche explore-seasonal-pv) en un fichier parquet
reutilisable, plus la fenetre d'irradiance PVGIS de reference associee.

Ce module ne fait PAS partie du pipeline MAS4TE (digital_twin.run_digital_twin
ne l'importe qu'au moment ou methode_affectation="banque_mensuelle" est
selectionne, via le fichier parquet deja materialise -- pas cette logique de
construction). Executer ce script une fois pour produire :

    data/consumption_bank_real_weeks.parquet
    data/pvgis_irradiance_quarter_hourly.parquet

avant d'utiliser methode_affectation="banque_mensuelle".

Reprend la logique deja validee dans explore_seasonal_pv.py (memes filtres,
memes tags mois/saison), simplement persistee plutot que recalculee a chaque
run.
"""
import os

import numpy as np
import pandas as pd

DATA_PATH = "data/"
RAW_EAN_FILE = "part-00000-3895d355-7ed8-4cff-a072-bdc8805c8651-c000.csv"
BANK_OUT_PATH = "data/consumption_bank_real_weeks.parquet"
IRRADIANCE_OUT_PATH = "data/pvgis_irradiance_quarter_hourly.parquet"

N_PER_WEEK = 7 * 96
NEG_TOLERANCE_KW = -0.01

SEASON_MONTHS = {
    "Hiver": [12, 1, 2], "Printemps": [3, 4, 5],
    "Ete": [6, 7, 8], "Automne": [9, 10, 11],
}
MONTH_TO_SEASON = {m: s for s, months in SEASON_MONTHS.items() for m in months}

LAT, LON = 50.639, 5.605
PVGIS_REFERENCE_YEAR = 2023  # cf. explore_seasonal_pv.py : SARAH3 couvre 2005-2023


def _ean_power_series(df, ean):
    """Serie tz-aware (index = measuredatetime reel, valeurs = kW).

    Rester en Series/DatetimeIndex pandas de bout en bout : passer par
    .values sur un index tz-aware degrade en datetime64 naif et casse
    silencieusement tout reindex ulterieur contre un index tz-aware.
    """
    g = df[df['ean_anon'] == ean].sort_values('measuredatetime').set_index('measuredatetime')
    return (g['measure_cons'].diff() * 4).iloc[1:]


def build_consumption_bank(data_path=DATA_PATH, out_path=BANK_OUT_PATH):
    """Extrait toutes les semaines calendaires reelles completes (672 pas,
    lun-dim) des EAN retenus, taguees par EAN/mois/saison/RUE, et les
    persiste en parquet (format large : colonnes meta + t0000..t0671 en MW).

    La colonne 'street' est necessaire a l'association par rue en aval
    (digital_twin._build_load_profiles_banque_mensuelle) : elle doit suivre
    EXACTEMENT la meme convention de nom que households.csv['source_street']
    pour que le rattachement rue->profil fonctionne. C'est le cas ici : meme
    champ source (addressStreet), meme correction ponctuelle ("Avenue
    Joseph-Merlot" -> "Avenue Joseph Merlot"), meme filtre EAN (total_cons
    >= 365) et donc meme indexation `unique_eans` -- verifie identique a
    celle de profil_conso.ipynb (meme ordre, memes 33 EAN).
    """
    df = pd.read_csv(
        data_path + RAW_EAN_FILE,
        usecols=['ean_anon', 'measuredatetime', 'measure_cons', 'addressStreet'],
    )
    total_cons = df.groupby('ean_anon')['measure_cons'].agg(lambda s: s.max() - s.min())
    keep_eans = total_cons[total_cons >= 365].index
    df = df[df['ean_anon'].isin(keep_eans)].copy()
    df['measuredatetime'] = pd.to_datetime(df['measuredatetime'])
    unique_eans = df['ean_anon'].unique()

    ean_street = df[['ean_anon', 'addressStreet']].drop_duplicates().set_index('ean_anon')['addressStreet']
    ean_street = ean_street.replace({"Avenue Joseph-Merlot": "Avenue Joseph Merlot"})
    street_by_index = [ean_street[unique_eans[i]] for i in range(len(unique_eans))]

    raw_series = {ean: _ean_power_series(df, ean) for ean in unique_eans}

    tmin, tmax = df['measuredatetime'].min(), df['measuredatetime'].max()
    full_range = pd.date_range(tmin.floor('15min'), tmax.ceil('15min'), freq='15min')
    first_monday = tmin.normalize() - pd.Timedelta(days=tmin.weekday())
    if first_monday < tmin:
        first_monday += pd.Timedelta(days=7)
    week_starts = pd.date_range(first_monday, tmax, freq='7D')

    rows = []
    for i, ean in enumerate(unique_eans):
        s = raw_series[ean].reindex(full_range)
        for ws in week_starts:
            i0 = full_range.get_indexer([ws])[0]
            if i0 == -1:
                continue
            i1 = i0 + N_PER_WEEK
            if i1 > len(full_range):
                break
            wv = s.iloc[i0:i1]
            if wv.isna().any() or (wv.values < NEG_TOLERANCE_KW).any():
                continue
            row = {
                "ean_idx": i, "ean_anon": ean, "street": street_by_index[i],
                "month": ws.month, "season": MONTH_TO_SEASON[ws.month],
                "week_start": ws.isoformat(),
            }
            # kW -> MW, coherent avec load_profiles.csv (deja en MW)
            row.update({f"t{k:04d}": v / 1000.0 for k, v in enumerate(wv.values)})
            rows.append(row)

    bank = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bank.to_parquet(out_path, index=False)
    print(f"Banque materialisee : {out_path} ({len(bank)} semaines, {bank['ean_idx'].nunique()} EAN)")
    print(bank.groupby("month").size().rename("n_semaines").to_string())
    return bank


def build_irradiance_reference(out_path=IRRADIANCE_OUT_PATH, year=PVGIS_REFERENCE_YEAR):
    """Irradiance globale PVGIS-SARAH3 (horaire -> quart-horaire par
    forward-fill, cf. explore_seasonal_pv.py pour la justification), persistee
    pour ne pas dependre d'un appel reseau a chaque run d'integration.
    """
    import pvlib

    data, meta = pvlib.iotools.get_pvgis_hourly(LAT, LON, start=year, end=year, raddatabase="PVGIS-SARAH3")
    ghi = (data["poa_direct"] + data["poa_sky_diffuse"] + data["poa_ground_diffuse"]).rename("ghi_w_m2")
    ghi.index = ghi.index.floor("h")

    full_index = pd.date_range(ghi.index[0], ghi.index[-1] + pd.Timedelta(minutes=45),
                                freq="15min", tz=ghi.index.tz)
    ghi_q = ghi.reindex(full_index, method="ffill")

    out = ghi_q.rename("ghi_w_m2").reset_index()
    out.columns = ["timestamp", "ghi_w_m2"]
    out["year"] = year
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Irradiance de reference materialisee : {out_path} ({len(out)} pas quart-horaires, annee {year})")
    return out


if __name__ == "__main__":
    build_consumption_bank()
    build_irradiance_reference()
