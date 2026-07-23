"""
Materialize the bank of real EAN weeks into a reusable parquet file, plus the
associated reference PVGIS irradiance window.

Building the bank (build_consumption_bank) is NOT part of the execution
pipeline: run_digital_twin never reads this parquet directly. The upstream
generator generate_load_profiles (below) uses it to write data/load_profiles.csv,
which run_digital_twin then reads as-is (generation / execution split). Run this
script once to produce:

    data/consumption_bank_real_weeks.parquet
    data/pvgis_irradiance_quarter_hourly.parquet

before calling generate_load_profiles.
"""
import os

import numpy as np
import pandas as pd

DATA_PATH = "data/"
RAW_EAN_FILE = "part-00000-3895d355-7ed8-4cff-a072-bdc8805c8651-c000.csv"
BANK_OUT_PATH = "data/consumption_bank_real_weeks.parquet"
IRRADIANCE_OUT_PATH = "data/pvgis_irradiance_quarter_hourly.parquet"
LOAD_PROFILES_OUT_PATH = "data/load_profiles.csv"

N_PER_WEEK = 7 * 96
NEG_TOLERANCE_KW = -0.01

SEASON_MONTHS = {
    "Hiver": [12, 1, 2], "Printemps": [3, 4, 5],
    "Ete": [6, 7, 8], "Automne": [9, 10, 11],
}
MONTH_TO_SEASON = {m: s for s, months in SEASON_MONTHS.items() for m in months}

LAT, LON = 50.639, 5.605
PVGIS_REFERENCE_YEAR = 2023  # SARAH3 covers 2005-2023


def _ean_power_series(df, ean):
    """tz-aware series (index = real measuredatetime, values = kW).

    Stay on a pandas Series/DatetimeIndex end to end: going through .values on a
    tz-aware index degrades it to naive datetime64 and silently breaks any later
    reindex against a tz-aware index.
    """
    g = df[df['ean_anon'] == ean].sort_values('measuredatetime').set_index('measuredatetime')
    return (g['measure_cons'].diff() * 4).iloc[1:]


def build_consumption_bank(data_path=DATA_PATH, out_path=BANK_OUT_PATH):
    """Extract all complete real calendar weeks (672 steps, Mon-Sun) of the kept
    EANs, tagged by EAN/month/season/STREET, and persist them to parquet (wide
    format: meta columns + t0000..t0671 in MW).

    The 'street' column is needed for the downstream per-street association
    (digital_twin._build_load_profiles_banque_mensuelle): it must follow EXACTLY
    the same naming convention as households.csv['source_street'] for the
    street->profile mapping to work. That holds here: same source field
    (addressStreet), same one-off correction ("Avenue Joseph-Merlot" -> "Avenue
    Joseph Merlot"), same EAN filter (total_cons >= 365) and therefore the same
    `unique_eans` indexing.
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
            # kW -> MW, consistent with load_profiles.csv (already in MW)
            row.update({f"t{k:04d}": v / 1000.0 for k, v in enumerate(wv.values)})
            rows.append(row)

    bank = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bank.to_parquet(out_path, index=False)
    print(f"Banque materialisee : {out_path} ({len(bank)} semaines, {bank['ean_idx'].nunique()} EAN)")
    print(bank.groupby("month").size().rename("n_semaines").to_string())
    return bank


def build_irradiance_reference(out_path=IRRADIANCE_OUT_PATH, year=PVGIS_REFERENCE_YEAR):
    """Global PVGIS-SARAH3 irradiance (hourly -> quarter-hourly by forward-fill),
    persisted so runs do not depend on a network call each time.
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


def generate_load_profiles(mois_simule, households=None, config=None,
                           out_path=LOAD_PROFILES_OUT_PATH, data_path=DATA_PATH):
    """Upstream generator (generation / execution split): produces
    data/load_profiles.csv from the bank of real EAN weeks of month
    `mois_simule`, in the exact format run_digital_twin reads (index
    'Mon 00:00'..'Sun 23:45', columns profile_{id}, values in MW).

    run_digital_twin receives this CSV and uses it as-is. This function wraps
    _build_load_profiles_banque_mensuelle (kept in digital_twin.py, also reused
    by pv_injection.py) and materializes its output on disk.

    Parameters
    ----------
    mois_simule : str | int   -- target month of the bank (French name or 1-12)
    households : pd.DataFrame, optional -- households.csv (read from data_path if None)
    config : dict, optional   -- overrides (seed, bank_path, ...); mois_simule is
                                  forced by the argument of the same name
    out_path : str            -- path of the generated CSV (data/load_profiles.csv)

    Returns
    -------
    (load_profiles : pd.DataFrame, warnings : list[str])
    """
    # Local import to avoid any import cycle (digital_twin does not import
    # consumption_bank; the import is kept close to its use).
    # _build_load_profiles_banque_mensuelle is the single source of the draw
    # logic (one shared implementation).
    from digital_twin.digital_twin import (
        _build_load_profiles_banque_mensuelle, _config_to_dict,
    )

    if households is None:
        households = pd.read_csv(data_path + "households.csv")

    cfg = _config_to_dict(config)
    cfg['mois_simule'] = mois_simule

    load_profiles, warnings = _build_load_profiles_banque_mensuelle(households, cfg)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    load_profiles.to_csv(out_path)

    # Round-trip: the written CSV must be read back IDENTICALLY by main.py, which
    # reads via pd.read_csv(out_path, index_col=0). Replay exactly that read and
    # compare (index, columns, values) -- a guardrail against any format drift
    # (separator, encoding, precision) between write and read.
    reread = pd.read_csv(out_path, index_col=0)
    same = (
        list(reread.index) == list(load_profiles.index)
        and list(reread.columns) == list(load_profiles.columns)
        and np.allclose(reread.to_numpy(dtype=float),
                        load_profiles.to_numpy(dtype=float))
    )
    if not same:
        raise RuntimeError(
            f"Aller-retour {out_path} echoue : le CSV relu differe du DataFrame "
            "genere (index/colonnes/valeurs)."
        )

    return load_profiles, warnings


if __name__ == "__main__":
    build_consumption_bank()
    build_irradiance_reference()
