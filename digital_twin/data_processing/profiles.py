"""
Consumption profile generation: materialize the bank of real EAN weeks into a
reusable parquet file, plus the associated reference PVGIS irradiance window,
then draw a monthly `households` (consumption timeseries) from that bank.

This is upstream DATA PREPARATION, not part of the execution pipeline:
run_digital_twin never reads the bank parquet directly, and does not generate
`households` itself -- it receives it, produced here, and uses it as-is
(generation / execution split). Run this module once to produce:

    data/bressoux/consumption_bank_real_weeks.parquet
    data/bressoux/pvgis_irradiance_quarter_hourly.parquet

before calling generate_load_profiles.
"""
import os

import numpy as np
import pandas as pd

from ..utils import _WEEK_TIME_INDEX, _config_to_dict, _normalize_month

# Scenario this module reads from and fills by default -- see
# digital_twin.data_processing.process_data for the same convention:
# <data_path>/inputs/ holds TRUE inputs (raw EAN CSV, coordinates, ...);
# <data_path>/out/processed/ holds DERIVED data (bank, irradiance,
# metadata.csv, load_profiles.csv -- the files run_digital_twin consumes).
SCENARIO_DATA_PATH = "data/bressoux/"

RAW_DATA_PATH = SCENARIO_DATA_PATH
RAW_EAN_FILE = "part-00000-3895d355-7ed8-4cff-a072-bdc8805c8651-c000.csv"
BANK_OUT_PATH = SCENARIO_DATA_PATH + "out/processed/consumption_bank_real_weeks.parquet"
IRRADIANCE_OUT_PATH = SCENARIO_DATA_PATH + "out/processed/pvgis_irradiance_quarter_hourly.parquet"

N_PER_WEEK = 7 * 96
NEG_TOLERANCE_KW = -0.01

SEASON_MONTHS = {
    "Hiver": [12, 1, 2], "Printemps": [3, 4, 5],
    "Ete": [6, 7, 8], "Automne": [9, 10, 11],
}
MONTH_TO_SEASON = {m: s for s, months in SEASON_MONTHS.items() for m in months}

LAT, LON = 50.639, 5.605
PVGIS_REFERENCE_YEAR = 2023  # SARAH3 covers 2005-2023

BANK_JITTER_STEPS = [-2, -1, 1, 2]  # +/- 15-30 min, applied only to duplicate draws


def _ean_power_series(df, ean):
    """tz-aware series (index = real measuredatetime, values = kW).

    Stay on a pandas Series/DatetimeIndex end to end: going through .values on a
    tz-aware index degrades it to naive datetime64 and silently breaks any later
    reindex against a tz-aware index.
    """
    g = df[df['ean_anon'] == ean].sort_values('measuredatetime').set_index('measuredatetime')
    return (g['measure_cons'].diff() * 4).iloc[1:]


def build_consumption_bank(data_path=RAW_DATA_PATH, out_path=BANK_OUT_PATH):
    """Extract all complete real calendar weeks (672 steps, Mon-Sun) of the kept
    EANs, tagged by EAN/month/season/STREET, and persist them to parquet (wide
    format: meta columns + t0000..t0671 in MW).

    The 'street' column is needed for the downstream per-street association
    (_build_load_profiles_banque_mensuelle below): it must follow EXACTLY the
    same naming convention as metadata.csv['source_street'] for the
    street->profile mapping to work. That holds here: same source field
    (addressStreet), same one-off correction ("Avenue Joseph-Merlot" -> "Avenue
    Joseph Merlot"), same EAN filter (total_cons >= 365) and therefore the same
    `unique_eans` indexing.
    """
    df = pd.read_csv(
        os.path.join(data_path, "inputs", RAW_EAN_FILE),
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


# ============================================================
#  Agnostic household profile generation: the 33 real EAN weekly profiles,
#  drawn (or, when more diversity is needed than exists, shift/scale
#  augmented) into a bank of any target size. This is the mechanism behind
#  the historical profil_conso.ipynb notebook, generalized to be agnostic to
#  how many real profiles are actually available.
# ============================================================
SHIFT_MAX_STEPS = 8        # load diversity: time shift +/- 8 steps (+/- 2h)
SCALE_SIGMA = 0.2          # load amplitude dispersion (~20%), lognormal

# The MAS4TE households.csv schema (profil_conso.ipynb): (dtype, default).
# Booleans/sizes default to an "off" state (False / 0.0); count/category
# fields with no meaningful "false" default to NA (unknown for now).
MAS4TE_METADATA_COLUMNS = {
    'constellation':          ('Int64', pd.NA),
    'building_type':          ('string', pd.NA),
    'heat_pump':              ('boolean', False),
    'electric_water_heating': ('boolean', False),
    'electric_vehicle_type':  ('string', pd.NA),
    'n_persons':              ('Int64', pd.NA),
    'pv_system_size_kwp':     ('Float64', 0.0),
    'battery_size_kwh':       ('Float64', 0.0),
}


def extract_real_weekly_profiles(data_path=RAW_DATA_PATH, raw_file=RAW_EAN_FILE):
    """Average each retained EAN's readings into a single 672-step weekly
    profile (kW), keyed by (day-of-week, time-of-day) -- port of
    profil_conso.ipynb's `weekly_profile`. Same EAN filter (total_cons >= 365)
    and street convention as build_consumption_bank, so `street_by_index`
    aligns with metadata['source_street'].

    Unlike build_consumption_bank (every complete calendar week, tagged by
    month), this AVERAGES over all available weeks per EAN into ONE
    representative profile -- the basis for generate_household_profiles below,
    independent of any month/season conditioning.

    Returns
    -------
    weekly_bank : np.ndarray (n_real, 672)  -- average active power [kW]
    street_by_index : list[str]
    ean_by_index : list[str]
    """
    df = pd.read_csv(
        os.path.join(data_path, "inputs", raw_file),
        usecols=['ean_anon', 'measuredatetime', 'measure_cons', 'addressStreet'],
    )
    total_cons = df.groupby('ean_anon')['measure_cons'].agg(lambda s: s.max() - s.min())
    keep_eans = total_cons[total_cons >= 365].index
    df = df[df['ean_anon'].isin(keep_eans)].copy()
    df['measuredatetime'] = pd.to_datetime(df['measuredatetime'])
    unique_eans = df['ean_anon'].unique()

    ean_street = df[['ean_anon', 'addressStreet']].drop_duplicates().set_index('ean_anon')['addressStreet']
    ean_street = ean_street.replace({"Avenue Joseph-Merlot": "Avenue Joseph Merlot"})
    street_by_index = [ean_street[ean] for ean in unique_eans]

    full_week_index = pd.MultiIndex.from_product(
        [range(7), range(0, 1440, 15)], names=['dow', 'tod'],
    )

    profiles = []
    for ean in unique_eans:
        g = df[df['ean_anon'] == ean].sort_values('measuredatetime').copy()
        g['power_cons'] = g['measure_cons'].diff() * 4    # kWh/15min cumulative -> kW
        g = g.iloc[1:]
        dow = g['measuredatetime'].dt.dayofweek
        tod = g['measuredatetime'].dt.hour * 60 + g['measuredatetime'].dt.minute
        prof = g.assign(dow=dow, tod=tod).groupby(['dow', 'tod'])['power_cons'].mean()
        prof = prof.reindex(full_week_index)
        profiles.append(prof.values)

    weekly_bank = np.vstack(profiles)

    def _fill_circular(row):
        # Residual NaN (weekly slot never observed on a truncated EAN):
        # circular linear interpolation so the power flow has a value
        # everywhere (Monday 00:00 follows Sunday 23:45).
        s = pd.Series(row)
        if s.isna().all():
            return np.zeros_like(row)
        return s.interpolate(limit_direction='both').values

    weekly_bank = np.vstack([_fill_circular(r) for r in weekly_bank])
    return weekly_bank, street_by_index, list(unique_eans)


def generate_household_profiles(n_profiles, data_path=RAW_DATA_PATH, raw_file=RAW_EAN_FILE, seed=42):
    """Generate `n_profiles` household profiles from the real EAN weekly
    profiles, AGNOSTIC to how many real profiles are actually available:

    - n_profiles <= n_real: draw WITHOUT replacement -- there is already
      enough real diversity, profiles are used as-is (no perturbation).
    - n_profiles > n_real: draw WITH replacement, each draw shifted
      (+/- SHIFT_MAX_STEPS steps) and scaled (lognormal, sigma=SCALE_SIGMA) so
      duplicated households don't look identical -- the historical
      profil_conso.ipynb `make_profile` mechanism.

    Returns
    -------
    households : pd.DataFrame  # index 'Mon 00:00'..'Sun 23:45', columns
                                # profile_{i}, values in MW
    metadata : pd.DataFrame    # profile_id, source_street, source_ean_index
                                # (traceability) + the MAS4TE household schema
                                # (defaulted, see MAS4TE_METADATA_COLUMNS)
    """
    weekly_bank, street_by_index, ean_by_index = extract_real_weekly_profiles(data_path, raw_file)
    n_real = len(weekly_bank)
    rng = np.random.default_rng(seed)

    if n_profiles <= n_real:
        picks = rng.choice(n_real, size=n_profiles, replace=False)
        profiles_kw = weekly_bank[picks]
    else:
        picks = rng.integers(0, n_real, size=n_profiles)
        profiles_kw = np.empty((n_profiles, weekly_bank.shape[1]))
        for i, src in enumerate(picks):
            shift = rng.integers(-SHIFT_MAX_STEPS, SHIFT_MAX_STEPS + 1)
            scale = rng.lognormal(mean=0, sigma=SCALE_SIGMA)
            profiles_kw[i] = np.roll(weekly_bank[src], shift) * scale

    households = pd.DataFrame(
        (profiles_kw / 1000.0).T,      # kW -> MW, shape (672, n_profiles)
        index=_WEEK_TIME_INDEX,
        columns=[f"profile_{i}" for i in range(n_profiles)],
    )
    households.index.name = "week_step"

    metadata = pd.DataFrame({
        'profile_id': range(n_profiles),
        **{col: pd.array([default] * n_profiles, dtype=dtype)
           for col, (dtype, default) in MAS4TE_METADATA_COLUMNS.items()},
        'source_street': [street_by_index[i] for i in picks],
        'source_ean_index': [int(i) for i in picks],
    })

    return households, metadata


def _load_consumption_bank(bank_path):
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            f"Banque de semaines reelles introuvable ({bank_path}). Executer "
            "`python -m digital_twin.data_processing.profiles` pour la materialiser "
            "avant de generer households via la banque mensuelle "
            "(generate_load_profiles)."
        )
    return pd.read_parquet(bank_path)


def _build_load_profiles_banque_mensuelle(metadata, config):
    """Fill a `households` timeseries in the load_profiles.csv format (index
    'Mon 00:00'..'Sun 23:45', columns profile_{id}, values in MW), but whose
    CONTENT comes from real EAN weeks of month `config['mois_simule']` (see
    build_consumption_bank above).

    For each profile_id in `metadata`, draw (with replacement, seed=
    config['seed']) one real week from the sub-bank of the SAME street (bank
    'street' column, aligned on metadata['source_street']; same source field
    and name correction, see build_consumption_bank). Global fallback (all streets)
    when a street has no week that month. Anti-duplicate micro-jitter (+/-1-2
    steps, 15-30 min) on rows drawn more than once. No shift/scale otherwise.

    Returns
    -------
    households : pd.DataFrame   # same shape as load_profiles.csv
    warnings : list[str]
    """
    warnings = []
    month = _normalize_month(config['mois_simule'])
    bank = _load_consumption_bank(config['bank_path'])
    if 'street' not in bank.columns:
        raise ValueError(
            f"{config['bank_path']} ne contient pas de colonne 'street' -- "
            "regenerer la banque avec `python -m digital_twin.data_processing.profiles` "
            "(version a jour, qui l'ajoute)."
        )

    pool_month = bank[bank['month'] == month].reset_index(drop=True)
    if len(pool_month) == 0:
        raise ValueError(f"Aucune semaine reelle disponible pour le mois {month} dans {config['bank_path']}.")

    value_cols = [c for c in bank.columns if c.startswith('t') and c[1:].isdigit()]
    # positions (in pool_month, shared indexing) of each street's weeks
    pool_positions_by_street = {
        street: idx.to_numpy() for street, idx in pool_month.groupby('street').groups.items()
    }
    all_positions = pool_month.index.to_numpy()

    rng = np.random.default_rng(config['seed'])

    street_by_profile = metadata.drop_duplicates('profile_id').set_index('profile_id')['source_street']
    profile_ids = sorted(metadata['profile_id'].unique())

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
        values = pool_month.loc[j, value_cols].to_numpy(dtype=float)  # already in MW

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

    households = pd.DataFrame(data, index=_WEEK_TIME_INDEX)
    households.index.name = "week_step"
    return households, warnings


def generate_load_profiles(mois_simule, metadata=None, config=None,
                           out_path=None, data_path=SCENARIO_DATA_PATH):
    """Upstream generator (generation / execution split): produces
    `<data_path>/out/processed/load_profiles.csv` from the bank of real EAN
    weeks of month `mois_simule`, in the exact format run_digital_twin's
    `households` argument expects (index 'Mon 00:00'..'Sun 23:45', columns
    profile_{id}, values in MW).

    run_digital_twin receives this CSV (as `households`) and uses it as-is.

    Parameters
    ----------
    mois_simule : str | int   -- target month of the bank (French name or 1-12)
    metadata : pd.DataFrame, optional -- metadata.csv (read from
                                  <data_path>/out/processed/ if None)
    config : dict, optional   -- overrides (seed, bank_path, ...); mois_simule is
                                  forced by the argument of the same name
    out_path : str, optional  -- path of the generated CSV (defaults to
                                  <data_path>/out/processed/load_profiles.csv)

    Returns
    -------
    (households : pd.DataFrame, warnings : list[str])
    """
    processed_dir = os.path.join(data_path, "out", "processed")
    if metadata is None:
        metadata = pd.read_csv(os.path.join(processed_dir, "metadata.csv"))
    if out_path is None:
        out_path = os.path.join(processed_dir, "load_profiles.csv")

    cfg = _config_to_dict(config)
    cfg['mois_simule'] = mois_simule

    households, warnings = _build_load_profiles_banque_mensuelle(metadata, cfg)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    households.to_csv(out_path)

    # Round-trip: the written CSV must be read back IDENTICALLY by process_data,
    # which reads via pd.read_csv(out_path, index_col=0). Replay exactly that
    # read and compare (index, columns, values) -- a guardrail against any
    # format drift (separator, encoding, precision) between write and read.
    reread = pd.read_csv(out_path, index_col=0)
    same = (
        list(reread.index) == list(households.index)
        and list(reread.columns) == list(households.columns)
        and np.allclose(reread.to_numpy(dtype=float),
                        households.to_numpy(dtype=float))
    )
    if not same:
        raise RuntimeError(
            f"Aller-retour {out_path} echoue : le CSV relu differe du DataFrame "
            "genere (index/colonnes/valeurs)."
        )

    return households, warnings


if __name__ == "__main__":
    build_consumption_bank()
    build_irradiance_reference()
