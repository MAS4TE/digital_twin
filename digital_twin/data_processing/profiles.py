"""
Consumption profile generation: materialize the bank of real EAN weeks into a
reusable parquet file, plus the associated reference PVGIS irradiance window,
then draw a monthly `households` (consumption timeseries) from that bank.

This is upstream DATA PREPARATION, not part of the execution pipeline:
run_digital_twin never reads the bank parquet directly, and does not generate
`households` itself -- it receives it, produced here, and uses it as-is
(generation / execution split). Run this module once to produce, under the
scenario's own <data_path>/out/processed/:

    consumption_bank_real_weeks.parquet
    pvgis_irradiance_quarter_hourly.parquet

before calling generate_load_profiles.

SEASONAL COHERENCE. `mois_simule` drives BOTH the month a profile is drawn from
here and the PV production month (data_processing.pv), so charge and sun stay on
the same season by construction rather than by two settings kept in sync by hand.
The month each profile actually came from is written into metadata.csv
(`source_month`), which is what lets digital_twin._validate_inputs warn when
load_profiles.csv was built for a different month than the one being simulated.

PORTABILITY. Two layers with very different reach live here:
  * the RAW INGESTION (build_consumption_bank) is Bressoux's own data
    preparation: it expects that site's smart-meter CSV schema (ean_anon /
    measuredatetime / measure_cons / addressStreet) and carries its street-name
    fix. Another site replaces this one function -- or, as MAS4TE agents will,
    supplies its `households` directly.
  * everything downstream is site-agnostic. The BANK PARQUET is the portable
    seam: a scenario that can produce a frame with `street`, `month` and
    t0000..t0671 columns (kW) gets generate_household_metadata and
    generate_load_profiles for free, whatever filled it -- without ever owning
    a raw meter file.
"""
import os

import numpy as np
import pandas as pd

from ..utils import (_WEEK_TIME_INDEX, _config_to_dict, _normalize_month,
                     bank_path_for, irradiance_path_for, load_site_network,
                     site_centroid)

# Scenario this module reads from and fills by default -- see
# digital_twin.data_processing.process_data for the same convention:
# <data_path>/inputs/ holds TRUE inputs (raw EAN CSV, coordinates, ...);
# <data_path>/out/processed/ holds DERIVED data (bank, irradiance,
# metadata.csv, load_profiles.csv -- the files run_digital_twin consumes).
SCENARIO_DATA_PATH = "data/bressoux/"

RAW_DATA_PATH = SCENARIO_DATA_PATH
RAW_EAN_FILE = "part-00000-3895d355-7ed8-4cff-a072-bdc8805c8651-c000.csv"
# No BANK_OUT_PATH / IRRADIANCE_OUT_PATH literals: both paths derive from the
# scenario's data_path through utils.bank_path_for / utils.irradiance_path_for,
# the single source shared with the consumer side. Two independent literals let
# the generator write where the consumer never looks.

N_PER_WEEK = 7 * 96
NEG_TOLERANCE_KW = -0.01

SEASON_MONTHS = {
    "Hiver": [12, 1, 2], "Printemps": [3, 4, 5],
    "Ete": [6, 7, 8], "Automne": [9, 10, 11],
}
MONTH_TO_SEASON = {m: s for s, months in SEASON_MONTHS.items() for m in months}

# No LAT/LON constant: where to fetch the irradiance is DERIVED from the site's
# own network (utils.site_centroid), see build_irradiance_reference. A fixed pair
# here is the one portability defect that raises nothing at all -- another site
# would quietly get Bressoux's sun.
PVGIS_REFERENCE_YEAR = 2023  # SARAH3 covers 2005-2023

# Anti-duplicate jitter, applied only to duplicate draws: when the same real week
# comes out several times, each EXTRA occurrence is rolled by a DISTINCT offset,
# so no two profiles can be bit-identical.
#
# Ordered by increasing magnitude (+1, -1, +2, -2, ...): a week drawn twice moves
# by 15 min, and only a heavily reused one reaches the far end. A deterministic
# ladder rather than a random pick from a short list, which lets two occurrences
# collide and come out strictly identical.
BANK_JITTER_MAX_STEPS = 8   # +/- 2 h at the far end of the ladder
BANK_JITTER_LADDER = [step * sign
                      for step in range(1, BANK_JITTER_MAX_STEPS + 1)
                      for sign in (1, -1)]


def _ean_power_series(df, ean):
    """tz-aware series (index = real measuredatetime, values = kW).

    Stay on a pandas Series/DatetimeIndex end to end: going through .values on a
    tz-aware index degrades it to naive datetime64 and silently breaks any later
    reindex against a tz-aware index.
    """
    g = df[df['ean_anon'] == ean].sort_values('measuredatetime').set_index('measuredatetime')
    return (g['measure_cons'].diff() * 4).iloc[1:]


def build_consumption_bank(data_path=RAW_DATA_PATH, out_path=None):
    """Extract all complete real calendar weeks (672 steps, Mon-Sun) of the kept
    EANs, tagged by EAN/month/season/STREET, and persist them to parquet (wide
    format: meta columns + t0000..t0671 in kW).

    `out_path` defaults to utils.bank_path_for(data_path) -- the SAME path the
    consumer resolves from config['bank_path'], so generating the bank here
    always makes it findable by _build_load_profiles_banque_mensuelle.

    The 'street' column is TRACEABILITY (which street a drawn week comes from),
    not a draw constraint: _build_load_profiles_banque_mensuelle draws from the
    whole month, all streets together -- see the "why global and not per-street"
    note there. It follows the same naming convention as
    metadata.csv['source_street'] (same source field addressStreet, same one-off
    correction "Avenue Joseph-Merlot" -> "Avenue Joseph Merlot", same EAN filter
    total_cons >= 365, hence the same `unique_eans` indexing), so the two stay
    comparable for a per-street analysis.
    """
    if out_path is None:
        out_path = bank_path_for(data_path)

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
            # A week with no variation at all (all-zero, or a constant power) is a
            # meter artifact rather than a household: no consumption shape to draw
            # from, and -- decisively -- INVARIANT under the anti-duplicate roll,
            # so every copy of it comes out identical whatever offset is applied
            # (see BANK_JITTER_LADDER).
            if wv.values.max() - wv.values.min() < 1e-12:
                continue
            row = {
                "ean_idx": i, "ean_anon": ean, "street": street_by_index[i],
                "month": ws.month, "season": MONTH_TO_SEASON[ws.month],
                "week_start": ws.isoformat(),
            }
            # kW, consistent with load_profiles.csv (also kW): no conversion.
            row.update({f"t{k:04d}": v for k, v in enumerate(wv.values)})
            rows.append(row)

    bank = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bank.to_parquet(out_path, index=False)
    print(f"Bank materialized: {out_path} ({len(bank)} weeks, {bank['ean_idx'].nunique()} EANs)")
    print(bank.groupby("month").size().rename("n_weeks").to_string())
    return bank


def build_irradiance_reference(config=None, out_path=None, year=PVGIS_REFERENCE_YEAR,
                               lat=None, lon=None):
    """Global PVGIS-SARAH3 irradiance (hourly -> quarter-hourly by forward-fill),
    persisted so runs do not depend on a network call each time.

    WHERE the irradiance is fetched is DERIVED from the site's own network
    (utils.site_centroid, the centre of its LV buses) rather than from a fixed
    pair of coordinates. A hard-coded LAT/LON is the one portability defect that
    produces no error at all: another site would quietly receive Bressoux's sun.
    `lat`/`lon` override the centroid explicitly.

    Parameters
    ----------
    config : dict or None  -- scenario config; its data_path locates the site's
                              network (for the centroid) and the output file
    out_path : str or None -- defaults to utils.irradiance_path_for(data_path)
    year : int             -- PVGIS reference year
    lat, lon : float or None -- explicit override of the derived centroid
    """
    import pvlib

    cfg = _config_to_dict(config)
    if out_path is None:
        out_path = irradiance_path_for(cfg['data_path'])
    if lat is None or lon is None:
        lat, lon = site_centroid(load_site_network(cfg))

    print(f"PVGIS irradiance for ({lat:.4f}, {lon:.4f}), year {year}")
    data, meta = pvlib.iotools.get_pvgis_hourly(lat, lon, start=year, end=year, raddatabase="PVGIS-SARAH3")
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
    print(f"Reference irradiance materialized: {out_path} ({len(out)} quarter-hourly steps, year {year})")
    return out


# ============================================================
#  Household metadata: the MAS4TE household schema, one row per profile.
#
#  ORDER MATTERS. Metadata is generated FIRST (it only needs to know how many
#  profiles), then generate_load_profiles draws the weeks and fills the source_*
#  columns from the weeks it actually drew. `source_street` is what
#  digital_twin._assign_loads matches against bus streets, so it has to describe
#  THE WEEK THAT WAS DRAWN -- filling it from an independent draw leaves metadata
#  describing a profile that exists nowhere.
# ============================================================

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

# Traceability of the drawn week, filled by generate_load_profiles. `source_month`
# is the load-bearing one: it is what makes the seasonal coherence between
# load_profiles.csv and config['mois_simule'] checkable at run time
# (digital_twin._validate_inputs), instead of a silent mismatch between
# January loads and June sun.
SOURCE_COLUMNS = {
    'source_street':     ('string', pd.NA),
    'source_month':      ('Int64', pd.NA),
    'source_week_start': ('string', pd.NA),
    'source_ean_index':  ('Int64', pd.NA),
}

METADATA_FILE = "metadata.csv"
LOAD_PROFILES_FILE = "load_profiles.csv"


def generate_household_metadata(network):
    """Skeleton `metadata`: ONE household per LV charge bus of `network`.

    `bus_id` is filled here, 1:1 with `profile_id` and in ascending bus order.
    That column IS the household -> bus assignment: declaring it upstream makes
    it visible, reproducible, and readable by the other bricks -- the BSS
    generator in particular schedules each battery against the load of its own
    bus rather than a pool mean.

    Exhaustive by construction -- every LV bus is a row -- which is what lets
    _validate_inputs check it by SET against the network rather than by count.

    Deliberately reads no meter data: no raw CSV, no bank. A site supplying its
    own bank parquet (the portable seam, see the module docstring) can build a
    full scenario without ever owning Bressoux's raw EAN file. The source_*
    columns are filled by generate_load_profiles, the only function that knows
    which real week each profile got.

    Parameters
    ----------
    network : dict  -- the bundle from utils.load_site_network

    Returns
    -------
    metadata : pd.DataFrame  # profile_id, bus_id + MAS4TE schema + empty source_*
    """
    bus_ids = sorted(int(b) for b in network['buses'].values())
    n = len(bus_ids)
    if n == 0:
        raise ValueError(
            "network['buses'] is empty: no LV bus to place a household on."
        )
    return pd.DataFrame({
        'profile_id': range(n),
        'bus_id': bus_ids,
        **{col: pd.array([default] * n, dtype=dtype)
           for col, (dtype, default) in MAS4TE_METADATA_COLUMNS.items()},
        **{col: pd.array([default] * n, dtype=dtype)
           for col, (dtype, default) in SOURCE_COLUMNS.items()},
    })


def _cyclic_draw(n_draws, pool, rng):
    """Draw `n_draws` items from `pool`, EXHAUSTING it before reusing anything.

    Permute the pool, walk it in order, re-permute when it runs out. Compared to
    `n_draws` independent draws with replacement, this is strictly better and
    costs nothing:

      * every element is used at least once as soon as n_draws >= len(pool)
        (with replacement, the birthday effect leaves some unused);
      * reuse is UNIFORM at floor/ceil(n_draws / len(pool)) instead of a long
        tail, which keeps the anti-duplicate jitter at its smallest offsets.

    When the pool is LARGER than n_draws, a random subset is taken and nothing
    is reused -- the correct behaviour, not a special case.

    Returns
    -------
    list : `n_draws` pool elements
    """
    pool = np.asarray(pool)
    if len(pool) == 0:
        raise ValueError("Empty pool: nothing to draw from.")
    out = []
    while len(out) < n_draws:
        perm = rng.permutation(pool)
        out.extend(perm[:n_draws - len(out)].tolist())
    return out


def _load_consumption_bank(bank_path):
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            f"Bank of real weeks not found ({bank_path}). Run "
            "`python -m digital_twin.data_processing.profiles` to materialize it "
            "before generating households from the monthly bank "
            "(generate_load_profiles)."
        )
    return pd.read_parquet(bank_path)


def _build_load_profiles_banque_mensuelle(metadata, config):
    """Fill a `households` timeseries in the load_profiles.csv format (index
    'Mon 00:00'..'Sun 23:45', columns profile_{id}, values in kW), but whose
    CONTENT comes from real EAN weeks of month `config['mois_simule']` (see
    build_consumption_bank above).

    For each profile_id in `metadata` -- hence for each LV BUS, the two being
    1:1 -- draw one real week from the WHOLE month sub-bank, all streets
    together, CYCLICALLY (see _cyclic_draw: the pool is exhausted before
    anything is reused, seed=config['seed']). A week drawn more than once has
    each extra occurrence rolled by a distinct offset (BANK_JITTER_LADDER), so
    no two profiles come out identical. No shift/scale otherwise: amplitude is
    left untouched.

    ONE DRAW, NOT TWO. This is the only place a bus's consumption is decided.
    A second draw downstream would undo the work done here: the jitter separates
    profiles from each other, but a profile copied onto several buses reaches
    them all unchanged.

    WHY GLOBAL AND NOT PER-STREET. Restricting the draw to the profile's own
    street costs a lot and buys nothing on the Bressoux bank:
      * cost -- only 3 of the 7 network streets carry a metered EAN, and two of
        them are thin (6 and 3 EANs). In June, 83 Rue Ernest Malvoz buses would
        share 8 candidate weeks (10.4x reuse) against a global monthly pool of
        133. A per-street draw also guarantees the duplicates are ELECTRICAL
        NEIGHBOURS, so their peaks add up on the very same LV feeder.
      * benefit -- street barely predicts consumption here: dispersion of the
        per-EAN weekly energy is ~7 kWh BETWEEN streets against ~21 kWh WITHIN a
        street, and the mean peak hours (14.1 / 15.0 / 15.5 h) are indistinguishable
        given ~5 h standard deviations.
    Drawing globally keeps reuse at ~2.5x, spread at random over the network
    instead of concentrated on one feeder -- which is what the +/-1-2 step jitter
    can actually decorrelate. The bank's 'street' column is traceability only.

    Returns
    -------
    households : pd.DataFrame   # same shape as load_profiles.csv
    warnings : list[str]
    """
    warnings = []
    month = _normalize_month(config['mois_simule'])
    bank = _load_consumption_bank(config['bank_path'])

    pool_month = bank[bank['month'] == month].reset_index(drop=True)
    if len(pool_month) == 0:
        raise ValueError(f"No real week available for month {month} in {config['bank_path']}.")

    value_cols = [c for c in bank.columns if c.startswith('t') and c[1:].isdigit()]
    all_positions = pool_month.index.to_numpy()

    rng = np.random.default_rng(config['seed'])
    profile_ids = sorted(metadata['profile_id'].unique())

    # One real week per profile -- and since metadata is 1:1 with the LV buses
    # (see generate_household_metadata), that is one real week PER BUS.
    # Drawn cyclically rather than with replacement: see _cyclic_draw.
    picks = _cyclic_draw(len(profile_ids), all_positions, rng)

    data = {}
    draw_counts = {}
    sources = []
    for pid, j in zip(profile_ids, picks):
        j = int(j)
        values = pool_month.loc[j, value_cols].to_numpy(dtype=float)  # already in kW

        draw_counts[j] = draw_counts.get(j, 0) + 1
        occurrence = draw_counts[j]
        if occurrence > 1:
            # 2nd occurrence -> ladder[0], 3rd -> ladder[1], ... Deterministic and
            # distinct, hence no collision; the modulo only bites past the end of
            # the ladder, which the [LOAD/draw] warning below reports explicitly.
            values = np.roll(values, BANK_JITTER_LADDER[(occurrence - 2) % len(BANK_JITTER_LADDER)])

        data[f"profile_{pid}"] = values

        # Which real week this profile actually came from. Recorded because
        # metadata.csv's source_* columns must describe THE DRAWN WEEK, not some
        # earlier unrelated draw: `source_street` is what _assign_loads matches
        # against bus streets, and `source_month` is what makes the seasonal
        # coherence with config['mois_simule'] checkable at all.
        # Only 'street'/'month' are required of the bank (the portable seam);
        # the finer traceability columns are taken when the site provides them.
        src = {'profile_id': pid, 'source_month': month,
               'source_street': pool_month.loc[j, 'street']}
        for bank_col, meta_col in (('week_start', 'source_week_start'),
                                   ('ean_idx', 'source_ean_index')):
            if bank_col in pool_month.columns:
                src[meta_col] = pool_month.loc[j, bank_col]
        sources.append(src)

    # Reuse is THE number to watch on a global draw: it tells whether the jitter
    # has few enough duplicates to decorrelate. Surfaced unconditionally so a
    # thin month cannot degrade the run silently.
    n_reused = sum(c - 1 for c in draw_counts.values() if c > 1)
    max_reuse = max(draw_counts.values())
    msg = (
        f"[LOAD/draw] {len(profile_ids)} profile(s) drawn from {len(all_positions)} "
        f"real week(s) of month {month}: {len(draw_counts)}/{len(all_positions)} "
        f"distinct week(s) used, max reuse {max_reuse}x, {n_reused} profile(s) jittered."
    )
    if max_reuse - 1 > len(BANK_JITTER_LADDER):
        msg += (
            f" WARNING: reuse exceeds the {len(BANK_JITTER_LADDER)} available offsets "
            "-- strictly identical profiles remain (bank too thin for this month)."
        )
    warnings.append(msg)

    households = pd.DataFrame(data, index=_WEEK_TIME_INDEX)
    households.index.name = "week_step"
    # Carried on .attrs rather than as a third return value: run_pv_scenario
    # already unpacks this as a 2-tuple, and pv.build_pv_production uses the same
    # attrs idiom for its own source week. Not persisted by to_csv -- consumed
    # in-process by generate_load_profiles, which writes it into metadata.csv.
    households.attrs['sources'] = pd.DataFrame(sources)
    return households, warnings


def generate_load_profiles(mois_simule, metadata=None, config=None,
                           out_path=None, data_path=None,
                           metadata_out_path=None, write_metadata=True):
    """Upstream generator (generation / execution split): produces
    `<data_path>/out/processed/load_profiles.csv` from the bank of real EAN
    weeks of month `mois_simule`, in the exact format run_digital_twin's
    `households` argument expects (index 'Mon 00:00'..'Sun 23:45', columns
    profile_{id}, values in kW).

    ALSO UPDATES AND REWRITES metadata.csv, because the two files are one act:
    the draw is what decides which real week -- hence which street and which
    month -- backs each profile_id. Writing load_profiles.csv while leaving
    metadata's source_* columns describing an unrelated draw is how the two
    silently drift apart.

    run_digital_twin receives both files and uses them as-is; `source_month` is
    what lets it warn when they were built for a different month than the one
    being simulated.

    Parameters
    ----------
    mois_simule : str | int   -- target month of the bank (French name or 1-12)
    metadata : pd.DataFrame, optional -- metadata skeleton (see
                                  generate_household_metadata); read from
                                  <data_path>/out/processed/metadata.csv if None
    config : dict, optional   -- overrides (seed, bank_path, ...); mois_simule is
                                  forced by the argument of the same name
    out_path : str, optional  -- path of the generated CSV (defaults to
                                  <data_path>/out/processed/load_profiles.csv)
    data_path : str, optional -- the scenario folder. Wins over config['data_path']
                                  when both are given; falls back to this module's
                                  own scenario.
    write_metadata : bool     -- write the updated metadata back to disk

    Returns
    -------
    (households : pd.DataFrame, metadata : pd.DataFrame, warnings : list[str])
    """
    # The scenario folder drives BOTH the outputs and the bank the draw reads
    # from. Resolving it once, before _config_to_dict, is what makes
    # cfg['bank_path'] land in the right scenario: otherwise a data_path passed
    # while config stays silent leaves the bank on the default 'data/', reading
    # one scenario's bank while writing another's.
    cfg_in = dict(config or {})
    if data_path is not None:
        cfg_in['data_path'] = data_path
    cfg_in.setdefault('data_path', SCENARIO_DATA_PATH)
    cfg = _config_to_dict(cfg_in)
    cfg['mois_simule'] = mois_simule

    processed_dir = os.path.join(cfg['data_path'], "out", "processed")
    if metadata is None:
        metadata = pd.read_csv(os.path.join(processed_dir, METADATA_FILE))
    if out_path is None:
        out_path = os.path.join(processed_dir, LOAD_PROFILES_FILE)
    if metadata_out_path is None:
        metadata_out_path = os.path.join(processed_dir, METADATA_FILE)

    households, warnings = _build_load_profiles_banque_mensuelle(metadata, cfg)

    # Fold the drawn weeks back into metadata, replacing any previous source_*.
    metadata = metadata.copy()
    sources = households.attrs['sources'].set_index('profile_id')
    for col in sources.columns:
        dtype = SOURCE_COLUMNS.get(col, (None, None))[0]
        mapped = metadata['profile_id'].map(sources[col])
        metadata[col] = mapped.astype(dtype) if dtype else mapped

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    households.to_csv(out_path)
    if write_metadata:
        os.makedirs(os.path.dirname(metadata_out_path) or ".", exist_ok=True)
        metadata.to_csv(metadata_out_path, index=False)

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
            f"Round-trip of {out_path} failed: the re-read CSV differs from the "
            "generated DataFrame (index/columns/values)."
        )

    return households, metadata, warnings


if __name__ == "__main__":
    # Bressoux is this module's own scenario (see the PORTABILITY note in the
    # module docstring), named explicitly rather than left to a default so
    # running another site is a matter of changing this line.
    _scenario = {'data_path': SCENARIO_DATA_PATH}
    build_consumption_bank(data_path=SCENARIO_DATA_PATH)
    build_irradiance_reference(_scenario)
