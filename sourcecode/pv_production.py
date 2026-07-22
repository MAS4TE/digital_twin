"""
PV volet -- Brick 2: irradiance -> PV production conversion.

PURE computation (no network, no pandapower, no create_sgen): converts the PVGIS
irradiance series (data/pvgis_irradiance_quarter_hourly.parquet) into per-
installation production profiles (data/pv_installations.csv, brick 1) for a given
month. Deliverable: build_pv_production().

Temporal consistency (irradiance <-> load):
  - Resolution: irradiance is quarter-hourly (35040 rows = 365 days x 96 steps/
    day, no residual hourly aggregation) -- alignable step by step with the load
    profiles (672 steps/week, same index convention "Mon 00:00".."Sun 23:45").
  - Month selection: same mois_simule parameter as the monthly bank (French
    month name or number 1-12). The irradiance (reference year 2023,
    PVGIS-SARAH3) and the load bank (real weeks Nov2024-Oct2025) do NOT share the
    same calendar year -- the consistency sought is at the MONTH level
    (seasonality), not the exact calendar week (impossible, different years).

Conversion model: LINEAR, isolated in irradiance_to_power(), replaceable later
by a full pvlib model (temperature, orientation, inverter) without touching the
rest of this brick.

    power_kW = irradiance_W_m2 / 1000 * kWp * system_factor

`system_factor` (default 0.8) encapsulates, IN A SINGLE NUMBER, both the real
system efficiency (inverter ~97%, wiring ~98%, soiling/mismatch ~95%, i.e. ~0.90
combined) AND the fact that we use GLOBAL HORIZONTAL irradiance (GHI, tilt=0 --
see sourcecode/consumption_bank.py) rather than plane-of-array irradiance (POA,
tilt ~30-35° typical in Belgium): a tilted panel collects relatively LESS at noon
in summer than a horizontal surface (sun near zenith) -- an effect a plain
constant factor cannot capture. Measured consequence: with system_factor=0.8,
June daily energy comes out a bit ABOVE the indicative 8-12 kWh/day for 3 kWp
range (~16 kWh/day) -- same order of magnitude, but outside the strict range.
Lowering it (~0.5) would match energy better but drop the peak below ~1.5 kW
instead of the ~3 kW expected at "full sun". This is a tension inherent to using
GHI rather than POA, not a bug -- `system_factor` remains THE exposed tuning
knob, and the whole conversion is isolated to allow a real POA computation
(pvlib) later without rewriting.
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
    """Accept a number (1-12) or a French month name; return 1-12."""
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
    """Real calendar week (672 quarter-hourly steps) representative of the
    requested month: the first Monday 00:00 whose month matches `mois` AND that
    has a full week of data following (may spill into the next month, like the
    real weeks of the load bank -- not an average, a real week).

    Returns
    -------
    irr_week : np.ndarray (672,)  -- irradiance [W/m2]
    monday : pd.Timestamp         -- start of the selected week
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
    """Linear model: power [kW] = irradiance [W/m2] / 1000 * kWp * system_factor.
    Isolated here so it can be replaced by a full pvlib model (temperature,
    orientation, inverter) without touching the rest of the brick -- see the
    module header for the default-factor rationale.

    Clips input irradiance at 0 (defensive: guarantees strictly zero, never
    negative, production at night, even if the source irradiance contained a
    negative numerical residue).
    """
    irr = np.maximum(np.asarray(irradiance_w_m2, dtype=float), 0.0)
    return irr / 1000.0 * kwp * system_factor


def build_pv_production(mois, pv_installations, irradiance, system_factor=DEFAULT_SYSTEM_FACTOR):
    """Build the PV production profiles (kW) of all installations in
    `pv_installations` (residential + rest home), for the representative week of
    month `mois`.

    Parameters
    ----------
    mois : str | int              -- target month (French name or 1-12)
    pv_installations : pd.DataFrame -- data/pv_installations.csv (bus_id, kwp, type, street)
    irradiance : pd.DataFrame       -- data/pvgis_irradiance_quarter_hourly.parquet (timestamp, ghi_w_m2, year)
    system_factor : float           -- see irradiance_to_power

    Returns
    -------
    pd.DataFrame : index 'Mon 00:00'..'Sun 23:45' (672 steps, SAME format as
    load_profiles), columns 'pv_{bus_id}' (kW), one per installation.
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
