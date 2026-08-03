"""
Presentation figure of the consumption profiles used in the network: two stacked
panels over one typical day (96 quarter-hourly steps), no twin axis.

  Panel 1: the 4 MV loads (rest home, horticultural centre, sports hall, school)
  Panel 2: 3 households at the 25th/50th/75th percentile

Presentation script, outside the production pipeline: read-only over the already
materialized data files, writes a PNG and a PDF.

DATA SOURCES
    1. load_habitation_day() -> out/processed/consumption_bank_real_weeks.parquet
       Schema: meta columns (ean_idx, ean_anon, street, month, season,
       week_start) + 672 value columns t0000..t0671, in kW (as load_profiles.csv).
       HAB_IDX holds `ean_idx` values -- real EAN identifiers, several weeks per
       EAN in the bank -- NOT row positions.
    2. load_mv_day() -> out/processed/mv_load_profiles.csv
       672 rows ('Mon 00:00'..'Sun 23:45'), one column per facility name (see
       data/bressoux/inputs/network.py, MV_LOADS). A SINGLE fixed weekly profile,
       with no per-month segmentation unlike the households. Also kW.

HOUSEHOLD SELECTION is automatic (_select_percentile_households): the 3 EANs
whose MEDIAN weekly energy, over all their available weeks across the whole year,
sits closest to the 25/50/75 percentiles of the population -- excluding
households whose MEDIAN peak/mean ratio marks a recurrently atypical profile.

The exclusion threshold is calibrated on the real distribution: a naive
"peak < 3-4x mean" would drop 97% of the weeks, since individual residential
meters have an intrinsically low load factor (real median ratio ~10x over the
bank's 1346 weeks). The per-EAN MEDIAN ratio spans 1.8x to 21.0x, and
PEAK_TO_MEAN_MAX = 15.0 excludes only the three EANs that stand clearly apart
from the rest of the population (<= 13.9x).

TEMPORAL ALIGNMENT (_get_common_week_rows): among the weeks where all three
HAB_IDX have data, pick the one maximising the MINIMUM of the daily peaks of the
first two. This guarantees the compared profiles come from the same real calendar
week AND day, and that both significant levels show visible activity that day --
maximising only the first one's peak can leave the second nearly flat. No
inter-week averaging, which would flatten the peaks.
"""

import calendar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend, so savefig does not need a display
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
N_STEPS_DAY = 96               # 24 h * 4 (quarter-hourly)
DT_H = 0.25

# --- Parameters -----------------------------------------------------------
BANK_PATH = "data/bressoux/out/processed/consumption_bank_real_weeks.parquet"
MV_PROFILES_PATH = "data/bressoux/out/processed/mv_load_profiles.csv"
MOIS = "juin"                 # month displayed; does not affect the household
                               # selection, which spans the whole year
                               # (see _select_percentile_households). The day of
                               # week shown is resolved by _get_common_week_rows.

PEAK_TO_MEAN_MAX = 15.0       # excludes households whose MEDIAN peak/mean ratio
                              # (all weeks, whole year) exceeds this -- see the
                              # module docstring for the calibration
HOUSEHOLD_PERCENTILES = (25, 50, 75)  # low / median / high

# Column names of mv_load_profiles.csv (see data/bressoux/inputs/network.py,
# MV_LOADS): rest home, horticultural centre, sports hall, school.
MV_NAMES = ["Paradis du Bouhay", "Plantations", "Omnisport BX", "Groupe Scolaire"]
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
    """Select 3 representative EANs (low/median/high) -- see the module docstring
    for the rationale and the threshold calibration.

    For each EAN, computed over ALL its available weeks (whole year, independent
    of the displayed month MOIS):
      - median_energy : median weekly energy [kWh], the household's typical size
      - median_ratio  : median peak/mean ratio [-], how typical the profile is
    EANs with median_ratio > ratio_threshold are excluded as recurrently
    atypical; for each requested percentile, the eligible EAN whose median_energy
    is closest to it is kept (percentiles computed over the eligible EANs only).

    Returns
    -------
    list[int] : 3 ean_idx, in the order of `percentiles` (low -> high).
    """
    df = pd.read_parquet(BANK_PATH)
    value_cols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]
    vals = df[value_cols].to_numpy(dtype=float)

    stats = df[["ean_idx"]].copy()
    with np.errstate(invalid="ignore"):  # zero-mean weeks (dead meter) -> NaN, skipped by median()
        stats["peak_to_mean"] = vals.max(axis=1) / vals.mean(axis=1)
    stats["energy_kwh"] = vals.sum(axis=1) * DT_H

    per_ean = stats.groupby("ean_idx").agg(
        median_ratio=("peak_to_mean", "median"),
        median_energy=("energy_kwh", "median"),
    )
    eligible = per_ean[per_ean["median_ratio"] <= ratio_threshold].copy()
    if eligible.empty:
        raise ValueError(f"No eligible household under the peak/mean threshold {ratio_threshold}.")

    targets = eligible["median_energy"].quantile([p / 100 for p in percentiles])
    chosen = []
    for target in targets:
        remaining = eligible.drop(index=chosen, errors="ignore")
        closest = int((remaining["median_energy"] - target).abs().idxmin())
        chosen.append(closest)

    excluded = sorted(set(per_ean.index) - set(eligible.index))
    print(f"Households selected (percentiles {percentiles}, peak/mean <= {ratio_threshold}x):")
    for p, idx in zip(percentiles, chosen):
        print(f"  p{p:<3} -> ean_idx={idx:<3} median energy={per_ean.loc[idx, 'median_energy']:.1f} kWh/week, "
              f"median peak/mean ratio={per_ean.loc[idx, 'median_ratio']:.2f}x")
    print(f"  households excluded (ratio > {ratio_threshold}x): {excluded}")

    return chosen


HAB_IDX = _select_percentile_households()


# --- DATA LOADING ---------------------------------------------------------
def _day_slice(week_values, day_index):
    """Extract day `day_index` (96 points) from a 672-point weekly profile."""
    week = np.asarray(week_values, dtype=float)
    start = day_index * N_STEPS_DAY
    return week[start:start + N_STEPS_DAY]


_COMMON_WEEK_CACHE = {}


def _get_common_week_rows(idxs, mois):
    """Among the weeks of `mois` where ALL `idxs` have data, find the
    (week, day) maximising the MINIMUM of the daily peaks of idxs[0] and
    idxs[1] -- so the compared profiles come from the same real calendar week
    and the two most significant levels both show a visible peak that day.

    Maximising ONLY idxs[0]'s peak is not enough: on some common weeks it peaks
    hard while idxs[1] stays nearly flat that very day.

    Returns
    -------
    rows : dict[idx -> pd.Series]  (one bank row per idx, same week)
    best_week : str                (ISO week_start of the selected week)
    best_day : int                 (0=Monday..6=Sunday)
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
            f"No calendar week common to ean_idx {idxs} for month {mois!r}."
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
    """96 values (kW) -- one real day of household `idx` (an ean_idx), taken from
    the calendar week common to all HAB_IDX, on the day selected by
    _get_common_week_rows."""
    rows, _, best_day = _get_common_week_rows(HAB_IDX, mois)
    row = rows[idx]
    value_cols = [c for c in row.index if c.startswith("t") and c[1:].isdigit()]
    week_kw = row[value_cols].to_numpy(dtype=float)
    return _day_slice(week_kw, best_day)


def load_mv_day(mv_name, mois):
    """96 values (kW) -- one day of the MV profile `mv_name`, on the same weekday
    as the household panel so both panels show a coherent "typical day".

    The MV profiles are a single fixed weekly profile with no per-month
    segmentation, so `mois` only serves to resolve, through HAB_IDX, which
    weekday to align on.
    """
    _, _, best_day = _get_common_week_rows(HAB_IDX, mois)
    mv = pd.read_csv(MV_PROFILES_PATH, index_col=0)
    week_kw = mv[mv_name].to_numpy(dtype=float)
    return _day_slice(week_kw, best_day)


def week_energy_kwh_hab(idx, mois):
    """Weekly energy (kWh) of a household, for the legend annotation -- same
    common week as load_habitation_day."""
    rows, _, _ = _get_common_week_rows(HAB_IDX, mois)
    row = rows[idx]
    value_cols = [c for c in row.index if c.startswith("t") and c[1:].isdigit()]
    week_kw = row[value_cols].to_numpy(dtype=float)
    return float(np.sum(week_kw) * DT_H)


# --- FIGURE ---------------------------------------------------------------
def build_figure():
    t = np.arange(N_STEPS_DAY) * DT_H
    fig, (ax_mv, ax_hab) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # Panel 1: the 4 MV loads
    for name, c in zip(MV_NAMES, COLORS_MV):
        y = load_mv_day(name, MOIS)
        ax_mv.plot(t, y, color=c, lw=2.0, label=MV_LABELS.get(name, name))
    ax_mv.set_ylabel("Power [kW]")
    ax_mv.set_title("MV loads", loc="left", fontsize=22)
    ax_mv.set_ylim(bottom=0)
    ax_mv.legend(frameon=True, fontsize=15)

    # Panel 2: 3 households (low/median/high percentiles, see
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
        fig.savefig(f"digital_twin/plots/profils_reseau.{ext}", dpi=200, bbox_inches="tight")
    print("Figures written: digital_twin/plots/profils_reseau.png / .pdf")
