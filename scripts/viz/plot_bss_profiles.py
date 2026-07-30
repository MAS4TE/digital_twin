"""
BSS: visual check of the battery fleet profiles.

Three figures, no power flow: everything is rebuilt from process_data (loads, PV,
BSS schedule) and _track_bss_soc, which takes seconds rather than the ~3 min of a
full run.

    1. Full week      : network load, PV, BSS fleet, net seen from the transformer
    2. BSS fleet + SoC: two stacked panels (never two superposed y axes)
    3. Daily profile  : the 7 days overlaid + the median, load and fleet

The fleet (central b0 + domestic b1..bn) is AGGREGATED here: these curves answer
"what does the transformer see". Per-battery detail belongs on a network map, not
on one more time series.

Why the engine internals rather than `out`: run_digital_twin returns only scalars
(summary), violations and warnings -- the time series computed in _run_powerflow
are consumed by _evaluate and then dropped. The other scripts/viz/ scripts reach
in the same way.

Check script, outside the production pipeline: read-only, writes a single PNG.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, so savefig does not depend on a display
import matplotlib.pyplot as plt

from digital_twin.data_processing.process_data import process_data
from digital_twin.digital_twin import (_BSS_KW_TO_MW, _assign_loads, _bss_battery_ids,
                                       _bss_fleet_specs, _bss_storage_name,
                                       _read_bss_schedule, _track_bss_soc)
from digital_twin.utils import STEPS_PER_DAY, STEPS_PER_WEEK, _config_to_dict

OUT_PATH = "scripts/viz/bss_profiles.png"

CONFIG = {
    'data_path': "data/bressoux",
    'n_timesteps': STEPS_PER_WEEK,
    'trafo_sn_mva': 0.22,
    'cos_phi': 0.95,
    'seed': 42,
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# Slots 1-3 of the reference categorical palette, in order. Validated: all pairs,
# light mode, white surface -> CVD dE 9.2 (>=8), normal vision dE 24.0 (>=15).
# The aqua sits at 2.82:1 on white, under the 3:1 threshold, so the relief rule
# applies -- hence the direct labels on the curves.
C_LOAD = "#2a78d6"   # slot 1, blue
C_PV = "#eb6834"     # slot 2, orange
C_BSS = "#1baf7a"    # slot 3, aqua
INK = "#0b0b0b"      # primary ink -- the net is a resultant, not a category
MUTED = "#898781"    # axes / labels
GRID = "#e1e0d9"


def load_series():
    """All series in kW, over 672 steps, positionally aligned.

    The load comes from _assign_loads and NOT from households.sum(): households
    is the POOL of profiles (333 columns, 48-110 kW), not the network load. The
    profile -> bus assignment and the 4 MV loads only exist after assignment,
    hence the ~199 kW real peak.
    """
    cfg = _config_to_dict(CONFIG)
    metadata, network, households, pv_profiles, schedules_BSS, equipment = process_data(CONFIG)
    if schedules_BSS is None:
        raise SystemExit(
            "No schedules_BSS for this scenario: the equipment table "
            "(out/processed/equipment.csv) carries no domestic battery. Generate it "
            "via digital_twin.data_processing.equipment.generate_equipment_table first."
        )

    df_p, _, _, _ = _assign_loads(network, metadata, households, cfg)
    load_kw = df_p.sum(axis=1).to_numpy() * 1000.0
    pv_kw = (pv_profiles.sum(axis=1).to_numpy() * 1000.0
             if pv_profiles is not None else np.zeros_like(load_kw))

    # The fleet is aggregated FOR THE FIGURE only; the twin injects one battery
    # per bus. These curves answer "what does the transformer see", and the sum
    # is the right answer to that question.
    specs = _bss_fleet_specs(cfg, network, equipment, _bss_battery_ids(schedules_BSS))
    df_bss_p, _ = _read_bss_schedule(schedules_BSS, df_p, cfg, specs,
                                     load_time_index=households.index)

    # SoC and the series ACTUALLY applied: _track_bss_soc can clip, and the
    # clipped version is what the twin injects. Plotting the raw schedule would
    # show a figure the power flow never sees.
    df_soc, df_applied, _ = _track_bss_soc(df_bss_p, specs, cfg)
    bss_kw = df_applied.sum(axis=1).to_numpy() / _BSS_KW_TO_MW

    # Fleet SoC = total stored energy / total capacity, hence weighted by the
    # capacities. A plain mean would give a 10 kWh battery the same weight as
    # the 2600 kWh park.
    caps = np.array([specs[b]['e_kwh'] for b in specs], dtype=float)
    order = [_bss_storage_name(b) for b in specs]
    soc = (df_soc[order].to_numpy() * caps).sum(axis=1) / caps.sum()

    # Sign: positive BSS = charging = consumption, so it ADDS to the net where
    # PV subtracts from it (see _inject_bss_storage).
    net_kw = load_kw - pv_kw + bss_kw
    return load_kw, pv_kw, bss_kw, net_kw, soc * 100.0


def _label_peak(ax, y, text, color):
    """Direct label placed on the curve's maximum.

    Not at the end of the curve: the four series converge towards 0 on Sunday
    evening, where the labels would overlap. On its peak each one is isolated.
    Also serves the relief rule -- the aqua sits at 2.82:1 on white.
    """
    k = int(np.argmax(y))
    ax.annotate(text, xy=(k, y[k]), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=10, color=color, fontweight="bold")


def _day_axis(ax, n):
    ax.set_xlim(0, n)
    ax.set_xticks(np.arange(0, n + 1, STEPS_PER_DAY))
    ax.set_xticklabels(DAYS[:n // STEPS_PER_DAY] + [""], ha="left")
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID)


def build_figure():
    load, pv, bss, net, soc = load_series()
    n = len(load)
    x = np.arange(n)

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.0, 0.8, 1.2], hspace=0.55, wspace=0.18)

    # --- 1. The week -----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axhline(0, color=MUTED, lw=1.0, zorder=1)
    ax1.plot(x, load, color=C_LOAD, lw=2.0, label="Network load", zorder=3)
    ax1.plot(x, pv, color=C_PV, lw=2.0, label="PV production", zorder=3)
    ax1.plot(x, bss, color=C_BSS, lw=2.0, label="BSS fleet (+ charge / − discharge)", zorder=3)
    ax1.plot(x, net, color=INK, lw=1.6, ls="--", label="Net seen from the transformer", zorder=4)
    _label_peak(ax1, bss, "BSS fleet", C_BSS)
    ax1.margins(y=0.16)
    _day_axis(ax1, n)
    ax1.set_ylabel("Power [kW]", color=MUTED)
    ax1.set_title(
        f"Simulated week — BSS fleet from {bss.min():+.0f} to {bss.max():+.0f} kW, "
        f"neighbourhood load peaking at {load.max():.0f} kW",
        loc="left", fontsize=13, color=INK)
    ax1.legend(frameon=False, fontsize=10, ncol=4, loc="upper left",
               bbox_to_anchor=(0, -0.10))

    # --- 2. BSS fleet then SoC: TWO PANELS, never two y axes --------------
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axhline(0, color=MUTED, lw=1.0)
    ax2.fill_between(x, 0, bss, where=bss >= 0, color=C_BSS, alpha=0.30, lw=0)
    ax2.fill_between(x, 0, bss, where=bss < 0, color=C_LOAD, alpha=0.30, lw=0)
    ax2.plot(x, bss, color=C_BSS, lw=1.6)
    _day_axis(ax2, n)
    ax2.set_ylabel("BSS fleet [kW]", color=MUTED)
    ax2.set_title("Fleet power — green area: charging, blue area: discharging",
                  loc="left", fontsize=12, color=INK)

    ax3 = fig.add_subplot(gs[2, :], sharex=ax2)
    ax3.axhspan(10, 100, color=GRID, alpha=0.55, lw=0, zorder=1)
    ax3.plot(x, soc, color=C_LOAD, lw=2.0, zorder=3)
    ax3.axhline(10, color=MUTED, lw=1.0, ls=":", zorder=2)
    _day_axis(ax3, n)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("SoC [%]", color=MUTED)
    ax3.set_title(f"State of charge — usable band 10–100 %, floor reached at "
                  f"{soc.min():.0f} %, end of week {soc[-1]:.0f} %",
                  loc="left", fontsize=12, color=INK)

    # --- 3. Daily profile: 7 days overlaid + median -----------------------
    n_days = n // STEPS_PER_DAY
    hours = np.arange(STEPS_PER_DAY) * 24.0 / STEPS_PER_DAY
    for col, (serie, titre, couleur) in enumerate(
            ((load, "Network load", C_LOAD), (bss, "BSS fleet", C_BSS))):
        ax = fig.add_subplot(gs[3, col])
        days = serie[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        for d in range(n_days):
            ax.plot(hours, days[d], color=MUTED, lw=0.9, alpha=0.45, zorder=2)
        med = np.median(days, axis=0)
        ax.plot(hours, med, color=couleur, lw=2.4, zorder=3, label="Median")
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        # Annotation placed on the INNER side of the extremum (above a minimum,
        # below a maximum): the other way round pushes it out of the axes, onto
        # the title on the left and under the ticks on the right.
        k = int(np.argmax(np.abs(med)))
        ax.margins(y=0.18)
        ax.annotate(f"{med[k]:+.0f} kW at {int(hours[k]):02d}:{int(hours[k] % 1 * 60):02d}",
                    xy=(hours[k], med[k]), xytext=(0, 12 if med[k] < 0 else -20),
                    textcoords="offset points", ha="center", fontsize=10, color=INK)
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 6)])
        ax.tick_params(colors=MUTED)
        ax.grid(color=GRID)
        ax.set_xlabel("Hour", color=MUTED)
        if col == 0:
            ax.set_ylabel("Power [kW]", color=MUTED)
        ax.set_title(f"{titre} — {n_days} days overlaid (grey) and median",
                     loc="left", fontsize=12, color=INK)
        ax.legend(frameon=False, fontsize=10, loc="best")

    fig.suptitle("Bressoux digital twin — battery fleet profiles (b0 + b1..bn)",
                 x=0.012, ha="left", fontsize=15, color=INK)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Figure written: {OUT_PATH}")
