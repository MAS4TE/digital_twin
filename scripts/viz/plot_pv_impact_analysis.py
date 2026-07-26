"""
Post-processing + plotting for the PV impact study.

scripts/pv_impact_analysis.py builds the base network, runs the 0/30/50/80 PV
scenarios (summer) and 0/30/50 (winter), and hands the raw power-flow results
here. This module turns those results into the violation/reverse-flow metrics
(analyze) and the two comparison figures -- separated out so the "generate
scenarios" step (digital_twin.data_processing.pv) stays distinct from the
"analyze and plot" step.

Not a reusable library module -- analysis tooling only, results synthesized
by scripts/pv_impact_analysis.py.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V_LOW = 1.05   # seuil EN 50160 (usage courant)
V_HIGH = 1.10  # seuil EN 50160 (tolerance limitee, ex court, 5% du temps)

plt.rcParams.update({
    "font.family": "serif", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def bus_coords_from_network(network):
    """dict bus_id -> (lon, lat), from the built network bundle (net + buses)."""
    net, buses = network["net"], network["buses"]
    coords = {}
    for (street, num), b in buses.items():
        geo = json.loads(net.bus.loc[b, "geo"])
        coords[int(b)] = tuple(geo["coordinates"])
    return coords


def analyze(res, label, mois, bus_to_street, steps_per_day):
    """Compute the worst-instant metrics (voltage, reverse flow, loadings) for
    one scenario result (as returned by digital_twin.data_processing.pv.run_pv_scenario)."""
    vm = res["vm"]
    per_step_max = vm.max(axis=1)
    worst_step = int(per_step_max.idxmax())
    worst_day = worst_step // steps_per_day
    worst_hour = (worst_step % steps_per_day) * 15 // 60
    worst_min = (worst_step % steps_per_day) * 15 % 60

    vm_worst = vm.iloc[worst_step]
    n_over_low = int((vm_worst > V_LOW).sum())
    n_over_high = int((vm_worst > V_HIGH).sum())
    over_buses = vm_worst[vm_worst > V_LOW].sort_values(ascending=False)
    over_streets = pd.Series({b: bus_to_street.get(int(b), "MV/autre") for b in over_buses.index})

    ext_grid_p = res["ext_grid_p"].iloc[worst_step]
    ext_grid_val = float(ext_grid_p.iloc[0]) if hasattr(ext_grid_p, "iloc") else float(ext_grid_p)
    trafo_p_hv = res["trafo_p_hv"].iloc[worst_step]
    reverse_trafos = trafo_p_hv[trafo_p_hv < 0]
    max_line_loading = float(res["line_load"].iloc[worst_step].max())
    max_trafo_loading = float(res["trafo_load"].iloc[worst_step].max())

    print(f"\n--- {mois} / {label} ---")
    print(f"  Pire instant : pas {worst_step} (jour {worst_day}, ~{worst_hour:02d}h{worst_min:02d} UTC)")
    print(f"  Tension max reseau : {per_step_max.max():.4f} pu")
    print(f"  Bus > {V_LOW} pu : {n_over_low} ; bus > {V_HIGH} pu : {n_over_high}")
    if len(over_buses) > 0:
        print(f"  Rues concernees : {over_streets.value_counts().to_dict()}")
    print(f"  ext_grid p_mw au pire instant : {ext_grid_val:.4f} MW "
          f"{'(EXPORT / flux inverse vers l amont)' if ext_grid_val < 0 else '(import normal)'}")
    print(f"  Trafo en flux inverse (p_hv_mw<0) : {len(reverse_trafos)}/{len(trafo_p_hv)}"
          + (f" -- {reverse_trafos.round(4).to_dict()}" if len(reverse_trafos) else ""))
    print(f"  Charge ligne max : {max_line_loading:.1f}% ; charge trafo max : {max_trafo_loading:.1f}%")

    return {
        "mois": mois, "config": label, "worst_step": worst_step,
        "v_max": float(per_step_max.max()), "n_over_low": n_over_low, "n_over_high": n_over_high,
        "ext_grid_p_mw": ext_grid_val, "n_reverse_trafo": int(len(reverse_trafos)),
        "max_line_loading_pct": max_line_loading, "max_trafo_loading_pct": max_trafo_loading,
        "vm_worst": vm_worst, "over_buses": over_buses,
    }


def build_summary_table(metrics_by_label, labels, out_csv=None):
    """Comparative table (v_max, violation counts, reverse flow, max loading)
    across the given scenario labels; optionally written to `out_csv`."""
    rows = []
    for label in labels:
        m = metrics_by_label[label]
        rows.append({
            "config": label, "v_max_pu": round(m["v_max"], 4),
            "n_bus_over_1.05": m["n_over_low"], "n_bus_over_1.10": m["n_over_high"],
            "ext_grid_p_mw": round(m["ext_grid_p_mw"], 4), "n_trafo_reverse": m["n_reverse_trafo"],
            "max_line_loading_pct": round(m["max_line_loading_pct"], 1),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    if out_csv:
        summary_df.to_csv(out_csv, index=False)
    return summary_df


def plot_voltage_max_profile(summer, summer_metrics, steps_per_day, out_dir, ref_label="50 PV"):
    """Figure 1: max-voltage profile on the worst day, 0/30/50/80 superposed (summer)."""
    worst_day_ref = summer_metrics[ref_label]["worst_step"] // steps_per_day
    t = np.arange(steps_per_day) * 0.25
    colors = {"0 PV": "#444444", "30 PV": "#0072B2", "50 PV": "#D55E00", "80 PV": "#009E73"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, res in summer.items():
        day_vals = res["vm"].max(axis=1).to_numpy()[worst_day_ref * steps_per_day:(worst_day_ref + 1) * steps_per_day]
        ax.plot(t, day_vals, color=colors.get(label, "#888888"), lw=2.0,
                label=f"{label} (max={day_vals.max():.4f} pu)")
    ax.axhline(V_LOW, color="black", linestyle="--", linewidth=1, label=f"Seuil {V_LOW} pu (EN 50160)")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 4)])
    ax.set_xlabel("Heure de la journée (UTC)")
    ax.set_ylabel("Tension max sur le réseau [pu]")
    ax.set_title("Tension maximale — 0 / 30 / 50 / 80 installations PV (été)", loc="left", fontsize=13)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "voltage_max_profile_summer.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure ecrite : {out_path}")


def plot_overvoltage_map(coords, res_bus_ids, summer_metrics, pv_installations, out_dir, ref_label="50 PV"):
    """Figure 2: map of buses in overvoltage at the worst instant (ref_label scenario)."""
    vm_worst = summer_metrics[ref_label]["vm_worst"]
    pv_bus_ids = set(int(b) for b in pv_installations["bus_id"])

    xs = [coords[int(b)][0] for b in res_bus_ids]
    ys = [coords[int(b)][1] for b in res_bus_ids]
    vals = [float(vm_worst.get(b, np.nan)) for b in res_bus_ids]

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(xs, ys, c=vals, cmap="RdYlGn_r", vmin=0.98, vmax=1.08, s=45, zorder=1)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Tension [pu]")

    over_ids = [int(b) for b in res_bus_ids if float(vm_worst.get(b, np.nan)) > V_LOW]
    over_xs = [coords[b][0] for b in over_ids]
    over_ys = [coords[b][1] for b in over_ids]
    ax.scatter(over_xs, over_ys, facecolors="none", edgecolors="red", linewidths=2.0, s=140,
               label=f"> {V_LOW} pu ({len(over_ids)} bus)", zorder=2)

    pv_xs = [coords[b][0] for b in res_bus_ids if int(b) in pv_bus_ids]
    pv_ys = [coords[b][1] for b in res_bus_ids if int(b) in pv_bus_ids]
    ax.scatter(pv_xs, pv_ys, marker="+", color="black", s=60, linewidths=1.2,
               label=f"Installation PV ({ref_label})", zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Tension au pire instant ({ref_label}, été) — surtensions vs clusters PV", loc="left", fontsize=13)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "overvoltage_map_worst_instant.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure ecrite : {out_path}")
