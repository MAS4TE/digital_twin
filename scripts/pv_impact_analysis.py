"""
Volet PV -- Brique 4 : analyse d'impact (surtensions / flux inverses).

Compare plusieurs taux de pénétration PV (0 / 30 / 50 / 80 installations) sur
un mois d'été, cible le PIRE INSTANT (pas une moyenne journalière), puis
refait la même comparaison (0/30/50, pas 80) sur un mois d'hiver comme
contrôle de complétude (confirmer l'absence de surtension hivernale).

S'appuie sur l'architecture réseau de base + deepcopy de la brique 3
(`sourcecode.pv_injection.build_base_network` / `run_pv_scenario`).
`verify_temporal_alignment()` reste active à l'intérieur de
`run_pv_scenario()` -- appelée systématiquement avant chaque power flow, pas
retirée (garde-fou explicitement demandé par le brief brique 4).

Génère aussi les 2 figures de contrôle dans outputs_pv_impact/ (backend Agg).
Pas un livrable réutilisable -- script d'analyse, résultats synthétisés dans
NEXT_STEPS.md.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sourcecode.pv_injection import build_base_network, run_pv_scenario, verify_bus_ids
from sourcecode.pv_production import STEPS_PER_DAY

OUT_DIR = "outputs_pv_impact"
V_LOW = 1.05   # seuil EN 50160 (usage courant)
V_HIGH = 1.10  # seuil EN 50160 (tolerance limitee, ex court, 5% du temps)

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

households = pd.read_csv("data/households.csv")
irradiance = pd.read_parquet("data/pvgis_irradiance_quarter_hourly.parquet")

pv_0 = pd.DataFrame(columns=["bus_id", "kwp", "type", "street"])
pv_30 = pd.read_csv("data/pv_installations.csv")
pv_50 = pd.read_csv("data/pv_installations_50.csv")
pv_80 = pd.read_csv("data/pv_installations_80.csv")

SUMMER_CONFIGS = [("0 PV", pv_0), ("30 PV", pv_30), ("50 PV", pv_50), ("80 PV", pv_80)]
WINTER_CONFIGS = [("0 PV", pv_0), ("30 PV", pv_30), ("50 PV", pv_50)]

print("=== Construction du reseau de base (une fois) ===")
t0 = time.time()
net_base, buses, mv_nodes, validation = build_base_network()
print(f"net_base construit en {time.time() - t0:.1f}s")

bus_to_street = {b: s for (s, num), b in buses.items()}
for _, pv in [("30", pv_30), ("50", pv_50), ("80", pv_80)]:
    verify_bus_ids(pv, buses, mv_nodes)
print("bus_id des 3 parcs PV (30/50/80) verifies contre le reseau reel : OK")


def _bus_coords():
    coords = {}
    for (street, num), b in buses.items():
        geo = json.loads(net_base.bus.loc[b, "geo"])
        coords[int(b)] = tuple(geo["coordinates"])
    return coords


def run_all(configs, mois):
    out = {}
    for label, pv in configs:
        t0 = time.time()
        res = run_pv_scenario(net_base, buses, mv_nodes, households, pv, irradiance,
                               config={"mois_simule": mois})
        print(f"  {mois} / {label:<6} : {time.time() - t0:.1f}s, "
              f"n_sgen={len(res['net'].sgen)}, vm shape={res['vm'].shape}")
        out[label] = res
    return out


print("\n=== Scenarios ETE (juin) : 0 / 30 / 50 / 80 PV ===")
summer = run_all(SUMMER_CONFIGS, "juin")

print("\n=== Scenarios HIVER (janvier) : 0 / 30 / 50 PV (controle de completude) ===")
winter = run_all(WINTER_CONFIGS, "janvier")


def analyze(res, label, mois):
    vm = res["vm"]
    per_step_max = vm.max(axis=1)
    worst_step = int(per_step_max.idxmax())
    worst_day = worst_step // STEPS_PER_DAY
    worst_hour = (worst_step % STEPS_PER_DAY) * 15 // 60
    worst_min = (worst_step % STEPS_PER_DAY) * 15 % 60

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


print("\n" + "=" * 70)
print("ANALYSE -- ETE")
print("=" * 70)
summer_metrics = {label: analyze(res, label, "juin") for label, res in summer.items()}

print("\n" + "=" * 70)
print("ANALYSE -- HIVER (controle de completude)")
print("=" * 70)
winter_metrics = {label: analyze(res, label, "janvier") for label, res in winter.items()}

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF -- ETE")
print("=" * 70)
summary_rows = []
for label in ["0 PV", "30 PV", "50 PV", "80 PV"]:
    m = summer_metrics[label]
    summary_rows.append({
        "config": label, "v_max_pu": round(m["v_max"], 4),
        "n_bus_over_1.05": m["n_over_low"], "n_bus_over_1.10": m["n_over_high"],
        "ext_grid_p_mw": round(m["ext_grid_p_mw"], 4), "n_trafo_reverse": m["n_reverse_trafo"],
        "max_line_loading_pct": round(m["max_line_loading_pct"], 1),
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUT_DIR, "summary_summer.csv"), index=False)

v0 = summer_metrics["0 PV"]["v_max"]
print("\nDelta tension max vs 0 PV :")
for label in ["30 PV", "50 PV", "80 PV"]:
    m = summer_metrics[label]
    print(f"  {label}: {m['v_max']:.4f} pu ({(m['v_max'] - v0) * 1000:+.1f} milli-pu vs 0 PV), "
          f"{m['n_over_low']} violation(s) > 1.05 pu")

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF -- HIVER (controle)")
print("=" * 70)
winter_rows = [{"config": label, "v_max_pu": round(winter_metrics[label]["v_max"], 4),
                "n_bus_over_1.05": winter_metrics[label]["n_over_low"]}
               for label in ["0 PV", "30 PV", "50 PV"]]
winter_df = pd.DataFrame(winter_rows)
print(winter_df.to_string(index=False))
winter_df.to_csv(os.path.join(OUT_DIR, "summary_winter.csv"), index=False)


# ============================================================
# Figure 1 : profil de tension max, jour du pire instant, 0/30/50/80 superposes (ete)
# ============================================================
worst_day_ref = summer_metrics["50 PV"]["worst_step"] // STEPS_PER_DAY
t = np.arange(STEPS_PER_DAY) * 0.25
colors = {"0 PV": "#444444", "30 PV": "#0072B2", "50 PV": "#D55E00", "80 PV": "#009E73"}

fig, ax = plt.subplots(figsize=(9, 5.5))
for label, res in summer.items():
    day_vals = res["vm"].max(axis=1).to_numpy()[worst_day_ref * STEPS_PER_DAY:(worst_day_ref + 1) * STEPS_PER_DAY]
    ax.plot(t, day_vals, color=colors[label], lw=2.0, label=f"{label} (max={day_vals.max():.4f} pu)")
ax.axhline(V_LOW, color="black", linestyle="--", linewidth=1, label=f"Seuil {V_LOW} pu (EN 50160)")
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0, 25, 4))
ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 4)])
ax.set_xlabel("Heure de la journée (UTC)")
ax.set_ylabel("Tension max sur le réseau [pu]")
ax.set_title("Tension maximale — 0 / 30 / 50 / 80 installations PV (été)", loc="left", fontsize=13)
ax.legend(frameon=True, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "voltage_max_profile_summer.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure ecrite : {OUT_DIR}/voltage_max_profile_summer.png")


# ============================================================
# Figure 2 : carte des bus en surtension au pire instant (config 50 PV, la plus parlante)
# ============================================================
coords = _bus_coords()
ref_label = "50 PV"
vm_worst = summer_metrics[ref_label]["vm_worst"]
pv_bus_ids = set(int(b) for b in pv_50["bus_id"])

res_bus_ids = [b for b in buses.values()]
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
fig.savefig(os.path.join(OUT_DIR, "overvoltage_map_worst_instant.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure ecrite : {OUT_DIR}/overvoltage_map_worst_instant.png")
