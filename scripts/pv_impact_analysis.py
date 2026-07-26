"""
Volet PV -- Brique 4 : analyse d'impact (surtensions / flux inverses).

Compare plusieurs taux de pénétration PV (0 / 30 / 50 / 80 installations) sur
un mois d'été, cible le PIRE INSTANT (pas une moyenne journalière), puis
refait la même comparaison (0/30/50, pas 80) sur un mois d'hiver comme
contrôle de complétude (confirmer l'absence de surtension hivernale).

S'appuie sur l'architecture réseau de base + deepcopy de la brique 3
(`digital_twin.data_processing.pv.build_base_network` / `run_pv_scenario`).
Le garde-fou d'alignement temporel (`digital_twin.digital_twin._verify_temporal_alignment`)
reste appelé à l'intérieur de `run_pv_scenario()` avant chaque power flow.

This script only RUNS the scenarios; the metrics/figures are produced by
scripts/viz/plot_pv_impact_analysis.py (analyze / build_summary_table /
plot_voltage_max_profile / plot_overvoltage_map).
"""
import os
import sys
import time

import pandas as pd

from digital_twin.data_processing.pv import build_base_network, run_pv_scenario, verify_bus_ids
from digital_twin.utils import STEPS_PER_DAY

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz.plot_pv_impact_analysis import (
    analyze, bus_coords_from_network, build_summary_table,
    plot_overvoltage_map, plot_voltage_max_profile,
)

OUT_DIR = "outputs_pv_impact"

os.makedirs(OUT_DIR, exist_ok=True)

metadata = pd.read_csv("data/bressoux/out/processed/metadata.csv")
irradiance = pd.read_parquet("data/bressoux/out/processed/pvgis_irradiance_quarter_hourly.parquet")

pv_0 = pd.DataFrame(columns=["bus_id", "kwp", "type", "street"])
pv_30 = pd.read_csv("data/bressoux/inputs/pv_installations.csv")
pv_50 = pd.read_csv("data/bressoux/inputs/pv_installations_50.csv")
pv_80 = pd.read_csv("data/bressoux/inputs/pv_installations_80.csv")

SUMMER_CONFIGS = [("0 PV", pv_0), ("30 PV", pv_30), ("50 PV", pv_50), ("80 PV", pv_80)]
WINTER_CONFIGS = [("0 PV", pv_0), ("30 PV", pv_30), ("50 PV", pv_50)]

print("=== Construction du reseau de base (une fois) ===")
t0 = time.time()
network = build_base_network()
print(f"network construit en {time.time() - t0:.1f}s")

bus_to_street = {b: s for (s, num), b in network["buses"].items()}
for _, pv in [("30", pv_30), ("50", pv_50), ("80", pv_80)]:
    verify_bus_ids(pv, network)
print("bus_id des 3 parcs PV (30/50/80) verifies contre le reseau reel : OK")


def run_all(configs, mois):
    out = {}
    for label, pv in configs:
        t0 = time.time()
        res = run_pv_scenario(network, metadata, pv, irradiance, config={"mois_simule": mois})
        print(f"  {mois} / {label:<6} : {time.time() - t0:.1f}s, "
              f"n_sgen={len(res['net'].sgen)}, vm shape={res['vm'].shape}")
        out[label] = res
    return out


print("\n=== Scenarios ETE (juin) : 0 / 30 / 50 / 80 PV ===")
summer = run_all(SUMMER_CONFIGS, "juin")

print("\n=== Scenarios HIVER (janvier) : 0 / 30 / 50 PV (controle de completude) ===")
winter = run_all(WINTER_CONFIGS, "janvier")


print("\n" + "=" * 70)
print("ANALYSE -- ETE")
print("=" * 70)
summer_metrics = {label: analyze(res, label, "juin", bus_to_street, STEPS_PER_DAY) for label, res in summer.items()}

print("\n" + "=" * 70)
print("ANALYSE -- HIVER (controle de completude)")
print("=" * 70)
winter_metrics = {label: analyze(res, label, "janvier", bus_to_street, STEPS_PER_DAY) for label, res in winter.items()}

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF -- ETE")
print("=" * 70)
summary_df = build_summary_table(summer_metrics, ["0 PV", "30 PV", "50 PV", "80 PV"],
                                  out_csv=os.path.join(OUT_DIR, "summary_summer.csv"))

v0 = summer_metrics["0 PV"]["v_max"]
print("\nDelta tension max vs 0 PV :")
for label in ["30 PV", "50 PV", "80 PV"]:
    m = summer_metrics[label]
    print(f"  {label}: {m['v_max']:.4f} pu ({(m['v_max'] - v0) * 1000:+.1f} milli-pu vs 0 PV), "
          f"{m['n_over_low']} violation(s) > 1.05 pu")

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF -- HIVER (controle)")
print("=" * 70)
build_summary_table(winter_metrics, ["0 PV", "30 PV", "50 PV"],
                    out_csv=os.path.join(OUT_DIR, "summary_winter.csv"))

coords = bus_coords_from_network(network)
res_bus_ids = list(network["buses"].values())
plot_voltage_max_profile(summer, summer_metrics, STEPS_PER_DAY, OUT_DIR, ref_label="50 PV")
plot_overvoltage_map(coords, res_bus_ids, summer_metrics, pv_50, OUT_DIR, ref_label="50 PV")
