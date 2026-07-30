"""
PV impact analysis: overvoltages and reverse flows.

Compares several PV penetration rates (0 / 30 / 50 / 80 installations) over a
summer month, targeting the WORST INSTANT rather than a daily average, then
repeats the comparison (0/30/50, no 80) over a winter month as a completeness
control confirming the absence of winter overvoltage.

Built on the base-network + deepcopy architecture
(`digital_twin.data_processing.pv.build_base_network` / `run_pv_scenario`). The
temporal alignment guardrail
(`digital_twin.digital_twin._verify_temporal_alignment`) is called inside
`run_pv_scenario()` before each power flow.

The PV fleets are NESTED EQUIPMENT TABLES built here by `equipment.extend_pv`:
the 50-fleet contains the 30-fleet, the 80 contains the 50. That is what isolates
the effect of the penetration RATE from the effect of PLACEMENT -- an independent
draw per rate would mix the two and make the deltas uninterpretable. Building
them on the fly also removes the risk of stale CSVs.

This script only RUNS the scenarios; the metrics/figures are produced by
scripts/viz/plot_pv_impact_analysis.py (analyze / build_summary_table /
plot_voltage_max_profile / plot_overvoltage_map).
"""
import os
import sys
import time

import pandas as pd

from digital_twin.data_processing.equipment import (
    build_equipment_skeleton, extend_pv, generate_equipment_table,
)
from digital_twin.data_processing.pv import build_base_network, run_pv_scenario
from digital_twin.utils import (BUS_DOMAIN_EQUIPABLE, STEPS_PER_DAY, check_bus_ids,
                                irradiance_path_for)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz.plot_pv_impact_analysis import (
    analyze, bus_coords_from_network, build_summary_table,
    plot_overvoltage_map, plot_voltage_max_profile,
)

DATA_PATH = "data/bressoux"
OUT_DIR = "outputs_pv_impact"

# Site-calibrated placement, as in main.py -- these describe Bressoux, not the
# engine, so they travel with the scenario.
SCENARIO = {
    "data_path": DATA_PATH,
    "street_weights": {"Rue Ernest Malvoz": 3.0},
    "mv_pv_installations": {"Paradis du Bouhay": 12.0},
}

os.makedirs(OUT_DIR, exist_ok=True)

metadata = pd.read_csv(f"{DATA_PATH}/out/processed/metadata.csv")
irradiance = pd.read_parquet(irradiance_path_for(DATA_PATH))

print("=== Building the base network (once) ===")
t0 = time.time()
network = build_base_network(SCENARIO)
print(f"network built in {time.time() - t0:.1f}s")

# Nested fleets: 0 c 30 c 50 c 80.
pv_0 = build_equipment_skeleton(network)                  # no installation
pv_30 = generate_equipment_table(SCENARIO, write=False)   # the scenario's own fleet
pv_50 = extend_pv(pv_30.copy(), 20, SCENARIO)
pv_80 = extend_pv(pv_50.copy(), 30, SCENARIO)

SUMMER_CONFIGS = [("0 PV", pv_0), ("30 PV", pv_30), ("50 PV", pv_50), ("80 PV", pv_80)]
WINTER_CONFIGS = [("0 PV", pv_0), ("30 PV", pv_30), ("50 PV", pv_50)]

bus_to_street = {b: s for (s, num), b in network["buses"].items()}
for label, table in SUMMER_CONFIGS[1:]:
    equipped = table[table["pv_system_size_kwp"].fillna(0.0) > 0]
    check_bus_ids(equipped["bus_id"], network, BUS_DOMAIN_EQUIPABLE, f"{label} fleet")
    print(f"  {label:<6} : {len(equipped)} installation(s), bus_ids checked")


def run_all(configs, mois):
    out = {}
    for label, pv in configs:
        t0 = time.time()
        res = run_pv_scenario(network, metadata, pv, irradiance, config={"mois_simule": mois})
        print(f"  {mois} / {label:<6} : {time.time() - t0:.1f}s, "
              f"n_sgen={len(res['net'].sgen)}, vm shape={res['vm'].shape}")
        out[label] = res
    return out


print("\n=== SUMMER scenarios (June): 0 / 30 / 50 / 80 PV ===")
summer = run_all(SUMMER_CONFIGS, "juin")

print("\n=== WINTER scenarios (January): 0 / 30 / 50 PV (completeness control) ===")
winter = run_all(WINTER_CONFIGS, "janvier")


print("\n" + "=" * 70)
print("ANALYSIS -- SUMMER")
print("=" * 70)
summer_metrics = {label: analyze(res, label, "juin", bus_to_street, STEPS_PER_DAY) for label, res in summer.items()}

print("\n" + "=" * 70)
print("ANALYSIS -- WINTER (completeness control)")
print("=" * 70)
winter_metrics = {label: analyze(res, label, "janvier", bus_to_street, STEPS_PER_DAY) for label, res in winter.items()}

print("\n" + "=" * 70)
print("COMPARISON TABLE -- SUMMER")
print("=" * 70)
summary_df = build_summary_table(summer_metrics, ["0 PV", "30 PV", "50 PV", "80 PV"],
                                  out_csv=os.path.join(OUT_DIR, "summary_summer.csv"))

v0 = summer_metrics["0 PV"]["v_max"]
print("\nMax voltage delta vs 0 PV:")
for label in ["30 PV", "50 PV", "80 PV"]:
    m = summer_metrics[label]
    print(f"  {label}: {m['v_max']:.4f} pu ({(m['v_max'] - v0) * 1000:+.1f} milli-pu vs 0 PV), "
          f"{m['n_over_low']} violation(s) > 1.05 pu")

print("\n" + "=" * 70)
print("COMPARISON TABLE -- WINTER (control)")
print("=" * 70)
build_summary_table(winter_metrics, ["0 PV", "30 PV", "50 PV"],
                    out_csv=os.path.join(OUT_DIR, "summary_winter.csv"))

coords = bus_coords_from_network(network)
res_bus_ids = list(network["buses"].values())
plot_voltage_max_profile(summer, summer_metrics, STEPS_PER_DAY, OUT_DIR, ref_label="50 PV")
plot_overvoltage_map(coords, res_bus_ids, summer_metrics, pv_50, OUT_DIR, ref_label="50 PV")
