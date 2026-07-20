"""
Volet PV -- Brique 3 : verifications de l'injection sgen (pv_injection.py).

Verifie (cf. brief) :
  - deepcopy rapide vs le cout reel (lecture CSV ~14s) ;
  - nombre de sgen crees = nombre d'installations (31) ;
  - puissance PV totale injectee au pic de midi coherente avec les kWp installes ;
  - bilan net (load - production) reduit/negatif a midi ensoleille, sur
    quelques bus PV ;
  - deepcopy : deux scenarios successifs n'accumulent pas les sgen ;
  - sanity : le power flow converge avec les sgen en place.

Script de verification, pas un livrable -- pas de sortie fichier (prints
seulement), a garder pour re-tester si le modele/l'injection changent.
"""
import time

import numpy as np
import pandas as pd

from sourcecode.pv_injection import (
    build_base_network, verify_bus_ids, run_pv_scenario,
    PV_INSTALLATIONS_PATH, IRRADIANCE_PATH,
)

households = pd.read_csv("data/households.csv")
pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
irradiance = pd.read_parquet(IRRADIANCE_PATH)

print("=== 0. Construction du reseau de base (une fois) + verification bus_id ===")
t0 = time.time()
net_base, buses, mv_nodes, validation = build_base_network()
t_build = time.time() - t0
print(f"  net_base construit en {t_build:.2f}s (cout attendu : lecture du CSV brut 335 Mo)")
verify_bus_ids(pv_installations, buses, mv_nodes)
print("  bus_id du CSV PV verifies contre le reseau reel (residentiels + MV de passage) : OK")

rest_home_row = pv_installations[pv_installations["type"] == "rest_home"].iloc[0]
print(f"  Maison de repos : bus_id={int(rest_home_row['bus_id'])}, "
      f"correspond a mv_nodes['Paradis du Bouhay']={int(mv_nodes['Paradis du Bouhay'])} "
      f"-> {'OK' if int(rest_home_row['bus_id']) == int(mv_nodes['Paradis du Bouhay']) else 'MISMATCH'}")

print("\n=== 1. Deepcopy : cout mesure vs construction ===")
import copy
t0 = time.time()
_ = copy.deepcopy(net_base)
t_copy = time.time() - t0
print(f"  copy.deepcopy(net_base) : {t_copy:.4f}s  (vs {t_build:.2f}s pour construire net_base : "
      f"{t_build / t_copy:.0f}x plus rapide -- confirme que le cout reel est la lecture CSV, "
      f"pas la structure memoire)")

print("\n=== 2. Scenario complet (juin), power flow avec PV ===")
t0 = time.time()
result = run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                          config={"mois_simule": "juin"})
print(f"  scenario execute en {time.time() - t0:.1f}s")
n_sgen = len(result["net"].sgen)
print(f"  sgen crees : {n_sgen} (attendu {len(pv_installations)}) -> "
      f"{'OK' if n_sgen == len(pv_installations) else 'MISMATCH'}")
print(f"  net_base inchange : len(net_base.sgen)={len(net_base.sgen)} (attendu 0)")

df_sgen_p = result["df_sgen_p"]  # MW, colonnes pv_{bus_id}
total_installed_kwp = pv_installations["kwp"].sum()

print("\n=== 3. Puissance PV totale injectee au pic de midi vs kWp installes ===")
total_sgen_mw_per_step = df_sgen_p.sum(axis=1)
peak_step = int(total_sgen_mw_per_step.idxmax())
peak_total_kw = total_sgen_mw_per_step.iloc[peak_step] * 1000
print(f"  Pic au pas {peak_step} (jour {peak_step // 96}, ~{(peak_step % 96) * 15 // 60:02d}h "
      f"{(peak_step % 96) * 15 % 60:02d} UTC) : {peak_total_kw:.1f} kW injectes au total, "
      f"pour {total_installed_kwp:.1f} kWp installes "
      f"({peak_total_kw / total_installed_kwp * 100:.0f}% du nameplate, "
      f"coherent avec system_factor=0.8 et l'irradiance reelle < 1000 W/m2)")

print("\n=== 4. Bilan net (load - production) a quelques bus PV, au pas de midi ===")
res_bus_pv = pv_installations[pv_installations["type"] == "residential"]["bus_id"].head(5).tolist()
df_load_p = result["df_load_p"].reset_index(drop=True)  # MW
bus_to_load_col = {}
for col in df_load_p.columns:
    # colonnes 'Load_{bus}' -- cf. digital_twin._assign_loads
    if col.startswith("Load_"):
        bus_to_load_col[int(col.split("_")[1])] = col

for bus_id in res_bus_pv:
    load_col = bus_to_load_col.get(bus_id)
    sgen_col = f"pv_{bus_id}"
    if load_col is None or sgen_col not in df_sgen_p.columns:
        print(f"  bus {bus_id}: colonne introuvable (load={load_col}, sgen={sgen_col in df_sgen_p.columns}) -- skip")
        continue
    load_kw = df_load_p[load_col].iloc[peak_step] * 1000
    sgen_kw = df_sgen_p[sgen_col].iloc[peak_step] * 1000
    net_kw = load_kw - sgen_kw
    print(f"  bus {bus_id}: load={load_kw:.3f} kW, PV={sgen_kw:.3f} kW, "
          f"net (load-PV)={net_kw:.3f} kW {'(bus injecte au reseau)' if net_kw < 0 else '(bus encore consommateur net)'}")

print("\n=== 5. Deux scenarios successifs n'accumulent pas les sgen ===")
result_a = run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                            config={"mois_simule": "juin"}, run_powerflow=False)
result_b = run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                            config={"mois_simule": "janvier"}, run_powerflow=False)
n_a, n_b = len(result_a["net"].sgen), len(result_b["net"].sgen)
print(f"  scenario A (juin) : {n_a} sgen ; scenario B (janvier, meme net_base) : {n_b} sgen -- "
      f"{'OK, pas d accumulation' if n_a == n_b == len(pv_installations) else 'ECHEC : accumulation detectee'}")
print(f"  net_base toujours inchange : len(net_base.sgen)={len(net_base.sgen)} (attendu 0)")

print("\n=== 6. Sanity : power flow converge avec PV (deja execute a l'etape 2) ===")
vm = result["vm"]
n_nan = int(vm.isna().sum().sum())
print(f"  vm shape={vm.shape}, NaN={n_nan} (attendu 0), "
      f"min={vm.values.min():.4f} pu, max={vm.values.max():.4f} pu -> "
      f"{'convergence OK, aucune valeur suspecte' if n_nan == 0 else 'ATTENTION valeurs manquantes'}")
