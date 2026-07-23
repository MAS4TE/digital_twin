"""
Volet PV -- Brique 2 : verifications de build_pv_production() / irradiance_to_power().

Verifie (cf. brief) :
  - une installation 3 kWp : pic cohérent en été, zéro la nuit ;
  - maison de repos (12 kWp) ~= 4x une résidentielle (3 kWp), mêmes heures ;
  - énergie journalière plausible (ordre de grandeur, pas 10x à côté) ;
  - juin >> janvier (sanity check de la sélection mensuelle) ;
  - production strictement nulle la nuit (pas de négatif/résiduel).

Script de verification, pas un livrable -- pas de sortie fichier (juste des
prints), a garder pour re-tester si le modele change.
"""
import numpy as np
import pandas as pd

from digital_twin.pv_production import (
    build_pv_production, DEFAULT_SYSTEM_FACTOR, STEPS_PER_DAY, PV_INSTALLATIONS_PATH, IRRADIANCE_PATH,
)

pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
irradiance = pd.read_parquet(IRRADIANCE_PATH)

print(f"system_factor utilise : {DEFAULT_SYSTEM_FACTOR}")
print(f"Installations : {len(pv_installations)} lignes -- "
      f"{(pv_installations['type']=='residential').sum()} residentielles "
      f"({pv_installations.loc[pv_installations['type']=='residential','kwp'].unique()} kWp), "
      f"{(pv_installations['type']=='rest_home').sum()} maison de repos "
      f"({pv_installations.loc[pv_installations['type']=='rest_home','kwp'].unique()} kWp)")

rest_home_bus = int(pv_installations.loc[pv_installations["type"] == "rest_home", "bus_id"].iloc[0])
res_bus_example = int(pv_installations.loc[pv_installations["type"] == "residential", "bus_id"].iloc[0])

results = {}
for mois in ("juin", "janvier"):
    prod = build_pv_production(mois, pv_installations, irradiance)
    results[mois] = prod

print("\n=== 1. Pic + nuit (installation residentielle 3 kWp, exemple bus "
      f"{res_bus_example}) ===")
for mois, prod in results.items():
    col = f"pv_{res_bus_example}"
    vals = prod[col].to_numpy()
    peak = vals.max()
    # "nuit" = minuit-5h et 21h-minuit (marge large, quel que soit le mois)
    night_mask = np.zeros(len(vals), dtype=bool)
    for day in range(7):
        base = day * STEPS_PER_DAY
        night_mask[base:base + 5 * 4] = True         # 00:00-05:00
        night_mask[base + 21 * 4:base + STEPS_PER_DAY] = True  # 21:00-24:00
    night_max = vals[night_mask].max()
    night_min = vals[night_mask].min()
    print(f"  {mois:<8} pic={peak:.3f} kW (kWp=3.0), nuit: min={night_min:.6f} max={night_max:.6f} kW")

print("\n=== 2. Maison de repos (12 kWp) ~= 4x residentielle (3 kWp), memes heures ===")
for mois, prod in results.items():
    ratio = (prod[f"pv_{rest_home_bus}"] / prod[f"pv_{res_bus_example}"].replace(0, np.nan)).dropna()
    ratio = ratio[np.isfinite(ratio)]
    print(f"  {mois:<8} ratio maison_de_repos/residentiel (heures productives) : "
          f"moyenne={ratio.mean():.3f}, min={ratio.min():.3f}, max={ratio.max():.3f} (attendu ~4.0)")

print("\n=== 3. Energie journaliere (kWh/jour, installation 3 kWp) ===")
print("  Cible indicative du brief : 8-12 kWh/jour ete, 1-3 kWh/jour hiver.")
for mois, prod in results.items():
    col = f"pv_{res_bus_example}"
    energy_per_day = prod[col].to_numpy().reshape(7, STEPS_PER_DAY).sum(axis=1) * 0.25
    print(f"  {mois:<8} energie/jour (7 jours) : {np.round(energy_per_day, 2).tolist()}, "
          f"moyenne={energy_per_day.mean():.2f} kWh/jour")

print("\n=== 4. Juin vs Janvier (sanity check selection mensuelle) ===")
e_juin = results["juin"][f"pv_{res_bus_example}"].sum() * 0.25
e_janvier = results["janvier"][f"pv_{res_bus_example}"].sum() * 0.25
print(f"  Energie semaine complete (3 kWp) : juin={e_juin:.1f} kWh, janvier={e_janvier:.1f} kWh, "
      f"ratio juin/janvier={e_juin / e_janvier:.2f}x")

print("\n=== 5. Production strictement nulle la nuit / jamais negative (toutes installations) ===")
for mois, prod in results.items():
    vals = prod.drop(columns=[]).to_numpy()
    n_negative = int((vals < 0).sum())
    night_mask = np.zeros(STEPS_PER_DAY * 7, dtype=bool)
    for day in range(7):
        base = day * STEPS_PER_DAY
        night_mask[base:base + 4 * 4] = True             # 00:00-04:00 (marge stricte)
        night_mask[base + 22 * 4:base + STEPS_PER_DAY] = True  # 22:00-24:00
    night_vals = vals[night_mask]
    print(f"  {mois:<8} valeurs negatives : {n_negative} (attendu 0) ; "
          f"max sur creneau 22h-4h : {night_vals.max():.6f} kW (attendu 0.0)")
