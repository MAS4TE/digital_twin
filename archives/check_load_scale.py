"""
Verification ponctuelle (integrate-bank-B, canal unique config['load_scale']) :
- non-regression actuelle/load_scale=1.0 (seed=369) vs les valeurs connues
- actuelle/load_scale=3 doit redonner EXACTEMENT les 72 violations connues
  (preuve que le nouveau canal reproduit l'ancien mecanisme x3 de main.py)
- sanity check : banque_mensuelle/load_scale=3 ~ 3x banque_mensuelle/load_scale=1
- detail des violations banque_mensuelle/load_scale=3 vs actuelle/load_scale=3 (72)
"""
import time

import pandas as pd

from digital_twin.digital_twin import run_digital_twin

DATA_PATH = "data/"
households = pd.read_csv(DATA_PATH + "households.csv")
load_profiles = pd.read_csv(DATA_PATH + "load_profiles.csv", index_col=0)

# Valeurs de reference deja mesurees (seed=369) :
PRE_REFACTOR_ACTUELLE_X1 = {
    "peak_agg_kw": 203.695737,
    "min_voltage_pu": 1.00316,
    "max_line_loading_pct": 29.511979,
}
KNOWN_ACTUELLE_X3_N_VIOLATIONS = 72

scenarios = {
    ("actuelle", "scale1"): {"methode_affectation": "actuelle", "load_scale": 1.0},
    ("actuelle", "scale3"): {"methode_affectation": "actuelle", "load_scale": 3.0},
    ("banque_mensuelle", "scale1"): {"methode_affectation": "banque_mensuelle", "load_scale": 1.0},
    ("banque_mensuelle", "scale3"): {"methode_affectation": "banque_mensuelle", "load_scale": 3.0},
}

results = {}
for key, cfg in scenarios.items():
    t0 = time.time()
    out = run_digital_twin(cfg, households, load_profiles)
    elapsed = time.time() - t0
    violations = out[out["category"] == "violation"].copy()
    summary = out[out["category"] == "summary"].set_index("metric")["value"]
    results[key] = (summary, violations)
    methode, scale = key
    print(f"{methode:<18} {scale:<7} | {elapsed:5.1f}s | "
          f"network_valid={summary['network_valid']} | "
          f"n_violations={len(violations)} | "
          f"peak_agg_kw={summary['peak_agg_kw']:.2f} | "
          f"min_voltage_pu={summary['min_voltage_pu']:.4f} | "
          f"max_line_loading_pct={summary['max_line_loading_pct']:.2f}")

print("\n=== Non-regression actuelle/load_scale=1.0 (seed=369) vs valeurs connues ===")
s = results[("actuelle", "scale1")][0]
for k, v_before in PRE_REFACTOR_ACTUELLE_X1.items():
    v_after = float(s[k])
    same = abs(v_after - v_before) < 1e-6
    print(f"  {k:<22} avant={v_before:.6f}  apres={v_after:.6f}  identique={same}")

print("\n=== Preuve : actuelle/load_scale=3 reproduit l'ancien x3 (72 violations) ===")
n_act_scale3 = len(results[("actuelle", "scale3")][1])
print(f"  n_violations actuelle/load_scale=3 = {n_act_scale3} "
      f"(connu avant, ancien mecanisme x3 = {KNOWN_ACTUELLE_X3_N_VIOLATIONS}) -> "
      f"identique={n_act_scale3 == KNOWN_ACTUELLE_X3_N_VIOLATIONS}")

print("\n=== Sanity check : banque_mensuelle load_scale=3 ~ 3x load_scale=1 ===")
p1 = float(results[("banque_mensuelle", "scale1")][0]["peak_agg_kw"])
p3 = float(results[("banque_mensuelle", "scale3")][0]["peak_agg_kw"])
print(f"  peak scale1={p1:.2f} kW, peak scale3={p3:.2f} kW, ratio={p3 / p1:.3f} (attendu ~3.0)")

print("\n=== Detail des violations (groupees par metrique + element) ===")
for key, (summary, violations) in results.items():
    methode, scale = key
    print(f"\n--- {methode} / {scale} ---")
    if len(violations) == 0:
        print("  (aucune violation)")
        continue
    grp = violations.groupby(["metric", "element"])["value"].agg(["count", "min", "max"])
    print(grp.sort_values("count", ascending=False).to_string())

print("\n=== Comparaison zones de violation : actuelle/scale3 vs banque_mensuelle/scale3 ===")
elems_act = set(zip(results[("actuelle", "scale3")][1]["metric"], results[("actuelle", "scale3")][1]["element"]))
elems_bank = set(zip(results[("banque_mensuelle", "scale3")][1]["metric"], results[("banque_mensuelle", "scale3")][1]["element"]))
print(f"  n elements distincts en violation -- actuelle: {len(elems_act)}, banque_mensuelle: {len(elems_bank)}")
print(f"  communs: {len(elems_act & elems_bank)}, uniquement actuelle: {len(elems_act - elems_bank)}, "
      f"uniquement banque_mensuelle: {len(elems_bank - elems_act)}")
