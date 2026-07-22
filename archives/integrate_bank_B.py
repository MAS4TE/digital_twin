"""
Comparaison "actuelle" vs "banque_mensuelle" (integrate-bank-B) : effet de la
nouvelle methode d'affectation des profils de charge (vraies semaines EAN,
segmentees par mois -- cf. explore_seasonal_pv.py) sur les resultats du power
flow complet (sourcecode.digital_twin.run_digital_twin), pour un mois d'hiver
et un mois d'ete.

Contexte : "actuelle" tire dans la banque plate de 350 profils deja
perturbes (shift+/-2h, scale lognormal) de profil_conso.ipynb -- elle ne varie
PAS par mois (cf. resultat ci-dessous : les baselines janvier/juillet sont
identiques par construction, c'est justement ce que banque_mensuelle corrige).
"banque_mensuelle" tire, avec la meme seed, une vraie semaine EAN dans la
sous-banque du mois choisi, sans shift/scale.

Prerequis : data/consumption_bank_real_weeks.parquet materialise (executer
`python sourcecode/consumption_bank.py` si absent -- digital_twin.py leve une
erreur explicite sinon).

Ne modifie aucune donnee source. Lit uniquement households.csv/load_profiles.csv
(deja existants, utilises tels quels par les deux methodes -- la methode
banque_mensuelle ne s'en sert pas pour l'affectation mais _validate_inputs les
requiert toujours en entree du contrat MAS4TE).
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # cf. incident recurrent matplotlib/savefig sur cette machine
import matplotlib.pyplot as plt

from sourcecode.digital_twin import run_digital_twin, CONSUMPTION_BANK_FILE

DATA_PATH = "data/"
OUT_DIR = "outputs_integrate_B"
os.makedirs(OUT_DIR, exist_ok=True)

bank_path = DATA_PATH + CONSUMPTION_BANK_FILE
if not os.path.exists(bank_path):
    raise FileNotFoundError(
        f"{bank_path} introuvable. Executer `python sourcecode/consumption_bank.py` "
        "avant de lancer cette comparaison."
    )

WINTER_MONTH = 1   # janvier -- saison dimensionnante (cf. brief)
SUMMER_MONTH = 7   # juillet
MONTHS = [WINTER_MONTH, SUMMER_MONTH]
MONTH_LABEL = {WINTER_MONTH: "Janvier (hiver)", SUMMER_MONTH: "Juillet (ete)"}

BASE_SEED = 42
ROBUSTNESS_SEEDS = list(range(5))     # 5 seeds (bas de la fourchette 5-10 demandee --
                                       # chaque run_digital_twin prend ~4-4.5 min sur cette
                                       # machine ; 5 seeds x 2 methodes = ~45 min, deja
                                       # consequent en plus des 4 runs de comparaison)
ROBUSTNESS_MONTH = WINTER_MONTH       # saison dimensionnante

METHODS = ["actuelle", "banque_mensuelle"]
METHOD_LABEL = {"actuelle": "Actuelle (banque 350, shift+scale)",
                "banque_mensuelle": "Banque mensuelle (vraies semaines)"}
COLORS = {"actuelle": "#0072B2", "banque_mensuelle": "#D55E00"}

days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
day_ticks = [d * 96 for d in range(7)]

print("Chargement households.csv / load_profiles.csv (entrees MAS4TE standard)...")
households = pd.read_csv(DATA_PATH + "households.csv")
load_profiles = pd.read_csv(DATA_PATH + "load_profiles.csv", index_col=0)


def run_one(methode, mois, seed, export_profile=True):
    cfg = {
        "methode_affectation": methode,
        "mois_simule": mois,
        "seed": seed,
        "export_aggregate_profile": export_profile,
    }
    t0 = time.time()
    out = run_digital_twin(cfg, households, load_profiles)
    elapsed = time.time() - t0

    summary = out[out["category"] == "summary"].set_index("metric")["value"]
    agg_profile_kw = None
    if export_profile:
        prof_rows = out[out["category"] == "profile_agg"].sort_values("timestep")
        # `out['value']` is object-dtype (mixed bool/float/str/None across
        # categories) -- cast explicitly or np.allclose/isfinite chokes on it.
        agg_profile_kw = prof_rows["value"].to_numpy(dtype=float) * 1000.0  # MW -> kW
    n_violations = int((out["category"] == "violation").sum())
    return summary, agg_profile_kw, n_violations, elapsed


# ============================================================
# Etape 1 : baseline "actuelle" (hiver + ete, seed fixee)
# Etape 2/3 : comparaison actuelle vs banque_mensuelle (meme seed, memes mois)
# ============================================================
print(f"\n=== Etapes 1-3 : comparaison actuelle vs banque_mensuelle (seed={BASE_SEED}) ===")
rows = []
aggs = {}       # (month, method) -> profil agrege [kW], 672 pas
for month in MONTHS:
    for methode in METHODS:
        summary, agg_kw, n_violations, elapsed = run_one(methode, month, BASE_SEED)
        print(f"  {MONTH_LABEL[month]:<16} | {METHOD_LABEL[methode]:<32} | "
              f"{elapsed:5.1f}s | pointe={summary['peak_agg_kw']:.1f} kW | "
              f"FC={summary['load_factor_agg']:.3f} | "
              f"Vmin={summary['min_voltage_pu']:.4f} pu | "
              f"charge ligne max={summary['max_line_loading_pct']:.1f}% | "
              f"violations={n_violations}")
        row = {"month": MONTH_LABEL[month], "methode": methode, "n_violations": n_violations}
        row.update(summary.to_dict())
        rows.append(row)
        aggs[(month, methode)] = agg_kw

results = pd.DataFrame(rows)
results_path = os.path.join(OUT_DIR, "comparison_table.csv")
results.to_csv(results_path, index=False)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n=== Tableau comparatif (etapes 1-3) ===")
print(results.to_string(index=False))
print(f"\nExporte : {results_path}")

same_baseline = np.allclose(aggs[(WINTER_MONTH, "actuelle")], aggs[(SUMMER_MONTH, "actuelle")])
print(f"\n(!) Baseline 'actuelle' identique hiver/ete : {same_baseline} "
      f"(attendu True -- la methode actuelle ne segmente pas par mois, "
      f"c'est precisement ce que banque_mensuelle corrige).")


# ============================================================
# Graphiques
# ============================================================
# --- 1. Profil agrege superpose, par mois ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, month in zip(axes, MONTHS):
    for methode in METHODS:
        ax.plot(aggs[(month, methode)], color=COLORS[methode], linewidth=1.1,
                label=METHOD_LABEL[methode])
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(days, fontsize=8, rotation=45)
    ax.set_title(MONTH_LABEL[month])
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Puissance agregee injectee [kW]")
fig.suptitle(f"Profil agrege injecte : actuelle vs banque_mensuelle (seed={BASE_SEED})", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_aggregate_profile_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 2. Metriques reseau : tension min et charge ligne max ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(MONTHS))
width = 0.35
for i, methode in enumerate(METHODS):
    vals = [results[(results.month == MONTH_LABEL[m]) & (results.methode == methode)]["min_voltage_pu"].iloc[0]
            for m in MONTHS]
    axes[0].bar(x + (i - 0.5) * width, vals, width, color=COLORS[methode], label=METHOD_LABEL[methode])
axes[0].axhline(0.95, color="#444444", linestyle="--", linewidth=1, label="Seuil v_min (0.95 pu)")
axes[0].set_xticks(x)
axes[0].set_xticklabels([MONTH_LABEL[m] for m in MONTHS])
axes[0].set_ylabel("Tension min [pu]")
axes[0].set_title("Tension minimale")
axes[0].legend(frameon=False, fontsize=8)
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(axis="y", alpha=0.25)

for i, methode in enumerate(METHODS):
    vals = [results[(results.month == MONTH_LABEL[m]) & (results.methode == methode)]["max_line_loading_pct"].iloc[0]
            for m in MONTHS]
    axes[1].bar(x + (i - 0.5) * width, vals, width, color=COLORS[methode], label=METHOD_LABEL[methode])
axes[1].axhline(90.0, color="#444444", linestyle="--", linewidth=1, label="Seuil loading_max (90%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels([MONTH_LABEL[m] for m in MONTHS])
axes[1].set_ylabel("Charge ligne max [%]")
axes[1].set_title("Charge de ligne maximale")
axes[1].legend(frameon=False, fontsize=8)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(axis="y", alpha=0.25)

fig.suptitle("Grandeurs reseau : actuelle vs banque_mensuelle", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_network_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Test de robustesse au bruit de tirage (5-10 seeds, mois dimensionnant)
# ============================================================
print(f"\n=== Robustesse : {len(ROBUSTNESS_SEEDS)} seeds par methode, "
      f"{MONTH_LABEL[ROBUSTNESS_MONTH]} ===")
robustness_rows = []
for methode in METHODS:
    for seed in ROBUSTNESS_SEEDS:
        summary, _, n_violations, elapsed = run_one(methode, ROBUSTNESS_MONTH, seed, export_profile=False)
        robustness_rows.append({
            "methode": methode, "seed": seed, "n_violations": n_violations,
            "peak_agg_kw": summary["peak_agg_kw"], "load_factor_agg": summary["load_factor_agg"],
            "min_voltage_pu": summary["min_voltage_pu"], "max_line_loading_pct": summary["max_line_loading_pct"],
        })
        print(f"  {METHOD_LABEL[methode]:<32} seed={seed} | {elapsed:4.1f}s | "
              f"pointe={summary['peak_agg_kw']:.1f} kW | Vmin={summary['min_voltage_pu']:.4f} pu")

robustness = pd.DataFrame(robustness_rows)
robustness_path = os.path.join(OUT_DIR, "robustness_runs.csv")
robustness.to_csv(robustness_path, index=False)

robustness_summary = robustness.groupby("methode")[
    ["peak_agg_kw", "load_factor_agg", "min_voltage_pu", "max_line_loading_pct"]
].agg(["mean", "std"])
robustness_summary_path = os.path.join(OUT_DIR, "robustness_summary.csv")
robustness_summary.to_csv(robustness_summary_path)

print(f"\n=== Robustesse : moyenne +/- ecart-type sur {len(ROBUSTNESS_SEEDS)} seeds ===")
print(robustness_summary.to_string())
print(f"\nExportes : {robustness_path}, {robustness_summary_path}")


# ============================================================
# Resume ecrit + recommandation
# ============================================================
row_act_w = results[(results.month == MONTH_LABEL[WINTER_MONTH]) & (results.methode == "actuelle")].iloc[0]
row_bank_w = results[(results.month == MONTH_LABEL[WINTER_MONTH]) & (results.methode == "banque_mensuelle")].iloc[0]

peak_gap_pct = (row_bank_w["peak_agg_kw"] / row_act_w["peak_agg_kw"] - 1) * 100
v_gap = row_bank_w["min_voltage_pu"] - row_act_w["min_voltage_pu"]
load_gap = row_bank_w["max_line_loading_pct"] - row_act_w["max_line_loading_pct"]

act_violations_total = results[results.methode == "actuelle"]["n_violations"].sum()
bank_violations_total = results[results.methode == "banque_mensuelle"]["n_violations"].sum()

std_peak_act = robustness_summary.loc["actuelle", ("peak_agg_kw", "std")]
std_peak_bank = robustness_summary.loc["banque_mensuelle", ("peak_agg_kw", "std")]
mean_peak_act = robustness_summary.loc["actuelle", ("peak_agg_kw", "mean")]
mean_peak_bank = robustness_summary.loc["banque_mensuelle", ("peak_agg_kw", "mean")]
gap_exceeds_noise = abs(mean_peak_bank - mean_peak_act) > (std_peak_act + std_peak_bank)

summary_text = f"""=== Resume integrate-bank-B ===

Comparaison actuelle vs banque_mensuelle (seed={BASE_SEED}, {MONTH_LABEL[WINTER_MONTH]}) :
  - Pointe agregee : {row_act_w['peak_agg_kw']:.1f} kW (actuelle) vs {row_bank_w['peak_agg_kw']:.1f} kW
    (banque_mensuelle), soit {peak_gap_pct:+.1f}%.
  - Tension min : {row_act_w['min_voltage_pu']:.4f} pu vs {row_bank_w['min_voltage_pu']:.4f} pu
    (delta {v_gap:+.4f} pu).
  - Charge ligne max : {row_act_w['max_line_loading_pct']:.1f}% vs {row_bank_w['max_line_loading_pct']:.1f}%
    (delta {load_gap:+.1f} pts).
  - Violations totales (2 mois cumules) : {act_violations_total} (actuelle) vs
    {bank_violations_total} (banque_mensuelle).

Robustesse (pointe agregee, {len(ROBUSTNESS_SEEDS)} seeds, {MONTH_LABEL[ROBUSTNESS_MONTH]}) :
  actuelle          : {mean_peak_act:.1f} +/- {std_peak_act:.1f} kW
  banque_mensuelle  : {mean_peak_bank:.1f} +/- {std_peak_bank:.1f} kW
  Ecart methode > dispersion de tirage : {gap_exceeds_noise}

Est-ce que banque_mensuelle change les CONCLUSIONS reseau (violations,
marges) ou seulement les profils en amont ?
  -> Voir n_violations et marges (min_voltage_pu / max_line_loading_pct)
     ci-dessus : si les deux methodes restent 'network_valid' (aucune
     violation) avec une marge de tension/charge du meme ordre de grandeur,
     la difference reste surtout un effet sur les PROFILS en amont (pointe,
     facteur de charge), pas sur le verdict de dimensionnement au regime
     nominal teste ici. Une hausse de pointe qui rapproche ou depasse les
     seuils (v_min=0.95pu, loading_max=90%) changerait la conclusion.

Recommandation : voir le detail chiffre ci-dessus (comparison_table.csv,
robustness_summary.csv) avant de basculer le defaut de methode_affectation
vers 'banque_mensuelle' -- notamment verifier que l'ecart de pointe/tension
depasse bien la dispersion de tirage (ligne 'Ecart methode > dispersion').
"""
print("\n" + summary_text)
with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
    f.write(summary_text)
print(f"Resume ecrit : {os.path.join(OUT_DIR, 'SUMMARY.md')}")
