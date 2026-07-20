"""
Volet PV -- Brique 2 : controle visuel production PV, ete vs hiver.

Profil de production d'une installation type (3 kWp) sur une journee, pour
un mois d'ete (juin) et un mois d'hiver (janvier) superposes, pour
visualiser l'ecart saisonnier. Backend Agg, PNG dans outputs_pv_production/.

Script de controle, hors pipeline de production : lecture seule.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sur (savefig fiable, cf. soucis machine recurrents)
import matplotlib.pyplot as plt

from sourcecode.pv_production import (
    build_pv_production, STEPS_PER_DAY, PV_INSTALLATIONS_PATH, IRRADIANCE_PATH,
)

OUT_DIR = "outputs_pv_production"
DAY_INDEX = 0  # premier jour (lundi) de la semaine representative de chaque mois

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLORS = {"juin": "#E69F00", "janvier": "#0072B2"}


def build_figure():
    pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
    irradiance = pd.read_parquet(IRRADIANCE_PATH)

    res_bus = int(pv_installations.loc[pv_installations["type"] == "residential", "bus_id"].iloc[0])
    col = f"pv_{res_bus}"

    t = np.arange(STEPS_PER_DAY) * 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mois in ("juin", "janvier"):
        prod = build_pv_production(mois, pv_installations, irradiance)
        day_vals = prod[col].to_numpy()[DAY_INDEX * STEPS_PER_DAY:(DAY_INDEX + 1) * STEPS_PER_DAY]
        energy = day_vals.sum() * 0.25
        ax.plot(t, day_vals, color=COLORS[mois], lw=2.2,
                label=f"{mois.capitalize()} — {energy:.1f} kWh/jour (pic {day_vals.max():.2f} kW)")

    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 4)])
    ax.set_xlabel("Heure de la journée (UTC)")
    ax.set_ylabel("Puissance produite [kW]")
    ax.set_title(f"Production PV — installation résidentielle {res_bus} (3 kWp) : été vs hiver", loc="left", fontsize=13)
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = build_figure()
    out_path = os.path.join(OUT_DIR, "pv_production_summer_vs_winter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure ecrite : {out_path}")
