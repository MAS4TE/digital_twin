"""
Volet BSS -- controle visuel des profils du parc de batteries.

Trois figures, sans aucun power flow : tout est reconstruit depuis process_data
(charges, PV, planning BSS) et _track_bss_soc. Compter quelques secondes, pas
les ~3 min d'un run complet.

    1. Semaine complete   : charge reseau, PV, parc BSS, net vu du transformateur
    2. Parc BSS + SoC     : deux panneaux empilés (jamais deux axes y superposés)
    3. Profil journalier  : les 7 jours superposés + la médiane, charge et parc

Pourquoi les internes plutot que `out` : run_digital_twin ne retourne que des
scalaires (summary), les violations et les warnings -- les series temporelles
calculees dans _run_powerflow sont consommees par _evaluate puis perdues. Les
quatre autres scripts de scripts/viz/ contournent de la meme facon.

Script de controle, hors pipeline de production : lecture seule, n'ecrit qu'un PNG.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sur (savefig fiable, cf. soucis machine recurrents)
import matplotlib.pyplot as plt

from digital_twin.data_processing.process_data import process_data
from digital_twin.digital_twin import _BSS_KW_TO_MW, _assign_loads, _track_bss_soc
from digital_twin.utils import STEPS_PER_DAY, STEPS_PER_WEEK, _config_to_dict

OUT_PATH = "scripts/viz/bss_profiles.png"

CONFIG = {
    'data_path': "data/bressoux",
    'n_timesteps': STEPS_PER_WEEK,
    'trafo_sn_mva': 0.22,
    'cos_phi': 0.95,
    'seed': 42,
}

DAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# Slots 1-3 de la palette categorielle de reference, dans l'ordre. Valides
# (portage Python du validateur de la skill dataviz, node indisponible ici) :
# toutes paires, mode clair, surface blanche -> CVD dE 9.2 (>=8), vision normale
# dE 24.0 (>=15). L'aqua est a 2.82:1 sur blanc, sous le seuil de 3:1 : la regle
# de relief s'applique, d'ou les etiquettes directes en bout de courbe.
C_LOAD = "#2a78d6"   # slot 1, bleu
C_PV = "#eb6834"     # slot 2, orange
C_BSS = "#1baf7a"    # slot 3, aqua
INK = "#0b0b0b"      # encre primaire -- le net est une resultante, pas une categorie
MUTED = "#898781"    # axes / etiquettes
GRID = "#e1e0d9"


def load_series():
    """Toutes les series en kW, sur 672 pas, alignees positionnellement.

    La charge vient de _assign_loads et NON de households.sum() : households est
    le POOL de profils (333 colonnes, 48-110 kW), pas la charge du reseau. Le
    tirage profil -> bus et les 4 charges MV n'existent qu'apres l'affectation,
    d'ou les ~199 kW de pointe reelle.
    """
    cfg = _config_to_dict(CONFIG)
    metadata, network, households, pv_profiles, schedules_BSS = process_data(CONFIG)
    if schedules_BSS is None:
        raise SystemExit(
            "Pas de schedules_BSS pour ce scenario : generer inputs/bss_shares.csv "
            "via digital_twin.data_processing.bss.generate_bss_shares d'abord."
        )

    df_p, _, _, _ = _assign_loads(network, metadata, households, cfg)
    load_kw = df_p.sum(axis=1).to_numpy() * 1000.0
    pv_kw = (pv_profiles.sum(axis=1).to_numpy() * 1000.0
             if pv_profiles is not None else np.zeros_like(load_kw))

    # SoC et serie REELLEMENT appliquee : _track_bss_soc peut ecreter, et c'est
    # la version ecretee que le twin injecte. Tracer le planning brut donnerait
    # une figure que le power flow ne verrait jamais.
    bss_raw_mw = schedules_BSS.sum(axis=1).to_numpy() * _BSS_KW_TO_MW
    soc, p_applied_mw, _ = _track_bss_soc(bss_raw_mw, cfg)
    bss_kw = p_applied_mw / _BSS_KW_TO_MW

    # Signe : BSS positif = charge = consommation, donc il S'AJOUTE au net,
    # la ou le PV se retranche (cf. _inject_bss_storage).
    net_kw = load_kw - pv_kw + bss_kw
    return load_kw, pv_kw, bss_kw, net_kw, soc * 100.0


def _label_peak(ax, y, text, color):
    """Etiquette directe posee sur le maximum de la courbe.

    Pas en bout de courbe : les quatre series y convergent vers 0 le dimanche
    soir et les etiquettes se chevauchaient. Sur le pic, chacune est isolee.
    Sert aussi la regle de relief -- l'aqua est a 2.82:1 sur blanc.
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

    # --- 1. La semaine ---------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axhline(0, color=MUTED, lw=1.0, zorder=1)
    ax1.plot(x, load, color=C_LOAD, lw=2.0, label="Charge réseau", zorder=3)
    ax1.plot(x, pv, color=C_PV, lw=2.0, label="Production PV", zorder=3)
    ax1.plot(x, bss, color=C_BSS, lw=2.0, label="Parc BSS (+ charge / − décharge)", zorder=3)
    ax1.plot(x, net, color=INK, lw=1.6, ls="--", label="Net vu du transformateur", zorder=4)
    _label_peak(ax1, bss, "Parc BSS", C_BSS)
    ax1.margins(y=0.16)
    _day_axis(ax1, n)
    ax1.set_ylabel("Puissance [kW]", color=MUTED)
    ax1.set_title(
        "Semaine simulée — le parc (±595 kW) domine largement la charge du quartier (pointe 199 kW)",
        loc="left", fontsize=13, color=INK)
    ax1.legend(frameon=False, fontsize=10, ncol=4, loc="upper left",
               bbox_to_anchor=(0, -0.10))

    # --- 2. Parc BSS puis SoC : DEUX PANNEAUX, jamais deux axes y ---------
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axhline(0, color=MUTED, lw=1.0)
    ax2.fill_between(x, 0, bss, where=bss >= 0, color=C_BSS, alpha=0.30, lw=0)
    ax2.fill_between(x, 0, bss, where=bss < 0, color=C_LOAD, alpha=0.30, lw=0)
    ax2.plot(x, bss, color=C_BSS, lw=1.6)
    _day_axis(ax2, n)
    ax2.set_ylabel("Parc BSS [kW]", color=MUTED)
    ax2.set_title("Puissance du parc — aire verte : charge, aire bleue : décharge",
                  loc="left", fontsize=12, color=INK)

    ax3 = fig.add_subplot(gs[2, :], sharex=ax2)
    ax3.axhspan(10, 100, color=GRID, alpha=0.55, lw=0, zorder=1)
    ax3.plot(x, soc, color=C_LOAD, lw=2.0, zorder=3)
    ax3.axhline(10, color=MUTED, lw=1.0, ls=":", zorder=2)
    _day_axis(ax3, n)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("SoC [%]", color=MUTED)
    ax3.set_title(f"État de charge — bande utile 10–100 %, plancher touché à "
                  f"{soc.min():.0f} %, fin de semaine {soc[-1]:.0f} %",
                  loc="left", fontsize=12, color=INK)

    # --- 3. Profil journalier : 7 jours superposes + mediane --------------
    n_days = n // STEPS_PER_DAY
    hours = np.arange(STEPS_PER_DAY) * 24.0 / STEPS_PER_DAY
    for col, (serie, titre, couleur) in enumerate(
            ((load, "Charge réseau", C_LOAD), (bss, "Parc BSS", C_BSS))):
        ax = fig.add_subplot(gs[3, col])
        days = serie[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        for d in range(n_days):
            ax.plot(hours, days[d], color=MUTED, lw=0.9, alpha=0.45, zorder=2)
        med = np.median(days, axis=0)
        ax.plot(hours, med, color=couleur, lw=2.4, zorder=3, label="Médiane")
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        # Annotation posee DU COTE INTERIEUR de l'extremum (au-dessus d'un
        # minimum, au-dessous d'un maximum) : l'inverse la poussait hors des
        # axes, sur le titre a gauche et sous les ticks a droite.
        k = int(np.argmax(np.abs(med)))
        ax.margins(y=0.18)
        ax.annotate(f"{med[k]:+.0f} kW à {int(hours[k]):02d}h{int(hours[k] % 1 * 60):02d}",
                    xy=(hours[k], med[k]), xytext=(0, 12 if med[k] < 0 else -20),
                    textcoords="offset points", ha="center", fontsize=10, color=INK)
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 6)])
        ax.tick_params(colors=MUTED)
        ax.grid(color=GRID)
        ax.set_xlabel("Heure", color=MUTED)
        if col == 0:
            ax.set_ylabel("Puissance [kW]", color=MUTED)
        ax.set_title(f"{titre} — {n_days} jours superposés (gris) et médiane",
                     loc="left", fontsize=12, color=INK)
        ax.legend(frameon=False, fontsize=10, loc="best")

    fig.suptitle("Digital twin Bressoux — profils du parc de batteries centralisé",
                 x=0.012, ha="left", fontsize=15, color=INK)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Figure ecrite : {OUT_PATH}")
