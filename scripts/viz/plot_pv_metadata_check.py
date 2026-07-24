"""
Volet PV -- Brique 1 : controle visuel du placement des installations PV.

Scatter des 333 bus residentiels reels (coordonnees issues de la
reconstruction reseau, cf. sourcecode/digital_twin._build_network) : bus non
equipes en gris clair, bus equipes PV colores par rue (pour verifier l'effet
grappe), maison de repos marquee a part. Backend Agg, sortie PNG dans
outputs_pv_metadata/.

Prerequis : data/pv_installations.csv doit exister (executer
`python -m sourcecode.pv_metadata` sinon).

Script de controle, hors pipeline de production : ne modifie pas
digital_twin.py, lecture seule.
"""
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sur (savefig fiable, cf. soucis machine recurrents)
import matplotlib.pyplot as plt

from sourcecode.digital_twin import _build_network, _config_to_dict

OUT_DIR = "outputs_pv_metadata"
PV_CSV = "data/pv_installations.csv"
REST_HOME_NAME = "Paradis du Bouhay"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
})


def _bus_coords_and_streets():
    """dict bus_id -> (lon, lat, street) pour les 333 bus residentiels,
    plus les coords du bus maison de repos, obtenus depuis le VRAI reseau
    reconstruit (pas une re-derivation independante depuis coordonnées.csv,
    pour garantir la coherence bus_id <-> position)."""
    cfg = _config_to_dict(None)
    net, buses, mv_nodes, validation, _ = _build_network(cfg)

    coords = {}
    for (street, num), b in buses.items():
        geo = json.loads(net.bus.loc[b, "geo"])
        lon, lat = geo["coordinates"]
        coords[int(b)] = (lon, lat, street)

    rest_home_bus = int(mv_nodes[REST_HOME_NAME])
    geo = json.loads(net.bus.loc[rest_home_bus, "geo"])
    rest_home_coords = tuple(geo["coordinates"])
    return coords, rest_home_bus, rest_home_coords


def build_figure():
    coords, rest_home_bus, rest_home_coords = _bus_coords_and_streets()
    pv = pd.read_csv(PV_CSV)
    pv_res_buses = set(pv.loc[pv["type"] == "residential", "bus_id"].astype(int))

    fig, ax = plt.subplots(figsize=(10, 10))

    xs_all = [c[0] for c in coords.values()]
    ys_all = [c[1] for c in coords.values()]
    ax.scatter(xs_all, ys_all, color="#d9d9d9", s=30, label=f"No PV ({len(coords) - len(pv_res_buses)})", zorder=1)

    streets_pv = sorted({coords[b][2] for b in pv_res_buses})
    cmap = plt.get_cmap("tab10")
    for i, street in enumerate(streets_pv):
        bus_ids = [b for b in pv_res_buses if coords[b][2] == street]
        xs = [coords[b][0] for b in bus_ids]
        ys = [coords[b][1] for b in bus_ids]
        ax.scatter(xs, ys, color=cmap(i % 10), s=60, label=f"{street} ({len(bus_ids)})", zorder=2)

    ax.scatter([rest_home_coords[0]], [rest_home_coords[1]], color="black", marker="*",
               s=350, label="Rest home PV", zorder=3, edgecolors="white", linewidths=0.8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("PV placement check — residential clusters by street", loc="left")
    ax.legend(frameon=True, fontsize=10, loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    if not os.path.exists(PV_CSV):
        raise FileNotFoundError(f"{PV_CSV} introuvable -- executer `python -m sourcecode.pv_metadata` d'abord.")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = build_figure()
    out_path = os.path.join(OUT_DIR, "pv_placement_check.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure ecrite : {out_path}")
