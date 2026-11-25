"""
Power-flow results: where and when the network breaks its voltage and loading
limits. Four figures written to digital_twin/plots/:

    powerflow_map_<scn>.png         -- pandapower-style result plot at the worst
                                       time step of the week, on an OpenStreetMap
                                       basemap: buses by voltage [0.9, 1.1] pu,
                                       LV lines by loading [0, 100] % (jet),
                                       transformers as grey squares.
    powerflow_week_<scn>.png        -- the week, three stacked panels on one
                                       time axis: voltage envelope over all LV
                                       buses, line loading envelope, and each
                                       MV/LV transformer's loading.
    powerflow_feeders_<scn>.png     -- voltage drop along the feeders at the
                                       worst time step: bus voltage against the
                                       cable distance from its transformer.
    powerflow_scenarios.png         -- low / medium / high compared: duration
                                       curves of the worst voltage and the worst
                                       line loading, and the share of buses /
                                       lines that break a limit at least once.
                                       Replaces the per-scenario bar charts of
                                       bressoux_setup.ipynb (one scalar each).

LIMITS are the engine's own (config v_min / v_max / loading_max, merged by
utils._config_to_dict), so a red element here is exactly a violation reported by
run_digital_twin -- not a looser or stricter rule drawn for the picture.

Why the engine internals rather than run_digital_twin: it returns only scalars,
violations and warnings; the (n_timesteps x n_element) series of _run_powerflow
are consumed by _evaluate and dropped. run_scenario() below chains the same
internal steps in the same order as run_digital_twin (validation included), and
keeps them. Same approach as plot_bss_profiles.py.

Check script, outside the production pipeline: read-only over the scenario
folders, writes PNGs only. One full week power flow is ~1-3 min per scenario.

    python -m digital_twin.plots.plot_powerflow_results            # all figures
    python -m digital_twin.plots.plot_powerflow_results --only high
"""
import argparse
import copy
import json
import os
import tempfile

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # headless backend, so savefig does not need a display
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D

from digital_twin.data_processing.process_data import process_data
from digital_twin.digital_twin import (_KW_TO_MW, _assign_loads, _bss_battery_ids,
                                       _bss_fleet_specs, _inject_bss_storage,
                                       _inject_pv_sgens, _read_bss_schedule,
                                       _run_powerflow, _track_bss_soc,
                                       _validate_inputs, _verify_temporal_alignment)
from digital_twin.utils import STEPS_PER_DAY, STEPS_PER_WEEK, _config_to_dict

OUT_DIR = "digital_twin/plots"

SCENARIOS = {  # ordered low -> high; built by data/bressoux/bressoux_setup.ipynb
    "low": "data/bressoux/scenarios/low",
    "medium": "data/bressoux/scenarios/medium",
    "high": "data/bressoux/scenarios/high",
}
FOCUS = "high"   # scenario drawn in detail (map, week, feeders)

# Run-time keys only, as bressoux_setup.ipynb section 9 passes them.
BASE_CONFIG = {
    "seed": 42,
    "mois_simule": 6,
    "trafo_sn_mva": 0.22,
    "n_timesteps": STEPS_PER_WEEK,
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- Style: reference palette (dataviz skill, references/palette.md) -------
INK, INK2 = "#191919", "#666666"
GRID, AXIS = "#e7e7e7", "#c2c2c2"
CRITICAL = "#dc3c3d"                       # status: limit violated
NEUTRAL = "#efefef"                        # diverging midpoint
# Categorical slots 1-4, fixed order: the four MV/LV transformers.
CATEGORICAL = ["#257de0", "#f47230", "#00b894", "#f2a600"]
# Ordinal blue ramp (>= step 250 on white): low / medium / high.
SCENARIO_COLORS = {"low": "#84b9f3", "medium": "#2370cd", "high": "#0e3d7e"}
BLUE_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
# Diverging: red (under-voltage) <- gray (1.0 pu) -> blue (over-voltage)
RED_ARM = ["#dc3c3d", "#ec8586", "#f6c3c3"]
BLUE_ARM = ["#b6d5f8", "#6aabf1", "#1b64be"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})


# ============================================================
#  Simulation: run_digital_twin's steps, keeping the time series
# ============================================================
def run_scenario(data_path, config=None):
    """Full week power flow of one scenario folder.

    Returns
    -------
    dict with
        cfg, network       -- merged config, network bundle (net carries results)
        vm                 -- DataFrame (steps x bus) [pu]
        line_load          -- DataFrame (steps x line) [%]
        trafo_load         -- DataFrame (steps x trafo) [%]
        warnings           -- the engine's warnings for this run
    """
    cfg = _config_to_dict({**BASE_CONFIG, **(config or {}), "data_path": data_path})
    metadata, network, households, pv, bss, equipment = process_data(cfg)
    network = {**network, "net": copy.deepcopy(network["net"])}

    warnings = _validate_inputs(cfg, metadata, network, households, pv, bss, equipment)
    df_p, df_q, _, w = _assign_loads(network, metadata, households, cfg)
    warnings += w
    net = network["net"]

    df_pv_p = None
    if pv is not None and len(pv) > 0:
        df_pv_p = pv.reset_index(drop=True) * _KW_TO_MW
        _verify_temporal_alignment(df_p, df_pv_p)
        _inject_pv_sgens(net, df_pv_p)

    df_bss_p = None
    if bss is not None and len(bss) > 0:
        specs = _bss_fleet_specs(cfg, equipment, _bss_battery_ids(bss))
        df_bss_p, w = _read_bss_schedule(bss, df_p, cfg, specs,
                                         load_time_index=households.index)
        warnings += w
        _, df_bss_p, w = _track_bss_soc(df_bss_p, specs, cfg)
        warnings += w
        _inject_bss_storage(net, specs, cfg)

    pf = _run_powerflow(net, df_p, df_q, cfg, df_pv_p=df_pv_p, df_bss_p=df_bss_p)
    return {"cfg": cfg, "network": network, "vm": pf["vm"],
            "line_load": pf["line_load"], "trafo_load": pf["trafo_load"],
            "warnings": warnings}


# ============================================================
#  Topology helpers
# ============================================================
def _lv_buses(network):
    return sorted(int(b) for b in network["buses"].values())


def _bus_xy(net):
    """{bus_id: (lon, lat)} for every bus carrying geodata."""
    xy = {}
    for b, geo in net.bus["geo"].items():
        if isinstance(geo, str):
            xy[int(b)] = tuple(json.loads(geo)["coordinates"])
    return xy


def _feeder_distances(net):
    """Per LV bus: (transformer index, cable distance [m] from its LV terminal).

    Shortest path over the in-service LINES only, from each transformer's LV bus
    -- radial network, so this is the feeder path.
    """
    g = nx.Graph()
    for _, ln in net.line[net.line.in_service].iterrows():
        g.add_edge(int(ln.from_bus), int(ln.to_bus), w=ln.length_km * 1000.0)
    owner, dist = {}, {}
    for t, tr in net.trafo.iterrows():
        lv = int(tr.lv_bus)
        if lv not in g:
            continue
        for b, d in nx.single_source_dijkstra_path_length(g, lv, weight="w").items():
            if b not in dist or d < dist[b]:
                owner[b], dist[b] = int(t), d
    return owner, dist


def _trafo_label(net, t):
    return str(net.trafo.loc[t, "name"]).replace("trafo_", "").replace("_", " ").title()


def _week_axis(ax):
    ax.set_xlim(0, STEPS_PER_WEEK)
    ax.set_xticks(np.arange(0, STEPS_PER_WEEK + 1, STEPS_PER_DAY))
    ax.set_xticklabels([])
    for d in range(7):
        ax.text((d + 0.5) * STEPS_PER_DAY, -0.02, DAYS[d], transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK2, fontsize=9)
    ax.grid(axis="x", color=GRID)


def _legend_above(ax, **kw):
    """Legend in one row above the panel, right-aligned: never over the data."""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=9,
              borderaxespad=0.2, **kw)


def _violation_spans(ax, mask, color=CRITICAL, alpha=0.16):
    """Shade the steps where `mask` is True (contiguous runs merged)."""
    mask = np.asarray(mask, dtype=bool)
    k = 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j < len(mask) and mask[j]:
                j += 1
            ax.axvspan(k, j, color=color, alpha=alpha, lw=0, zorder=0)
            k = j
        else:
            k += 1


# ============================================================
#  Figure 1 -- the map, on an OpenStreetMap basemap
# ============================================================
# Feeders kept OUT of the map frame (still drawn, just off-screen): Cimetiere
# Bressoux sits ~500 m south of the rest and carries no violation in any
# scenario, so framing it would shrink the stressed area to a strip. Its
# transformer is pointed at from the frame edge instead. Counts in the legends
# stay NETWORK-WIDE either way. Set to () to frame the whole LV network.
MAP_FRAME_EXCLUDE = ("trafo_CIMETIERE_BRESSOUX",)

# OpenStreetMap standard tiles, MUTED in _basemap (mostly desaturated and
# lightened) so the data carries the colour. Keyless providers of pale tiles
# (CARTO light, Esri light gray) now watermark or stop below this zoom. Tiles
# are fetched once and cached in the system temp dir, per the OSM tile usage
# policy; when unreachable the map is drawn without a basemap.
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = "© OpenStreetMap contributors"
TILE_MAX_ZOOM = 18
TILE_SATURATION = 0.25   # 0 = grayscale, 1 = original colours
TILE_FADE = 0.55         # share of the original contrast kept (1 = untouched)
TILE_CACHE = os.path.join(tempfile.gettempdir(), "mas4te_basemap_tiles")
_MERC_R = 6378137.0
_MERC_E = np.pi * _MERC_R      # half the Web Mercator world width [m]


def _to_merc(lon, lat):
    """lon/lat [deg] -> Web Mercator x/y [m] (EPSG:3857), the tiles' own frame."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return (_MERC_R * np.deg2rad(lon),
            _MERC_R * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2)))


def _basemap(ax, x0, x1, y0, y1, max_tiles=60):
    """Draw the tiles covering [x0, x1] x [y0, y1] (Web Mercator) under the data.

    Returns True when drawn, False when the tiles could not be fetched.
    """
    import io
    import urllib.request
    from PIL import Image

    for z in range(TILE_MAX_ZOOM, 10, -1):  # finest zoom that stays under max_tiles
        size = 2 * _MERC_E / 2 ** z
        tx0, tx1 = int((x0 + _MERC_E) // size), int((x1 + _MERC_E) // size)
        ty0, ty1 = int((_MERC_E - y1) // size), int((_MERC_E - y0) // size)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles:
            break
    os.makedirs(TILE_CACHE, exist_ok=True)
    rows = []
    try:
        for ty in range(ty0, ty1 + 1):
            row = []
            for tx in range(tx0, tx1 + 1):
                path = os.path.join(TILE_CACHE, f"{z}_{tx}_{ty}.png")
                if not os.path.exists(path):
                    req = urllib.request.Request(TILE_URL.format(z=z, x=tx, y=ty),
                                                 headers={"User-Agent": "mas4te-twin-plots/1.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp, open(path, "wb") as f:
                        f.write(resp.read())
                row.append(np.asarray(Image.open(path).convert("RGB")))
            rows.append(np.concatenate(row, axis=1))
    except OSError as exc:
        print(f"Basemap skipped (tiles unreachable: {exc})")
        return False
    img = np.concatenate(rows, axis=0).astype(float)
    gray = img @ np.array([0.299, 0.587, 0.114])
    img = TILE_SATURATION * img + (1 - TILE_SATURATION) * gray[..., None]
    img = (255 - (255 - img) * TILE_FADE).clip(0, 255).astype(np.uint8)
    ax.imshow(img, extent=(-_MERC_E + tx0 * size, -_MERC_E + (tx1 + 1) * size,
                           _MERC_E - (ty1 + 1) * size, _MERC_E - ty0 * size),
              interpolation="lanczos", zorder=0)
    ax.text(0.995, 0.004, TILE_ATTRIBUTION, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK2, zorder=10,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.7))
    return True


def worst_step(res):
    """Time step with the lowest LV bus voltage (in the Bressoux scenarios it is
    also the step of the highest line and transformer loading)."""
    lv = _lv_buses(res["network"])
    return int(np.argmin(res["vm"][lv].min(axis=1).to_numpy()))


def _step_label(k):
    day, step = divmod(int(k), STEPS_PER_DAY)
    return f"{DAYS[day]} {step * 15 // 60:02d}:{step * 15 % 60:02d}"


def figure_map(res, step=None, basemap=True):
    """pandapower-style result plot at ONE time step (default: the worst one).

    Standard pandapower collections (create_bus_collection /
    create_line_collection): buses coloured by voltage on [0.9, 1.1] pu, lines
    by loading on [0, 100] %, jet colormap, transformers as grey squares -- the
    usual pandapower result picture, drawn on a Web Mercator basemap.
    """
    import pandapower.plotting as ppplot

    cfg = res["cfg"]
    k = worst_step(res) if step is None else int(step)
    lv = _lv_buses(res["network"])

    # Copy of the net carrying this step's results, with bus geodata projected
    # to Web Mercator so collections and basemap tiles share one frame.
    net = copy.deepcopy(res["network"]["net"])
    net.res_bus["vm_pu"] = res["vm"].iloc[k].reindex(net.bus.index).to_numpy()
    net.res_line["loading_percent"] = res["line_load"].iloc[k].reindex(net.line.index).to_numpy()
    net.res_trafo["loading_percent"] = res["trafo_load"].iloc[k].reindex(net.trafo.index).to_numpy()
    xy = {}
    for b, (lon, lat) in _bus_xy(net).items():
        x, y = (float(c) for c in _to_merc(lon, lat))
        xy[b] = (x, y)
        net.bus.at[b, "geo"] = json.dumps({"type": "Point", "coordinates": [x, y]})

    # Frame: every feeder except MAP_FRAME_EXCLUDE (see its comment).
    owner, _ = _feeder_distances(net)
    hidden = {t for t in net.trafo.index if net.trafo.loc[t, "name"] in MAP_FRAME_EXCLUDE}
    framed = np.array([xy[b] for b in lv if owner.get(b) not in hidden]
                      + [xy[int(net.trafo.loc[t, "lv_bus"])] for t in net.trafo.index
                         if t not in hidden])
    pad = 0.05 * max(np.ptp(framed[:, 0]), np.ptp(framed[:, 1]))
    x0, x1 = framed[:, 0].min() - pad, framed[:, 0].max() + pad
    y0, y1 = framed[:, 1].min() - pad, framed[:, 1].max() + pad

    fs = 13
    panel_w = 10.5
    fig = plt.figure(figsize=(panel_w + 2.9, panel_w * (y1 - y0) / (x1 - x0)))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.025, 0.03, 0.13, 0.03], wspace=0)
    ax = fig.add_subplot(gs[0])
    cax_v, cax_l = fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
    if basemap:
        _basemap(ax, x0, x1, y0, y1)

    vnorm = Normalize(0.9, 1.1)
    lnorm = Normalize(0.0, 100.0)
    jet = plt.get_cmap("jet")

    # LV lines coloured by loading; the 15 kV links between substations as thin
    # dashed grey, as pandapower draws non-result connections.
    lv_side = set(lv) | set(net.trafo.lv_bus.astype(int))
    lv_lines = net.line.index[net.line.from_bus.isin(lv_side) & net.line.to_bus.isin(lv_side)]
    mv_lines = net.line.index.difference(lv_lines)
    lc = ppplot.create_line_collection(
        net, lines=lv_lines, use_bus_geodata=True, cmap=jet, norm=lnorm,
        linewidths=3.0, z=net.res_line.loc[lv_lines, "loading_percent"],
        plot_colormap=False, zorder=2)
    mvc = ppplot.create_line_collection(
        net, lines=mv_lines, use_bus_geodata=True, color="grey", linewidths=1.2,
        linestyles="dashed", plot_colormap=False, zorder=1)
    ax.add_collection(mvc)
    bc = ppplot.create_bus_collection(
        net, buses=lv, size=5.5, cmap=jet, norm=vnorm, z=net.res_bus.loc[lv, "vm_pu"],
        plot_colormap=False, zorder=3, linewidth=0.4, edgecolor="white")
    tc = ppplot.create_bus_collection(
        net, buses=net.trafo.lv_bus.astype(int).tolist(), size=13, patch_type="rect",
        color="grey", plot_colormap=False, zorder=4, linewidth=1.2, edgecolor="white")
    for c in (lc, bc, tc):
        ax.add_collection(c)

    for t, tr in net.trafo.iterrows():
        px, py = xy[int(tr.lv_bus)]
        if x0 <= px <= x1 and y0 <= py <= y1:
            ax.annotate(f"{_trafo_label(net, t)}  {net.res_trafo.at[t, 'loading_percent']:.0f} %",
                        (px, py), xytext=(12, 9), textcoords="offset points",
                        fontsize=fs - 1, color=INK, zorder=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(AXIS)

    ax.text(0.012, 0.985, f"{_step_label(k)} — worst time step of the week",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs, color=INK, zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9))

    # two vertical colorbars on the right, as pandapower draws them
    for cax, norm, label, extend in (
            (cax_v, vnorm, "Bus Voltage [pu]", "neither"),
            (cax_l, lnorm, "Line Loading [%]", "max")):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=jet), cax=cax,
                          extend=extend)
        cb.set_label(label, fontsize=fs, color=INK2)
        cb.ax.tick_params(labelsize=fs - 1, colors=INK2, length=0)
        cb.outline.set_edgecolor(AXIS)
    return fig


# ============================================================
#  Figure 2 -- the week
# ============================================================
def figure_week(res):
    cfg, net = res["cfg"], res["network"]["net"]
    lv = _lv_buses(res["network"])
    vm, ll, tl = res["vm"][lv], res["line_load"], res["trafo_load"]
    x = np.arange(len(vm))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)

    # voltage envelope
    ax = axes[0]
    vmin_t, vmax_t = vm.min(axis=1), vm.max(axis=1)
    p05, p95 = vm.quantile(0.05, axis=1), vm.quantile(0.95, axis=1)
    _violation_spans(ax, (vmin_t < cfg["v_min"]) | (vmax_t > cfg["v_max"]))
    ax.fill_between(x, vmin_t, vmax_t, color="#cce3fd", lw=0, label="All LV buses (min–max)")
    ax.fill_between(x, p05, p95, color="#6aabf1", lw=0, label="5th–95th percentile")
    ax.plot(x, vmin_t, color="#0e3d7e", lw=1.2, label="Lowest bus")
    ax.axhline(cfg["v_min"], color=CRITICAL, lw=1.2, ls="--")
    ax.text(STEPS_PER_WEEK - 2, cfg["v_min"], f"v_min {cfg['v_min']:.2f}", color=CRITICAL,
            ha="right", va="bottom", fontsize=9)
    k = int(np.argmin(vmin_t.to_numpy()))
    ax.annotate(f"{vmin_t.iloc[k]:.3f} p.u.", (k, vmin_t.iloc[k]), xytext=(8, -4),
                textcoords="offset points", color=INK, fontsize=9, va="top")
    ax.set_ylabel("Voltage [p.u.]")
    ax.set_title("Bus voltage", loc="left", fontsize=12)
    _legend_above(ax)
    ax.margins(y=0.12)

    # line loading envelope
    ax = axes[1]
    lmax_t = ll.max(axis=1)
    _violation_spans(ax, lmax_t > cfg["loading_max"])
    ax.fill_between(x, 0, ll.quantile(0.95, axis=1), color="#9dc8f7", lw=0,
                    label="95th percentile of lines")
    ax.plot(x, lmax_t, color="#0e3d7e", lw=1.2, label="Most loaded line")
    ax.axhline(cfg["loading_max"], color=CRITICAL, lw=1.2, ls="--")
    ax.text(STEPS_PER_WEEK - 2, cfg["loading_max"], f"limit {cfg['loading_max']:.0f} %",
            color=CRITICAL, ha="right", va="bottom", fontsize=9)
    k = int(np.argmax(lmax_t.to_numpy()))
    ax.annotate(f"{lmax_t.iloc[k]:.0f} %", (k, lmax_t.iloc[k]), xytext=(8, 0),
                textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_ylabel("Loading [%]")
    ax.set_ylim(0, None)
    ax.set_title("Line loading", loc="left", fontsize=12)
    _legend_above(ax)

    # transformer loading
    ax = axes[2]
    _violation_spans(ax, (tl > cfg["loading_max"]).any(axis=1))
    for i, t in enumerate(tl.columns):
        ax.plot(x, tl[t], color=CATEGORICAL[i % 4], lw=1.3, label=_trafo_label(net, t))
    ax.axhline(cfg["loading_max"], color=CRITICAL, lw=1.2, ls="--")
    ax.text(STEPS_PER_WEEK - 2, cfg["loading_max"], f"limit {cfg['loading_max']:.0f} %",
            color=CRITICAL, ha="right", va="bottom", fontsize=9)
    ax.set_ylabel("Loading [%]")
    ax.set_ylim(0, None)
    ax.set_title(f"MV/LV transformers ({cfg['trafo_sn_mva'] * 1000:.0f} kVA)",
                 loc="left", fontsize=12)
    _legend_above(ax)

    for ax in axes:
        _week_axis(ax)
    fig.tight_layout(h_pad=2.2)
    return fig


# ============================================================
#  Figure 3 -- voltage along the feeders at the worst step
# ============================================================
def figure_feeders(res):
    cfg, net = res["cfg"], res["network"]["net"]
    lv = _lv_buses(res["network"])
    vm = res["vm"]
    k = int(np.argmin(vm[lv].min(axis=1).to_numpy()))
    owner, dist = _feeder_distances(net)
    day, step = divmod(k, STEPS_PER_DAY)
    when = f"{DAYS[day]} {step * 15 // 60:02d}:{step * 15 % 60:02d}"

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axhline(cfg["v_min"], color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax.text(0.995, cfg["v_min"], f"v_min {cfg['v_min']:.2f}", color=CRITICAL, ha="right",
            va="bottom", fontsize=9, transform=ax.get_yaxis_transform())

    for i, t in enumerate(net.trafo.index):
        bs = [b for b in lv if owner.get(b) == t]
        if not bs:
            continue
        lvb = int(net.trafo.loc[t, "lv_bus"])
        d = np.array([0.0] + [dist[b] for b in bs])
        v = np.array([vm.iloc[k][lvb]] + [vm.iloc[k][b] for b in bs])
        n_under = int((v[1:] < cfg["v_min"]).sum())
        ax.scatter(d, v, s=22, color=CATEGORICAL[i % 4], edgecolor="white", linewidth=0.5,
                   zorder=3, label=f"{_trafo_label(net, t)} — {len(bs)} buses, "
                                   f"{n_under} under limit")
    ax.set_xlabel("Cable distance from the transformer [m]")
    ax.set_ylabel("Voltage [p.u.]")
    ax.set_xlim(0, None)
    ax.margins(y=0.08)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, cfg["v_min"] - 0.01), hi)
    ax.axhspan(ax.get_ylim()[0], cfg["v_min"], color=CRITICAL, alpha=0.07, lw=0, zorder=0)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Voltage along the LV feeders at the worst time step ({when})",
                 loc="left", fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================
#  Figure 4 -- scenario comparison
# ============================================================
def _hours(h):
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h".replace(".0 h", " h")


def figure_scenarios(results):
    names = list(results)
    cfg = results[names[0]]["cfg"]
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.28)

    # duration curves
    ax_v, ax_l = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for name in names:
        r = results[name]
        lv = _lv_buses(r["network"])
        n = len(r["vm"])
        hours = np.arange(1, n + 1) * 0.25
        vmin_sorted = np.sort(r["vm"][lv].min(axis=1).to_numpy())
        lmax_sorted = np.sort(r["line_load"].max(axis=1).to_numpy())[::-1]
        c = SCENARIO_COLORS.get(name, INK)
        ax_v.plot(hours, vmin_sorted, color=c, lw=2, label=name)
        ax_l.plot(hours, lmax_sorted, color=c, lw=2, label=name)
        h_v = (vmin_sorted < cfg["v_min"]).sum() * 0.25
        h_l = (lmax_sorted > cfg["loading_max"]).sum() * 0.25
        if h_v:
            ax_v.annotate(_hours(h_v), (h_v, cfg["v_min"]), xytext=(4, -12),
                          textcoords="offset points", color=c, fontsize=9)
        if h_l:
            ax_l.annotate(_hours(h_l), (h_l, cfg["loading_max"]), xytext=(4, 4),
                          textcoords="offset points", color=c, fontsize=9)
    for ax, lim, lab in ((ax_v, cfg["v_min"], f"v_min {cfg['v_min']:.2f}"),
                         (ax_l, cfg["loading_max"], f"limit {cfg['loading_max']:.0f} %")):
        ax.axhline(lim, color=CRITICAL, lw=1.2, ls="--")
        ax.text(0.99, lim, lab, color=CRITICAL, ha="right", va="bottom", fontsize=9,
                transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_xlim(0.25, STEPS_PER_WEEK * 0.25)
        ax.set_xticks([0.25, 1, 4, 12, 24, 72, 168])
        ax.set_xticklabels(["15 min", "1 h", "4 h", "12 h", "1 d", "3 d", "1 wk"])
        ax.minorticks_off()
        ax.set_xlabel("Duration exceeded over the week (log scale)")
        ax.legend(loc="lower right" if ax is ax_v else "upper right", fontsize=9)
    ax_v.set_ylabel("Lowest bus voltage [p.u.]")
    ax_v.set_title("Voltage duration curve", loc="left", fontsize=12)
    ax_l.set_ylabel("Most loaded line [%]")
    ax_l.set_title("Line loading duration curve", loc="left", fontsize=12)

    # share of elements breaking a limit at least once
    ax = fig.add_subplot(gs[2])
    groups = ["Buses under\nv_min", "Lines over\nlimit", "Transformers\nover limit"]
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        r = results[name]
        lv = _lv_buses(r["network"])
        shares = [
            100 * (r["vm"][lv].min() < cfg["v_min"]).mean(),
            100 * (r["line_load"].max() > cfg["loading_max"]).mean(),
            100 * (r["trafo_load"].max() > cfg["loading_max"]).mean(),
        ]
        xs = np.arange(len(groups)) + (j - (len(names) - 1) / 2) * width
        bars = ax.bar(xs, shares, width=width * 0.92, color=SCENARIO_COLORS.get(name, INK),
                      label=name, zorder=2)
        for b, s in zip(bars, shares):
            ax.text(b.get_x() + b.get_width() / 2, s + 1.5, f"{s:.0f}", ha="center",
                    va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, max(10.0, ax.get_ylim()[1] * 1.15))
    ax.set_ylabel("Share of elements [%]")
    ax.grid(axis="x", visible=False)
    ax.set_title("Elements in violation at least once", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.14, top=0.9)
    return fig


# ============================================================
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written: {path}")
    return path


def main(only=None):
    names = [only] if only else list(SCENARIOS)
    results = {}
    for name in names:
        print(f"Power flow: {name} ({SCENARIOS[name]}) ...")
        results[name] = run_scenario(SCENARIOS[name])
    focus = only or FOCUS
    if focus in results:
        save(figure_map(results[focus]), f"powerflow_map_{focus}.png")
        save(figure_week(results[focus]), f"powerflow_week_{focus}.png")
        save(figure_feeders(results[focus]), f"powerflow_feeders_{focus}.png")
    if len(results) > 1:
        save(figure_scenarios(results), "powerflow_scenarios.png")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=list(SCENARIOS),
                        help="run and draw a single scenario (no comparison figure)")
    main(parser.parse_args().only)
