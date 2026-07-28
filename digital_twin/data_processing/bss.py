"""
BSS generation: contractual share register + net-load-driven schedule generator,
producing the `schedules_BSS` input of run_digital_twin.

Reference implementation and test harness, not the production source: in MAS4TE
the schedule comes from the agents' own optimisation. Nothing here is imported by
digital_twin.digital_twin -- the dependency goes one way only.

MODEL. One physical central park at an MV node (config['bss_bus']) plus N
contractual shares of it. A share is not a battery in a garage: the household
exports its surplus and the park absorbs the equivalent at the MV node. The
energy accounting holds, the electrons do not take that path -- so the park
supports the MV node, NOT the owner's LV feeder. Do not read a local voltage
benefit into the results.

Brick 1 -- generate_bss_shares : one share per LV bus, equal energy -> bss_shares.csv
Brick 2 -- build_bss_schedule  : per-share net load -> N columns in kW (no pandapower)

Each share schedules against ITS OWN bus's net load, hence one column per share
rather than a single aggregate: a PV bus goes into surplus at midday, a bus
without PV never does -- two different shapes, not one shape at two scales.
digital_twin._read_bss_schedule sums the columns at the interface.

UNITS. kW, positive = CHARGING (the storage convention of the twin's interface,
see digital_twin._inject_bss_storage). `schedules_BSS` is the twin's only kW
interface -- unlike pv.build_pv_production, whose kW are converted to MW upstream
in process_data -- so nothing is converted here.
"""
import os

import numpy as np
import pandas as pd

from ..utils import (STEPS_PER_DAY, STEPS_PER_WEEK, _WEEK_TIME_INDEX,
                     _config_to_dict, load_site_network)

BSS_SHARES_PATH = "data/bressoux/inputs/bss_shares.csv"

# Mirrors digital_twin.BSS_SOC_MIN_FRAC, duplicated rather than imported to keep
# this module independent of the engine. If the engine's reserve changes, change
# it here too: a mismatch only produces [BSS/soc] warnings, never silent drift.
SHARE_SOC_MIN_FRAC = 0.1
SHARE_SOC_INIT = 0.5

# No 'seed' key: the register is deterministic (one slice per LV bus, equal
# energy), so there is no draw left to seed.
DEFAULTS = {
    "out_path": BSS_SHARES_PATH,
    # Quantiles of each share's own daily net load, never absolute kW: the
    # aggregate net load never goes negative at this site, so an absolute
    # "charge on surplus" rule would almost never fire.
    "q_charge": 0.30,              # charge below this daily quantile of p_net
    "q_discharge": 0.75,           # discharge above this daily quantile
}


# ============================================================
#  Brick 1 -- contractual share register
# ============================================================
def generate_bss_shares(config=None):
    """Generate `bss_shares.csv` (or cfg['out_path']): one share per LV bus, all
    of equal energy.

    Every bus of network['buses'] gets a slice -- no draw, no selection, so the
    share count is not configurable: it IS the number of LV buses. Sizing is
    uniform (bss_park_e_kwh / n), deliberately not weighted by load or PV: this
    is a test fixture, and a uniform split makes `sum(e_kwh) == bss_park_e_kwh`
    hold by construction.

    Expected consequence, not a defect: only 30 of the 333 Bressoux LV buses
    carry PV, so 303 shares schedule with no PV term at all and their "charge in
    the night valley" shape dominates the park aggregate. That is this
    neighbourhood's real 9% PV penetration showing through.

    Returns
    -------
    shares : pd.DataFrame   # columns bus_id, e_kwh, street
    """
    cfg = {**DEFAULTS, **_config_to_dict(config)}

    network = load_site_network(cfg)          # READ-ONLY, for bus_id <-> street
    buses = network['buses']                  # (street, num) -> LV bus id

    share_buses = sorted(int(b) for b in buses.values())
    n_shares = len(share_buses)
    if n_shares == 0:
        raise ValueError(
            "network['buses'] is empty: no LV bus to place a BSS share on."
        )

    # The last slice absorbs the division residual: utils.check_bss_energetic_
    # feasibility compares with a strict `>` and no tolerance, so a total of
    # 2600.0001 would be reported as an oversubscription of "0.0 kWh".
    park_e = float(cfg['bss_park_e_kwh'])
    e_kwh = np.round(np.full(n_shares, park_e / n_shares), 6)
    e_kwh[-1] += park_e - e_kwh.sum()

    bus_street = {int(b): s for (s, _n), b in buses.items()}
    shares = pd.DataFrame({
        "bus_id": share_buses,
        "e_kwh": e_kwh,
        "street": [bus_street[b] for b in share_buses],
    })

    out_path = cfg['out_path']
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shares.to_csv(out_path, index=False)
    return shares


# ============================================================
#  Brick 2 -- net-load-driven schedule
# ============================================================
def _representative_load_kw(households):
    """Mean household load across the profile pool [kW].

    Not the actual load of each share's bus: the household -> bus assignment is
    drawn inside digital_twin._assign_loads at run time and does not exist
    upstream. The signal that shapes the schedule is PV presence, which IS known
    exactly per bus -- so the exact part stays exact and only the load is
    approximated.
    """
    return households.mean(axis=1).to_numpy(dtype=float) * 1000.0  # MW -> kW


def _schedule_one_share(p_net_kw, e_kwh, p_max_kw, q_charge, q_discharge, eta):
    """Greedy self-consumption + peak-shaving schedule for ONE share.

    Per day, the share charges when its net load sits in the low quantile (PV
    surplus hours and the night valley) and discharges in the high quantile (the
    evening peak). Per day rather than per week so the rule follows the
    weekday/weekend shape. Power is proportional to the excursion beyond the
    threshold, so the deepest surplus hour charges hardest.

    Each day closes energy-neutral, balanced on CELL-side energy: charging stores
    p*dt*eta and discharging draws p*dt/eta, exactly as the engine's
    _track_bss_soc computes it, so a full cycle costs eta**2. Balancing on
    meter-side kWh instead ignores the round-trip loss and drains the park a
    little every day.

    The SoC walk on top caps power by the room left in [SHARE_SOC_MIN_FRAC, 1.0],
    which is what makes the result feasible rather than merely plausible.

    Returns
    -------
    p_kw : np.ndarray   # kW, positive = charging, negative = discharge
    """
    n = len(p_net_kw)
    p_kw = np.zeros(n, dtype=float)
    dt_h = 24.0 / STEPS_PER_DAY

    e_max = float(e_kwh)
    e_min = SHARE_SOC_MIN_FRAC * e_max
    e = SHARE_SOC_INIT * e_max

    for start in range(0, n, STEPS_PER_DAY):
        day = p_net_kw[start:start + STEPS_PER_DAY]
        if len(day) == 0:
            continue
        lo_thr = float(np.quantile(day, q_charge))
        hi_thr = float(np.quantile(day, q_discharge))

        want_chg = np.clip(lo_thr - day, 0.0, None)
        want_dis = np.clip(day - hi_thr, 0.0, None)

        # Each side scaled INDEPENDENTLY to the rating. A shared
        # max(chg, dis) normaliser would let a PV bus's deep midday trough shrink
        # its own discharge side, and the neutrality rebalance would then drag
        # the charge down to match -- throttling the shares with the most surplus.
        if want_chg.max() > 0:
            want_chg *= p_max_kw / want_chg.max()
        if want_dis.max() > 0:
            want_dis *= p_max_kw / want_dis.max()
        stored_in = want_chg.sum() * eta       # what charging actually stores
        drawn_out = want_dis.sum() / eta       # what discharging actually costs
        if stored_in <= 0 or drawn_out <= 0:
            continue
        if stored_in > drawn_out:
            want_chg *= drawn_out / stored_in
        else:
            want_dis *= stored_in / drawn_out

        for k in range(len(day)):
            if want_chg[k] > 0:
                # Band headroom as METER power: storing the last kWh costs 1/eta.
                p = min(want_chg[k], max(0.0, (e_max - e) / (dt_h * eta)))
                e += p * dt_h * eta
            elif want_dis[k] > 0:
                # Symmetrically, usable stock at the meter is (e - e_min) * eta.
                p = -min(want_dis[k], max(0.0, (e - e_min) * eta / dt_h))
                e += p * dt_h / eta
            else:
                p = 0.0
            p_kw[start + k] = p

    return p_kw


def build_bss_schedule(households, pv_profiles, shares, config=None):
    """Build `schedules_BSS`: one kW column per contractual share.

    Parameters
    ----------
    households : pd.DataFrame   # consumption timeseries, MW
    pv_profiles : pd.DataFrame or None
        PV production, MW, columns 'pv_{bus_id}'. None, or a bus with no PV,
        means no surplus term for that share -- it then schedules on the load
        valley/peak alone, which is the correct degenerate case, not an error.
    shares : pd.DataFrame       # bss_shares.csv: bus_id, e_kwh
    config : dict or None
        Reads bss_park_e_kwh, bss_park_p_kw, bss_eta, q_charge, q_discharge.
        `bss_eta` is the SAME key the engine uses -- planning against a different
        efficiency than the twin applies is what makes a schedule drift.

    Returns
    -------
    schedules_BSS : pd.DataFrame
        Columns 'bss_{bus_id}', kW, positive = charging. Index is the labelled
        weekly index matching `households`, so _read_bss_schedule's [BSS/align]
        check can verify alignment instead of trusting a bare RangeIndex.
    """
    cfg = {**DEFAULTS, **_config_to_dict(config)}
    if shares is None or len(shares) == 0:
        raise ValueError("`shares` is empty: nothing to schedule (see generate_bss_shares).")

    load_kw = _representative_load_kw(households)
    n_steps = len(load_kw)

    park_e = float(cfg['bss_park_e_kwh'])
    park_p = float(cfg['bss_park_p_kw'])

    columns = {}
    for _, row in shares.iterrows():
        bus_id = int(row['bus_id'])
        e_kwh = float(row['e_kwh'])

        # PV of THIS bus, exactly -- the part of the signal worth keeping exact.
        pv_col = f"pv_{bus_id}"
        if pv_profiles is not None and pv_col in pv_profiles.columns:
            pv_kw = pv_profiles[pv_col].to_numpy(dtype=float)[:n_steps] * 1000.0
        else:
            pv_kw = np.zeros(n_steps, dtype=float)

        p_net_kw = load_kw - pv_kw

        # Power slice proportional to the energy slice: a share cannot claim more
        # of the inverter than it owns of the cells. Keeps the summed aggregate
        # within bss_park_p_kw by construction, so [BSS/schedule] stays silent.
        p_max_kw = park_p * (e_kwh / park_e) if park_e > 0 else 0.0

        columns[f"bss_{bus_id}"] = _schedule_one_share(
            p_net_kw, e_kwh, p_max_kw, cfg['q_charge'], cfg['q_discharge'],
            float(cfg['bss_eta']))

    index = _WEEK_TIME_INDEX[:n_steps] if n_steps <= STEPS_PER_WEEK else None
    return pd.DataFrame(columns, index=index)


def load_bss_schedule(households, pv_profiles, config=None, inputs_dir=None):
    """Read `<inputs_dir>/bss_shares.csv` and build the schedule from it.

    Mirrors process_data._load_pv_profiles: returns None when the scenario has no
    share register, a scenario opting into BSS simply by having the file.

    Parameters
    ----------
    households : pd.DataFrame            # consumption timeseries, MW
    pv_profiles : pd.DataFrame or None   # PV production, MW
    config : dict or None
    inputs_dir : str or None             # defaults to <data_path>/inputs

    Returns
    -------
    schedules_BSS : pd.DataFrame or None   # kW, one column per share
    """
    cfg = {**DEFAULTS, **_config_to_dict(config)}
    if inputs_dir is None:
        inputs_dir = f"{cfg['data_path'].rstrip('/')}/inputs"
    shares_path = os.path.join(inputs_dir, "bss_shares.csv")
    if not os.path.exists(shares_path):
        return None
    shares = pd.read_csv(shares_path)
    return build_bss_schedule(households, pv_profiles, shares, config)
