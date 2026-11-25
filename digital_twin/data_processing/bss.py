"""
BSS generation: the reference schedule generator producing the `schedules_BSS`
input of run_digital_twin.

Reference implementation and test harness, NOT the production source: in MAS4TE
the schedule comes from the agents' own optimisation. digital_twin.digital_twin
never imports this module -- the dependency goes one way only, and this module
imports the engine's format constants rather than restating them, so the two
cannot drift.

MODEL. A FLEET of physically distinct batteries, read off the scenario's
equipment table (utils.bss_register) -- BOTH kinds are now rows of that same
table, generated once (see data_processing.equipment.place_central_bss /
place_bss):
    b0     -- the CENTRAL battery at an MV node, capacity AND power read
              directly off its own row (battery_size_kwh, battery_p_max_kw) --
              a park cannot derive its own power from itself.
    b1..bn -- one DOMESTIC battery per equipped LV bus, capacity from the table's
              `battery_size_kwh` column, power DERIVED from the central battery's
              C-rate (utils.bss_c_rate) so the fleet stays one technology at two
              scales. A domestic-only fleet (no b0 row) needs an explicit
              config['bss_c_rate'] instead -- bss_c_rate raises otherwise.

ACTIVATION is fully DATA-DRIVEN, uniformly for both kinds: b0 is scheduled iff
the equipment table carries a central-battery row, b1..bn iff it carries any
domestic row. All four combinations (central-only, domestic-only, both,
neither) are expressible without a flag.

A domestic battery sits on its own bus and its power flows through its own LV
feeder -- it is a physical battery in a garage, not an accounting share of the
central park.

Each battery schedules against ITS OWN net load, hence one series per battery: a
bus with PV goes into surplus at midday, a bus without never does -- two
different shapes, not one shape at two scales.

OUTPUT FORMAT is LONG (index = week steps repeated per battery, columns
battery_id + p_kw), because that is what carries the identifier the twin matches
against the equipment table. See digital_twin.digital_twin's "2ter" section.

UNITS. kW throughout, positive = CHARGING (the storage convention of the
twin's interface, see digital_twin._inject_bss_storage). `households`,
`pv_profiles` and this module's own `schedules_BSS` output are all kW at the
twin's interface -- the kW -> MW conversion only happens inside digital_twin.py,
right before pandapower injection -- so nothing is converted here.
"""
import numpy as np
import pandas as pd

from ..digital_twin import BSS_BATTERY_ID_COLUMN, BSS_POWER_COLUMN
from ..utils import (BSS_CENTRAL_ID, BSS_SOC_INIT, BSS_SOC_MIN_FRAC,
                     STEPS_PER_DAY, STEPS_PER_WEEK, _WEEK_TIME_INDEX,
                     _config_to_dict, bss_c_rate, bss_register)
from .equipment import BUS_TYPE_LV

# No 'seed' key: the fleet comes from the equipment table and every schedule is
# a deterministic function of the net load, so there is no draw left to seed.
DEFAULTS = {
    # Quantiles of each battery's own daily net load, never absolute kW: the
    # aggregate net load never goes negative at this site, so an absolute
    # "charge on surplus" rule would almost never fire.
    "q_charge": 0.30,              # charge below this daily quantile of p_net
    "q_discharge": 0.75,           # discharge above this daily quantile
}


def _bus_load_lookup(households, metadata):
    """{bus_id -> that bus's household load [kW]}, from the declared assignment.

    Empty when `metadata` carries no `bus_id`: callers then fall back to the pool
    mean, which plans every battery against the average household rather than the
    load it will actually see.
    """
    if metadata is None or 'bus_id' not in getattr(metadata, 'columns', []):
        return {}
    lookup = {}
    for bus_id, pid in zip(metadata['bus_id'], metadata['profile_id']):
        col = f"profile_{pid}"
        if pd.notna(bus_id) and col in households.columns:
            lookup[int(bus_id)] = households[col].to_numpy(dtype=float)
    return lookup


def _mean_load_kw(households):
    """Mean household load across the profile pool [kW] -- the FALLBACK used when
    no assignment is declared, and the shape the central battery would see if it
    had to guess."""
    return households.mean(axis=1).to_numpy(dtype=float)


def _pv_kw(pv_profiles, bus_id, n_steps):
    """PV production of ONE bus [kW], zeros when it carries none.

    A bus without PV is the correct degenerate case, not an error: its battery
    then schedules on the load valley/peak alone.
    """
    if pv_profiles is None:
        return np.zeros(n_steps, dtype=float)
    col = f"pv_{int(bus_id)}"
    if col not in pv_profiles.columns:
        return np.zeros(n_steps, dtype=float)
    return pv_profiles[col].to_numpy(dtype=float)[:n_steps]


def _schedule_one_battery(p_net_kw, e_kwh, p_max_kw, q_charge, q_discharge, eta,
                          soc_min=BSS_SOC_MIN_FRAC, soc_init=BSS_SOC_INIT):
    """Greedy self-consumption + peak-shaving schedule for ONE battery.

    Per day, the battery charges when its net load sits in the low quantile (PV
    surplus hours and the night valley) and discharges in the high quantile (the
    evening peak). Per day rather than per week so the rule follows the
    weekday/weekend shape. Power is proportional to the excursion beyond the
    threshold, so the deepest surplus hour charges hardest.

    Each day closes energy-neutral, balanced on CELL-side energy: charging stores
    p*dt*eta and discharging draws p*dt/eta, exactly as the engine's
    _track_one_soc computes it, so a full cycle costs eta**2. Balancing on
    meter-side kWh instead ignores the round-trip loss and drains the battery a
    little every day.

    The SoC walk on top caps power by the room left in [soc_min, 1.0], starting
    at soc_init -- build_bss_schedule passes config['bss_soc_min'/'bss_soc_init'],
    the SAME keys the engine reads, so a schedule built here is feasible there.

    Returns
    -------
    p_kw : np.ndarray   # kW, positive = charging, negative = discharge
    """
    n = len(p_net_kw)
    p_kw = np.zeros(n, dtype=float)
    dt_h = 24.0 / STEPS_PER_DAY

    e_max = float(e_kwh)
    e_min = soc_min * e_max
    e = soc_init * e_max

    for start in range(0, n, STEPS_PER_DAY):
        day = p_net_kw[start:start + STEPS_PER_DAY]
        if len(day) == 0:
            continue
        lo_thr = float(np.quantile(day, q_charge))
        hi_thr = float(np.quantile(day, q_discharge))

        want_chg = np.clip(lo_thr - day, 0.0, None)
        want_dis = np.clip(day - hi_thr, 0.0, None)

        # Each side scaled INDEPENDENTLY to the rating. A shared max(chg, dis)
        # normaliser would let a PV bus's deep midday trough shrink its own
        # discharge side, and the neutrality rebalance would drag the charge down
        # to match -- throttling the batteries with the most surplus.
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


def build_bss_schedule(households, pv_profiles, equipment, metadata=None, config=None):
    """Build `schedules_BSS` in LONG format for the scenario's whole fleet.

    Parameters
    ----------
    households : pd.DataFrame   # consumption timeseries, kW
    pv_profiles : pd.DataFrame or None
        PV production, kW, columns 'pv_{bus_id}'. None, or a bus with no PV,
        means no surplus term for that battery.
    equipment : pd.DataFrame    # the bus-level register (bus_id, battery_id,
        battery_size_kwh, battery_p_max_kw) -- the fleet's placement and sizing,
        central battery (b0) included, see data_processing.equipment.
    metadata : pd.DataFrame or None
        Household table carrying the declared household -> bus assignment
        (`bus_id`). When given, EACH domestic battery is scheduled against the
        load of ITS OWN bus. Without it, every battery falls back to the pool
        mean and plans against a load its bus will never see.
    config : dict or None
        Reads bss_c_rate, bss_eta, bss_soc_min, bss_soc_init, q_charge, q_discharge. `bss_eta` is the SAME
        key the engine uses -- planning against a different efficiency than the
        twin applies is what makes a schedule drift.

    Returns
    -------
    schedules_BSS : pd.DataFrame
        Index = the labelled weekly index matching `households`, repeated once
        per battery; columns 'battery_id' and 'p_kw'. b0 first (when present),
        then b1..bn. The labelled index (rather than a bare RangeIndex) is what
        lets the engine's [BSS/align] check verify alignment instead of
        trusting the provider.
    """
    cfg = {**DEFAULTS, **_config_to_dict(config)}

    full_register = bss_register(equipment)
    central = full_register.get(BSS_CENTRAL_ID)
    domestic = {b: v for b, v in full_register.items() if b != BSS_CENTRAL_ID}
    if central is None and not domestic:
        raise ValueError(
            "Nothing to schedule: the equipment table carries no battery (neither a "
            "central b0 row nor any domestic b1..bn row)."
        )

    mean_load_kw = _mean_load_kw(households)
    bus_load = _bus_load_lookup(households, metadata)
    n_steps = len(mean_load_kw)
    eta = float(cfg['bss_eta'])
    soc_band = dict(soc_min=float(cfg['bss_soc_min']), soc_init=float(cfg['bss_soc_init']))
    q_charge, q_discharge = cfg['q_charge'], cfg['q_discharge']

    series = {}

    if central is not None:
        # The central battery answers to the WHOLE neighbourhood's net load, so
        # its signal is the aggregate: the representative household load times
        # the number of LV buses, minus all the PV. MV load facilities are not in
        # it -- their profiles are read at run time by _assign_loads and do not
        # exist upstream of the twin.
        if equipment is None or 'bus_type' not in getattr(equipment, 'columns', []):
            raise ValueError(
                "Central battery schedule: the equipment table and its 'bus_type' "
                "column are required to count the neighbourhood's LV buses "
                "(see data_processing.equipment.build_equipment_skeleton)."
            )
        n_lv = int((equipment['bus_type'] == BUS_TYPE_LV).sum())
        if n_lv == 0:
            raise ValueError(
                "Central battery schedule: no LV bus in the equipment table "
                f"(bus_type == '{BUS_TYPE_LV}'), so the aggregate load would be zero "
                "and the schedule constantly flat."
            )
        # Sum of the loads actually assigned when the assignment is declared;
        # mean x n_lv otherwise. The two coincide under a 1:1 assignment, but the
        # sum stays correct if that stops holding.
        agg_load_kw = (np.sum(list(bus_load.values()), axis=0) if bus_load
                       else mean_load_kw * n_lv)
        agg_pv_kw = (pv_profiles.sum(axis=1).to_numpy(dtype=float)[:n_steps]
                     if pv_profiles is not None else np.zeros(n_steps, dtype=float))
        series[BSS_CENTRAL_ID] = _schedule_one_battery(
            agg_load_kw - agg_pv_kw, central['e_kwh'], central['p_max_kw'],
            q_charge, q_discharge, eta, **soc_band)

    # A domestic battery's rating follows the central battery's C-rate (see
    # utils.bss_c_rate), keeping the fleet one technology at two scales -- or an
    # explicit config['bss_c_rate'] override for a domestic-only fleet.
    c_rate = bss_c_rate(cfg, equipment) if domestic else None
    for battery_id in sorted(domestic, key=lambda i: int(i[1:])):
        entry = domestic[battery_id]
        e_kwh = float(entry['e_kwh'])
        # ITS OWN bus's load, not the pool average -- a home battery arbitrages
        # the household it sits behind. Falls back to the mean only when no
        # assignment was declared.
        own_load_kw = bus_load.get(int(entry['bus_id']), mean_load_kw)[:n_steps]
        p_net_kw = own_load_kw - _pv_kw(pv_profiles, entry['bus_id'], n_steps)
        series[battery_id] = _schedule_one_battery(
            p_net_kw, e_kwh, e_kwh * c_rate, q_charge, q_discharge, eta, **soc_band)

    index = _WEEK_TIME_INDEX[:n_steps] if n_steps <= STEPS_PER_WEEK else range(n_steps)
    return pd.concat([
        pd.DataFrame({BSS_BATTERY_ID_COLUMN: battery_id, BSS_POWER_COLUMN: values},
                     index=index)
        for battery_id, values in series.items()
    ])


def load_bss_schedule(households, pv_profiles, equipment, metadata=None, config=None):
    """Build the fleet schedule from the scenario's equipment table, or None.

    ACTIVATION BY PRESENCE, exactly as process_data._load_pv_profiles does for
    PV: the carrier is the table's CONTENT (any row with battery_size_kwh > 0,
    central b0 or domestic b1..bn alike), not the existence of a file. A
    central-only scenario is reachable through this automatic path.

    Returns
    -------
    schedules_BSS : pd.DataFrame or None   # long format, kW
    """
    if not bss_register(equipment):
        return None
    return build_bss_schedule(households, pv_profiles, equipment, metadata, config)
