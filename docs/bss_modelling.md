# BSS modelling

The digital twin simulates **battery storage (BSS)** natively:
[`run_digital_twin`](interfacing.md) accepts a `schedules_BSS` argument and,
when it is provided, creates one pandapower `storage` element per battery and
runs the power flow with loads, PV **and** the batteries together. When
`schedules_BSS` is `None`, no storage is created and the behaviour is identical
to a run without it — the same activation-by-presence rule as `pv_profiles`.

## The model: a fleet of physical batteries

This is the single most important thing to get right before reading anything
else on this page.

The twin models a **fleet**, and every battery in it is a physical object at a
real point of the grid. Both kinds are now **rows of the same `equipment.csv`**
(see [Placement](#1-placement--the-equipment-table) below) — nothing about a
battery's bus, capacity or power lives in runtime `config` any more:

| id | what | where | sized by |
|---|---|---|---|
| `b0` | central battery | an MV node | its own `battery_size_kwh` / `battery_p_max_kw` row |
| `b1`…`bn` | domestic batteries | one per equipped LV bus | `battery_size_kwh` in the equipment table |

A battery exists exactly when the schedule names it. Central-only,
domestic-only and both are all expressible, and none of them needs a flag —
the fourth combination, neither, is simply an `equipment.csv` with no
`battery_size_kwh > 0` row at all, and `process_data` returns `schedules_BSS
= None`.

Domestic batteries take the **central battery's C-rate** rather than a power
spec of their own: `utils.bss_c_rate(config, equipment)` returns
`battery_p_max_kw / battery_size_kwh` of the equipment table's `b0` row
(0.229 C on the Bressoux defaults) unless the scenario sets
`config['bss_c_rate']`, and each domestic battery's rating is
`battery_size_kwh × c_rate`. The fleet is then one technology at two scales,
and re-speccing the central battery drags the homes along instead of silently
pairing a fast park with slow batteries. A **domestic-only** fleet (no `b0`
row) has no park to derive a rate from, so `bss_c_rate` *raises* unless
`config['bss_c_rate']` is set explicitly.

```{mermaid}
flowchart LR
    N[(inputs/network.py)] --> E[generate_equipment_table]
    E --> C[(out/processed/equipment.csv)]
    C --> B(build_bss_schedule)
    L[(out/processed/load_profiles.csv)] --> B
    P[pv_profiles] --> B
    B -->|schedules_BSS, long, kW| D(process_data)
    C --> D
    D --> T(run_digital_twin)
    T --> S0[storage b0 on an MV node]
    T --> S1[storage b1..bn on their LV buses]
    T --> R[verdict + bss_* metrics]
```

```{note}
Everything under `digital_twin/data_processing/bss.py` is a **reference
implementation and test fixture**, not the production source of schedules. In
MAS4TE the schedule is the output of the agents' own optimisation and they
supply their own file. It exists so the twin's BSS path can be exercised on a
realistic input, and so a partner has a worked example of the expected format.
```

Each battery is scheduled against the signal that fits what it is: `b1`…`bn` are
behind-the-meter batteries, so a **per-bus** signal is the right one for them —
a bus with PV goes into surplus at midday, a bus without never does, and
aggregating first destroys that surplus, since at neighbourhood scale the PV sum
never exceeds the load sum. `b0` sits on an MV node and gets the **aggregate**
signal for the same reason. Voltage support therefore happens where the battery
physically is.

## 1. Placement — the equipment table

Which buses carry a battery, and how big, is decided in
[`data_processing.equipment`](data_files.md) and written to
`out/processed/equipment.csv`, one row per equipable bus. `place_bss` draws
among the PV-equipped LV buses by default (`batteries_follow_pv`), since a home
battery is almost always installed alongside a PV system, and assigns
`battery_id` as `b1`…`bn` in ascending `bus_id` order — deterministic, so the
same table always yields the same identifiers.

`b0` **is** a row of that table too, placed separately by `place_central_bss`:
opt-in by `config['bss_bus']` presence (an MV node name or bus id — no key
means no central battery, exactly like `mv_pv_installations`), sized by
`config['bss_park_e_kwh']`/`config['bss_park_p_kw']` (both required and
validated `> 0` as soon as `bss_bus` is set — unlike a domestic battery, the
central one has no C-rate to derive its own power from). These three keys are
**generation-time only**: consumed exclusively inside `generate_equipment_table`,
never read by `run_digital_twin`, which in fact *raises* if it sees them in a
caller's config (a loud guard against the wrong-result trap of setting a key
that used to matter and now does nothing).

## 2. Schedule generation — `build_bss_schedule`

Pure computation, no network and no pandapower. Returns the **long format**
described below, on the labelled weekly index so the twin's `[BSS/align]`
pre-check can actually verify alignment rather than trusting a bare
`RangeIndex`.

Each battery is scheduled against **its own** net-load signal, `load − PV`, and
follows a greedy self-consumption plus peak-shaving rule:

- **Charge** when the net load sits in the low quantile of that day
  (`q_charge`, 0.30) — the PV surplus hours and the night valley.
- **Discharge** when it sits in the high quantile (`q_discharge`, 0.75) — the
  evening peak.

For `b1`…`bn` the signal is **its own bus's** household load minus **its own**
PV; for `b0` it is the neighbourhood aggregate (the sum of the assigned loads,
minus all the PV).

```{note}
A domestic battery is scheduled against the household it sits behind, which
requires knowing *which* household that is — `metadata['bus_id']`, the declared
assignment. On Bressoux the 20 schedules then cycle between 11.7 and 43.1
kWh/week, tracking households that consume between 0.1 and 75.9 kWh/week.

Passing `metadata=None` falls back to the **pool mean**, for a caller that has
no assignment to declare. Note what that fallback costs: the 20 batteries all
sit on 3 kWp PV buses, so they receive the same net-load signal and their
schedules come out bit-identical.
```

```{warning}
The quantile rule degenerates on a **near-flat** load. A household consuming
0.1 kWh over the whole week still has a "low quantile" and a "high quantile", so
its battery is scheduled to arbitrage between them — 22 kWh/week of cycling for
a dwelling that consumes almost nothing. Bounded and harmless for the network,
but meaningless economically; see also the clipping note in
[Load assignment](load_assignment.md).
```

Thresholds are **quantiles of each day**, never absolute kW. At this site the
aggregate net load never goes negative, so an absolute "charge on surplus" rule
would almost never fire; and per-day thresholds let the rule follow the
weekday/weekend shape on its own.

Two properties make the result usable rather than merely plausible:

- **Power is proportional** to how far the net load sits beyond the threshold,
  so the deepest surplus hour charges hardest. A flat "charge at max whenever
  below threshold" rule produces a square wave.
- **Each day closes energy-neutral**, balanced on *cell-side* energy using
  `config['bss_eta']` exactly as the engine's SoC tracker does — charging
  stores `p·dt·η`, discharging draws `p·dt/η`, so a full cycle costs `η²`.
  Balancing on meter-side kWh instead ignores the round-trip loss and drains
  the battery a little every day.

The SoC walk on top caps power by the room left in the
`[config['bss_soc_min'], 1.0]` band, starting at `config['bss_soc_init']` —
the *same* keys the engine reads, so a schedule built here is feasible there.
They default to 0.1 and 0.5 (`utils.BSS_SOC_MIN_FRAC` / `BSS_SOC_INIT`). Set
both to 0 to match a market model that starts the batteries empty and uses
their full capacity.

## 3. The `schedules_BSS` format

**Long**: the index is the time steps repeated once per battery, and there are
exactly two columns.

| index | `battery_id` | `p_kw` |
|---|---|---|
| `Mon 00:00` | `b0` | `+142.7` |
| `Mon 00:15` | `b0` | `+138.2` |
| … | … | … |
| `Mon 00:00` | `b1` | `+1.4` |

The identifier column is what makes the format **checkable**: a declared
`b0`…`bn` can be matched against the equipment table and the network, where the
free-named column of a wide frame carries no such meaning. That check is what
lets each battery keep its own storage element, bus and state of charge instead
of being summed into a single aggregate.

Two contracts are enforced in `_validate_inputs`, both as **errors**:

- **Identifiers.** `b1`…`bn` in the schedule must be exactly the set of
  `battery_id` values the equipment table carries with `battery_size_kwh > 0`.
  Compared by **set**, never by count: "20 batteries scheduled for 20 equipped
  buses" can match by construction while designating different batteries. Both
  directions are reported, because they mean opposite things — a schedule for a
  battery nobody installed, or an installed battery nobody scheduled.
- **Bus ids.** `b1`…`bn` must land in `BUS_DOMAIN_LV`, `b0` in
  `BUS_DOMAIN_MV_NODE`. A domestic battery on an MV node is a placement error,
  and the central one has no LV bus to sit on.

## 4. Consuming the schedule in the digital twin

`process_data` builds `schedules_BSS` and returns it with the equipment table;
`main.py` passes both straight through. Inside the engine the path is:

1. `_bss_fleet_specs` — resolves the fleet once: each battery's bus, capacity
   and power rating. The contract check, the injection and the SoC tracking all
   read this one dict rather than re-deriving placement three times.
2. `_read_bss_schedule` — pivots long → wide, converts kW → MW (the same
   `_KW_TO_MW` conversion point `_assign_loads` and the PV injection use), and
   checks each battery's instantaneous power against **its own** rating.
3. `_track_bss_soc` — integrates each state of charge separately, since
   pandapower does not update `soc_percent` during a time-series run. Per
   battery rather than fleet-wide because the capacities differ by two orders of
   magnitude: a fleet-level SoC would let the park's headroom mask a domestic
   battery driven flat. Each walk starts at `config['bss_soc_init']` and stays
   within `[config['bss_soc_min'], 1.0]`.
4. `_inject_bss_storage` — creates one `storage` element per battery, on its own
   bus, `b0` and `b1`…`bn` through the same loop.
5. `_run_powerflow` — drives each storage through a `ConstControl`, exactly as
   the PV `sgen` elements are driven.

The series that reaches the power flow is the one `_track_bss_soc` returns, not
the raw schedule — otherwise `bss_enforce_limits` would only ever produce a
warning while the infeasible schedule ran anyway.

Sign, unit and feasibility conventions are documented once, in
[Interfacing](interfacing.md#the-schedules_bss-input). The short version:
**kW at the interface, positive = charging**, which is the opposite of the
`sgen` convention `pv_profiles` follows.

## Reported metrics

The run verdict carries the fleet **aggregate**: `bss_n_batteries`,
`bss_n_domestic`, `bss_e_installed_kwh`, `bss_p_agg_max_kw` /
`bss_p_agg_min_kw` (signed, so max > 0 is the strongest charge), and
`bss_soc_min_pct` / `bss_soc_max_pct` across the whole fleet.

Per-battery state of charge is opt-in through `config['export_bss_soc']`, which
appends `category == 'bss_soc'` rows (one per battery per timestep). Off by
default because that is `n_batteries × n_timesteps` rows on every run.

## Generating the input file

The fleet comes from the equipment table, so there is no separate register to
generate any more. Include `bss_bus`/`bss_park_e_kwh`/`bss_park_p_kw` to place
a central battery too — omit them for a domestic-only fleet (`place_central_bss`
then no-ops, table unchanged):

```bash
python -c "from digital_twin.data_processing.equipment import generate_equipment_table; generate_equipment_table({'data_path': 'data/bressoux', 'bss_bus': 'MALVOZ BRESSOUX', 'bss_park_e_kwh': 2600.0, 'bss_park_p_kw': 595.0})"
```

`process_data` then builds the schedule from that table on every call.

## Visual check

`digital_twin/plots/plot_bss_profiles.py` draws the week (load, PV, fleet, net), the
fleet power against its capacity-weighted state of charge, and the daily
profiles superimposed — without running any power flow. The fleet is aggregated
*for the figure only*; those curves answer "what does the transformer see". Run
it from the repository root.
