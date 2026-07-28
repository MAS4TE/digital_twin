# BSS modelling

The digital twin simulates a **battery storage system (BSS)** natively:
[`run_digital_twin`](interfacing.md) accepts a `schedules_BSS` argument and,
when it is provided, creates a pandapower `storage` element and runs the power
flow with loads, PV **and** the battery together. When `schedules_BSS` is
`None`, no storage is created and the behaviour is identical to a run without
it — the same activation-by-presence rule as `pv_profiles`.

## The model: one central park, N contractual shares

This is the single most important thing to get right before reading anything
else on this page.

The twin models **one physical battery park connected to an MV node**
(`config['bss_bus']`, `MALVOZ BRESSOUX` by default), sized by
`bss_park_e_kwh` and `bss_park_p_kw`. Households do not own batteries in their
garages: they hold **contractual shares** of that single park. A household with
a PV surplus exports it onto the LV network and the park absorbs the equivalent
energy at the MV node.

The energy accounting holds; the electrons do not take that path. Two
consequences worth stating plainly:

- The park supports the **MV node**, not the LV feeder of the household that
  owns the share. Do not read a local voltage benefit into the results.
- `bss_shares.csv` maps `bus_id → e_kwh`, but `bus_id` says *who holds the
  share*, not *where a battery is installed*.

```{mermaid}
flowchart LR
    G[generate_bss_shares] --> C[(inputs/bss_shares.csv)]
    C --> B(build_bss_schedule)
    L[(out/processed/load_profiles.csv)] --> B
    P[pv_profiles] --> B
    B -->|schedules_BSS, kW| D(process_data)
    D --> T(run_digital_twin)
    T --> S[storage on config'bss_bus']
    T --> R[verdict + bss_* metrics]
```

```{note}
Everything under `digital_twin/data_processing/bss.py` is a **reference
implementation and test fixture**, not the production source of schedules. In
MAS4TE the schedule is the output of the agents' own optimisation and they
supply their own file. It exists so the twin's BSS path can be exercised on a
realistic input, and so a partner has a worked example of the expected format.
```

## 1. Share register — `generate_bss_shares`

Writes `inputs/bss_shares.csv` (schema in [Data files](data_files.md)). One
share per LV bus, all of equal energy, `bss_park_e_kwh / n` each. There is no
share count to configure — it *is* the number of LV buses (333 on Bressoux).

The split is deliberately uniform, not weighted by load or PV: this is a test
fixture, and a uniform split makes `sum(e_kwh) == bss_park_e_kwh` hold by
construction, which is exactly the invariant
`utils.check_bss_energetic_feasibility` verifies. The register is fully
deterministic — no draw, hence nothing to seed.

## 2. Schedule generation — `build_bss_schedule`

Pure computation, no network and no pandapower. Returns **one kW column per
share**, on the labelled weekly index so the twin's `[BSS/align]` pre-check can
actually verify alignment rather than trusting a bare `RangeIndex`.

Each share is scheduled against a net-load signal, `load − PV`, and follows a
greedy self-consumption plus peak-shaving rule:

- **Charge** when the net load sits in the low quantile of that day
  (`q_charge`, 0.30) — the PV surplus hours and the night valley.
- **Discharge** when it sits in the high quantile (`q_discharge`, 0.75) — the
  evening peak.

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
  the park a little every day.

Each share's power is capped at its own slice of the park rating, proportional
to its slice of the energy — a share cannot claim more of the inverter than it
owns of the cells. This is what keeps the summed aggregate within
`bss_park_p_kw` by construction.

## 3. Consuming the schedule in the digital twin

`process_data` builds `schedules_BSS` from `inputs/bss_shares.csv` and returns
it as the fifth element of its tuple; `main.py` passes it straight through.
Inside the engine the path is:

1. `_read_bss_schedule` — sums the N columns into one aggregate park series,
   converts kW → MW (the only such conversion point), and checks the
   instantaneous power against `bss_park_p_kw`.
2. `_inject_bss_storage` — creates the single `storage` element on
   `config['bss_bus']`.
3. `_track_bss_soc` — integrates the state of charge step by step, since
   pandapower does not update `soc_percent` during a time-series run.
4. `_run_powerflow` — drives the storage through a `ConstControl`, exactly as
   the PV `sgen` elements are driven.

Sign, unit and feasibility conventions are documented once, in
[Interfacing](interfacing.md#the-schedules_bss-input). The short version:
**kW at the interface, positive = charging**, which is the opposite of the
`sgen` convention `pv_profiles` follows.

## A known modelling tension

Worth stating because it affects how the results should be read, and because it
is not settled.

The park is **one MV-connected asset**, so physically it only ever sees
aggregate flows at its connection point. The generator above, however, builds
its signal **per bus**: a bus carrying PV goes into surplus at midday, a bus
without PV never does. Those per-bus signals are then summed.

That choice was made because aggregating first destroys the surplus: at
neighbourhood scale the PV sum never exceeds the load sum, so a schedule built
from the aggregate net load would never see one. Per-share, a PV-equipped
household genuinely does.

The tension is that per-bus signals are strictly meaningful only for storage
sitting *behind each meter* — which is not the model. For a central park, the
coherent signal would be aggregate on both sides. The two readings cannot be
served by the same calculation, and the current code takes the per-bus one.

One visible consequence: only 30 of the 333 Bressoux LV buses carry PV, so 303
shares schedule with no PV term at all and their "charge in the night valley"
shape dominates the park aggregate. The midday charging window is faint. That
is the neighbourhood's real 9% PV penetration showing through, not a defect —
but it does mean the aggregate profile looks close to a plain night-charge /
evening-discharge schedule.

## Generating the input file

```bash
python -c "from digital_twin.data_processing.bss import generate_bss_shares; generate_bss_shares({'data_path': 'data/bressoux'})"
```

Writes `inputs/bss_shares.csv`. Only needed to generate a *new* register; the
one already in `inputs/` is used as-is otherwise.

## Visual check

`scripts/viz/plot_bss_profiles.py` draws the week (load, PV, park, net), the
park power against its state of charge, and the daily profiles superimposed —
without running any power flow. Run it from the repository root.
