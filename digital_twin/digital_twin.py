"""
Digital twin engine -- network-agnostic.

Exposes a single entry point `run_digital_twin` callable by the MAS4TE
framework. Given a network, household metadata and consumption profiles, it
assigns loads, runs a time-series power flow, and returns a verdict on whether
the network keeps voltages and line/transformer loadings within acceptable
limits.

Pipeline (see docs/pipeline.md):
    1. Data ingestion / input validation
    2. Load assignment (network reconstruction is now a CALLER responsibility,
       see `network` below -- this file no longer builds any topology itself)
    3. Time-series power flow
    4. Results evaluation (verdicts)
    5. Output formatting

Scope: LOADS + PV. `pv_profiles` is a native capability (sgen injection, see
_inject_pv_sgens / _run_powerflow): when provided, the power flow runs loads and
PV together; when None, loads-only. `schedules_BSS` is accepted in the signature
(MAS4TE contract) but not implemented yet -> ignored with a warning.

`network` is a caller-supplied input (see digital_twin.utils.load_site_network
and e.g. data/bressoux/inputs/network.py for how a site builds one), NOT
built internally: this is what makes the engine reusable for a different LV
network without touching this package. A caller that reuses the SAME
`network` object across repeated run_digital_twin calls is responsible for
copying it first (net.load/net.sgen are cleared and repopulated in place
every call) -- see digital_twin.data_processing.pv.run_pv_scenario for the
equivalent copy.deepcopy(network['net']) pattern.
"""

import datetime
import os

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

from .utils import _config_to_dict

# ============================================================
#  1. DATA INGESTION / INPUT VALIDATION
# ============================================================
def _validate_inputs(config, metadata, households, pv_profiles, schedules_BSS):
    """Check that injected DataFrames have the expected structure.

    Returns
    -------
    warnings : list[str]
        Non-fatal issues (e.g. optional inputs missing, unsupported columns).

    Raises
    ------
    ValueError
        On fatal problems (missing mandatory columns, empty households,
        inconsistent lengths).
    """
    warnings = []

    # --- metadata ---
    if metadata is None or len(metadata) == 0:
        raise ValueError("`metadata` is empty or None: at least one household is required.")
    missing_cols = {'profile_id', 'source_street'} - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"`metadata` is missing required column(s): {sorted(missing_cols)}")

    # --- households (consumption timeseries) ---
    if households is None or len(households) == 0:
        raise ValueError("`households` is empty or None: at least one timestep of data is required.")
    missing_profiles = sorted(
        pid for pid in metadata['profile_id'].unique()
        if f"profile_{pid}" not in households.columns
    )
    if missing_profiles:
        preview = missing_profiles[:10]
        raise ValueError(
            "`households` is missing column(s) for profile_id(s) referenced by "
            f"`metadata`: {preview}{' ...' if len(missing_profiles) > 10 else ''}"
        )

    # --- config sanity ---
    # (required keys are always present at this point: _config_to_dict merges
    #  the caller's config with defaults before _validate_inputs is called;
    #  here we only check that the VALUES make physical sense.)
    if config['v_min'] >= config['v_max']:
        raise ValueError(f"config['v_min'] ({config['v_min']}) must be < config['v_max'] ({config['v_max']}).")
    if config['loading_max'] <= 0:
        raise ValueError(f"config['loading_max'] ({config['loading_max']}) must be positive.")
    if not (0 < config['cos_phi'] <= 1):
        raise ValueError(f"config['cos_phi'] ({config['cos_phi']}) must be in (0, 1].")
    if config['n_timesteps'] > len(households):
        raise ValueError(
            f"config['n_timesteps'] ({config['n_timesteps']}) exceeds the number of "
            f"timesteps available in `households` ({len(households)})."
        )

    # --- pv_profiles (optional) ---
    # Expected format: columns 'pv_{bus_id}' (integer bus_id), time index aligned
    # on households ('Mon 00:00'..'Sun 23:45'), values in MW (same unit as
    # households, see run_digital_twin). Here we validate the column shape and
    # the time coverage; bus existence and index alignment are checked at
    # injection time (_inject_pv_sgens / _verify_temporal_alignment).
    if pv_profiles is not None and len(pv_profiles) > 0:
        bad_cols = [c for c in pv_profiles.columns
                    if not (str(c).startswith('pv_') and str(c)[3:].isdigit())]
        if bad_cols:
            raise ValueError(
                "`pv_profiles` a des colonnes hors format 'pv_{bus_id}' (bus_id entier) : "
                f"{bad_cols[:10]}{' ...' if len(bad_cols) > 10 else ''}"
            )
        if config['n_timesteps'] > len(pv_profiles):
            raise ValueError(
                f"config['n_timesteps'] ({config['n_timesteps']}) depasse le nombre de pas "
                f"disponibles dans `pv_profiles` ({len(pv_profiles)})."
            )

    # --- schedules_BSS: not supported yet ---
    if schedules_BSS is not None and len(schedules_BSS) > 0:
        warnings.append("`schedules_BSS` was provided but is not yet supported (loads + PV only) -- ignored.")

    return warnings


# ============================================================
#  2. LOAD ASSIGNMENT
#  Consumes the `network` (caller-supplied, see module docstring) and the
#  `households` consumption timeseries (an interface contract: it does not
#  matter who filled it or how -- see digital_twin/data_processing/profiles.py
#  for the Bressoux upstream generator).
# ============================================================
def _assign_loads(network, metadata, households, config):
    """Assign a load profile to each LV charge bus, plus the fixed deterministic
    profiles of the network's MV load facilities (network['mv_load_names']).

    Uses the household->street mapping (light street-based association; global
    draw for orphan buses) and computes reactive power from cos_phi.

    Returns
    -------
    df_profiles_p : pd.DataFrame   # active power per load [MW], n_timesteps rows
    df_profiles_q : pd.DataFrame   # reactive power per load [MVAr]
    bus_to_profile : dict          # load name -> profile_id (metadata) or
                                    # facility name (MV load), traceability
    warnings : list[str]           # e.g. households not matched to a street
    """
    warnings = []
    rng = np.random.default_rng(config['seed'])

    net = network['net']
    buses = network['buses']
    mv_nodes = network['mv_nodes']
    mv_load_names = network['mv_load_names']

    # street -> [profile_id, ...] ; global fallback pool for unmatched buses
    profiles_by_street = metadata.groupby('source_street')['profile_id'].apply(list).to_dict()
    all_ids = metadata['profile_id'].tolist()

    net.load = net.load.iloc[0:0]

    profiles_dict = {}
    bus_to_profile = {}

    for (street, num), b in buses.items():
        load_name = f"Load_{b}"
        pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)

        if street in profiles_by_street:
            pick = rng.choice(profiles_by_street[street])
        else:
            # Global draw, expected for streets with no dedicated profile in
            # metadata (normal behaviour, not an anomaly to report).
            pick = rng.choice(all_ids)

        profiles_dict[load_name] = households[f"profile_{pick}"].values
        bus_to_profile[load_name] = pick

    # --- Fixed profiles for the network's MV load buses (rest home,
    # horticulture centre, sports hall, school, ...) -- attached here rather
    # than left at zero load.
    mv_profiles_path = os.path.join(config['data_path'], "out", "processed", "mv_load_profiles.csv")
    if os.path.exists(mv_profiles_path):
        mv_load_profiles = pd.read_csv(mv_profiles_path, index_col=0)
        for name in mv_load_names:
            b = mv_nodes[name]
            load_name = f"Load_{b}"
            pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)
            profiles_dict[load_name] = mv_load_profiles[name].values
            bus_to_profile[load_name] = name
    else:
        warnings.append(
            f"MV load profiles file not found ({mv_profiles_path}) -- "
            f"{', '.join(mv_load_names)} left at zero load."
        )

    df_profiles_p = pd.DataFrame(profiles_dict).clip(lower=1e-4)

    tan_phi = np.tan(np.arccos(config['cos_phi']))
    df_profiles_q = df_profiles_p * tan_phi

    return df_profiles_p, df_profiles_q, bus_to_profile, warnings


# ============================================================
#  2bis. PV INJECTION (native capability)
# ============================================================
def _inject_pv_sgens(net, pv_profiles):
    """Create one sgen per 'pv_{bus_id}' column of `pv_profiles`, on `net`.

    pandapower convention: sgen p_mw POSITIVE = injection (NOT a negative load).
    p_mw=q_mvar=0 at creation (filled per timestep by ConstControl in
    _run_powerflow). q_mvar stays 0 for the whole run (inverters as pure active
    power). net.sgen is emptied first so sgens never accumulate across calls.

    Returns
    -------
    sgen_bus : dict {index in net.sgen -> bus_id}
    """
    net.sgen = net.sgen.iloc[0:0]
    valid_bus = set(net.bus.index)
    sgen_bus = {}
    for col in pv_profiles.columns:
        # 'pv_123' -> 123.
        bus_id = int(str(col).split('_', 1)[1])
        if bus_id not in valid_bus:
            raise ValueError(
                f"pv_profiles: column {col!r} targets a bus_id {bus_id} absent from the network."
            )
        idx = pp.create_sgen(net, bus=bus_id, p_mw=0.0, q_mvar=0.0, name=str(col))
        sgen_bus[idx] = bus_id
    return sgen_bus


def _verify_temporal_alignment(df_load_p, df_pv_p):
    """Guardrail: loads and PV must share the SAME index (same length, same
    labels, same order). Step-by-step mapping in the power flow is only correct
    if they do. Raise a clear error rather than let a silent offset skew the net
    production/consumption balance."""
    if len(df_load_p) != len(df_pv_p):
        raise ValueError(
            f"Different lengths: load={len(df_load_p)} steps, PV={len(df_pv_p)} steps."
        )
    if list(df_load_p.index) != list(df_pv_p.index):
        raise ValueError(
            "Different index between load and PV production profiles -- "
            "step-by-step alignment is no longer mechanically guaranteed."
        )
    return True


# ============================================================
#  3. TIME-SERIES POWER FLOW
# ============================================================
def _run_powerflow(net, df_profiles_p, df_profiles_q, config, df_pv_p=None):
    """Attach controllers and run the time-series power flow.

    Loads are ALWAYS included. When `df_pv_p` is provided (columns 'pv_{bus_id}',
    MW; sgens already created by _inject_pv_sgens), the PV layer is added and the
    extra flow signals (direction/reversal) useful for PV impact analysis
    (overvoltage, reverse flows) are logged too. When `df_pv_p` is None, the
    power flow is loads-only.

    Returns
    -------
    dict : 'vm', 'line_load', 'trafo_load' (always) and, with PV,
    'line_p_from', 'trafo_p_hv', 'ext_grid_p', 'sgen_p_out'.
    """
    net.controller = net.controller.iloc[0:0]  # avoid stacking controllers on repeated calls

    data_source_p = DFData(df_profiles_p)
    data_source_q = DFData(df_profiles_q)

    for i, load in net.load.iterrows():
        load_column = load["name"]
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=data_source_p, profile_name=load_column)
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=data_source_q, profile_name=load_column)

    with_pv = df_pv_p is not None
    if with_pv:
        data_source_pv = DFData(df_pv_p)
        for i, sgen in net.sgen.iterrows():
            ConstControl(net, element='sgen', variable='p_mw', element_index=[i],
                         data_source=data_source_pv, profile_name=sgen["name"])

    n_timesteps = config['n_timesteps']
    # output_path=None -> results collected in memory only (ow.output), no CSV
    # round-trip through disk: cleaner for a callable module.
    ow = OutputWriter(net, time_steps=range(n_timesteps), output_path=None)
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')
    if with_pv:
        ow.log_variable('res_line', 'p_from_mw')     # sign: flow direction (reverse flows)
        ow.log_variable('res_trafo', 'p_hv_mw')      # sign: <0 = reverse HV->LV flow on this transformer
        ow.log_variable('res_ext_grid', 'p_mw')      # sign: <0 = network exports upstream
        ow.log_variable('res_sgen', 'p_mw')

    run_timeseries(net, time_steps=range(n_timesteps), verbose=False)

    out = {
        'vm': ow.output['res_bus.vm_pu'],
        'line_load': ow.output['res_line.loading_percent'],
        'trafo_load': ow.output['res_trafo.loading_percent'],
    }
    if with_pv:
        out['line_p_from'] = ow.output['res_line.p_from_mw']
        out['trafo_p_hv'] = ow.output['res_trafo.p_hv_mw']
        out['ext_grid_p'] = ow.output['res_ext_grid.p_mw']
        out['sgen_p_out'] = ow.output['res_sgen.p_mw']
    return out


# ============================================================
#  4. RESULTS EVALUATION (verdicts)
# ============================================================
def _collect_violations(df, metric, threshold, mode):
    """Scan a (n_timesteps x n_element) DataFrame for out-of-bounds values.

    mode: 'below' -> df < threshold ; 'above' -> df > threshold.
    Returns a list of violation dicts, one per (timestep, element) breach.
    """
    mask = (df < threshold) if mode == "below" else (df > threshold)
    if not mask.values.any():
        return []
    detail = f"{'<' if mode == 'below' else '>'} {threshold}"
    violations = []
    for (timestep, element), value in df.where(mask).stack().items():
        violations.append({
            'element': element,
            'metric': metric,
            'value': float(value),
            'timestep': int(timestep) if isinstance(timestep, (int, np.integer)) else timestep,
            'detail': detail,
        })
    return violations


def _aggregate_profile_summary(df_profiles_p):
    """Aggregated metrics: sum of the active power of all loads at each step,
    reported in kW for readability."""
    agg_mw = df_profiles_p.sum(axis=1)
    peak_mw = float(agg_mw.max())
    mean_mw = float(agg_mw.mean())
    summary = {
        'peak_agg_kw': peak_mw * 1000.0,
        'mean_agg_kw': mean_mw * 1000.0,
        'load_factor_agg': (mean_mw / peak_mw) if peak_mw > 0 else None,
    }
    return summary, agg_mw


def _evaluate(vm, line_load, trafo_load, config):
    """Turn raw results into a network verdict + list of violations.

    Returns
    -------
    summary : dict           # network_valid, min_voltage, max_loading, counts
    violations : list[dict]  # one entry per (element, timestep) out of bounds
    """
    v_min, v_max = config['v_min'], config['v_max']
    load_max = config['loading_max']

    violations = []
    if vm is not None:
        violations += _collect_violations(vm, 'vm_pu', v_min, 'below')
        violations += _collect_violations(vm, 'vm_pu', v_max, 'above')
    if line_load is not None:
        violations += _collect_violations(line_load, 'line_loading_pct', load_max, 'above')
    if trafo_load is not None:
        violations += _collect_violations(trafo_load, 'trafo_loading_pct', load_max, 'above')

    n_voltage_violations = sum(1 for v in violations if v['metric'] == 'vm_pu')
    n_loading_violations = len(violations) - n_voltage_violations

    summary = {
        'network_valid': len(violations) == 0,
        'min_voltage_pu': float(vm.values.min()) if vm is not None else None,
        'max_voltage_pu': float(vm.values.max()) if vm is not None else None,
        'max_line_loading_pct': float(line_load.values.max()) if line_load is not None else None,
        'max_trafo_loading_pct': float(trafo_load.values.max()) if trafo_load is not None else None,
        'n_voltage_violations': n_voltage_violations,
        'n_loading_violations': n_loading_violations,
    }
    return summary, violations


# ============================================================
#  5. OUTPUT FORMATTING
# ============================================================
def _format_output(summary, violations, warnings):
    """Assemble the single output DataFrame (verdict + violations + warnings).

    Long format, one row per record, so a MAS4TE agent can filter by
    `category`:
        - category == 'summary'   -> the verdict (is it viable?)
        - category == 'violation' -> where/when limits were breached
        - category == 'warning'   -> issues during the run (unmatched households...)
    """
    rows = []

    # Verdict lines
    for metric, value in summary.items():
        rows.append({'category': 'summary', 'element': 'network',
                     'metric': metric, 'value': value,
                     'timestep': None, 'detail': None})

    # Violation lines
    for v in violations:
        rows.append({'category': 'violation', 'element': v.get('element'),
                     'metric': v.get('metric'), 'value': v.get('value'),
                     'timestep': v.get('timestep'), 'detail': v.get('detail')})

    # Warning lines
    for w in warnings:
        rows.append({'category': 'warning', 'element': None,
                     'metric': None, 'value': None,
                     'timestep': None, 'detail': w})

    return pd.DataFrame(rows, columns=[
        'category', 'element', 'metric', 'value', 'timestep', 'detail'])


def _format_report_value(value):
    """Round floats to 4 decimals for report readability; leave bools/ints/
    strings/None as-is (isinstance(True, int) is True in Python, so bool is
    checked first)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def save_run_report(out, out_path):
    """Render the long-format `out` DataFrame (see run_digital_twin/
    _format_output) as a human-readable Markdown report and write it to
    `out_path`.

    Markdown rather than plain text: the summary/violations tables and the
    warnings list render directly, while the file still reads fine as plain
    text if opened without a renderer.

    Returns
    -------
    out_path : str
    """
    summary = out[out['category'] == 'summary']
    violations = out[out['category'] == 'violation']
    run_warnings = out[out['category'] == 'warning']

    network_valid = summary.set_index('metric')['value'].get('network_valid')

    lines = [
        "# Digital twin run report",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"**Network valid:** {network_valid}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {row['metric']} | {_format_report_value(row['value'])} |" for _, row in summary.iterrows()]

    lines += ["", f"## Violations ({len(violations)})", ""]
    if len(violations) == 0:
        lines.append("No violations.")
    else:
        lines += ["| Element | Metric | Value | Timestep | Detail |", "|---|---|---|---|---|"]
        lines += [
            f"| {row['element']} | {row['metric']} | {row['value']:.4f} | "
            f"{row['timestep']} | {row['detail']} |"
            for _, row in violations.iterrows()
        ]

    lines += ["", f"## Warnings ({len(run_warnings)})", ""]
    if len(run_warnings) == 0:
        lines.append("No warnings.")
    else:
        lines += [f"- {row['detail']}" for _, row in run_warnings.iterrows()]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path


def run_digital_twin(config, metadata, network, households,
                     pv_profiles=None, schedules_BSS=None):
    """Run the digital twin on a given network and return a network verdict.

    Parameters
    ----------
    config : dict                # tunable simulation/validation parameters
    metadata : pd.DataFrame      # per-household attributes (profile_id, source_street, ...)
    network : dict                # caller-supplied network bundle, see
        {"net", "buses", "mv_nodes", "mv_load_names", ...} -- built by a
        site-specific inputs/network.py, see digital_twin.utils.load_site_network.
        `net` is mutated in place (loads/sgens cleared and repopulated) -- copy
        it yourself before calling if you plan to reuse `network` across calls.
    households : pd.DataFrame    # consumption profiles (n_timesteps x n_profiles), MW
    pv_profiles : pd.DataFrame, optional
        PV production. Columns 'pv_{bus_id}' (integer network bus_id), time index
        aligned on households ('Mon 00:00'..'Sun 23:45'), values in MW (SAME
        unit as households -- if the production comes from
        pv.build_pv_production, which outputs kW, convert /1000 upstream, see
        main.py). When provided: injected as sgen, loads + PV power flow. When
        None: no PV (loads-only).
    schedules_BSS : pd.DataFrame, optional   # not supported yet (warning if provided)

    Returns
    -------
    out : pd.DataFrame
        Long-format result: summary (verdict) + violations + warnings.
    """
    # fill in any config keys the caller didn't set with their defaults
    cfg = _config_to_dict(config)

    warnings = _validate_inputs(cfg, metadata, households, pv_profiles, schedules_BSS)
    warnings += network.get('warnings', [])

    df_p, df_q, bus_to_profile, w_assign = _assign_loads(
        network, metadata, households, cfg)
    warnings += w_assign

    net = network['net']

    # --- PV injection (native capability, driven by the pv_profiles argument) ---
    # pv_profiles provided: create one sgen per 'pv_{bus_id}' column and run the
    # loads + PV power flow together. pv_profiles=None: no PV, loads-only.
    df_pv_p = None
    if pv_profiles is not None and len(pv_profiles) > 0:
        # Positional alignment: _assign_loads builds df_p via .values (text index
        # 'Mon 00:00'.. is lost -> RangeIndex 0..N-1). Put pv_profiles on the same
        # RangeIndex BEFORE the alignment check and the power flow (DFData/
        # ConstControl index by position).
        df_pv_p = pv_profiles.reset_index(drop=True)
        _verify_temporal_alignment(df_p, df_pv_p)   # index guardrail
        _inject_pv_sgens(net, df_pv_p)

    pf = _run_powerflow(net, df_p, df_q, cfg, df_pv_p=df_pv_p)
    vm, line_load, trafo_load = pf['vm'], pf['line_load'], pf['trafo_load']

    summary, violations = _evaluate(vm, line_load, trafo_load, cfg)

    agg_summary, agg_mw = _aggregate_profile_summary(df_p)
    summary.update(agg_summary)

    out = _format_output(summary, violations, warnings)

    # Optional export of the full aggregate profile (672 steps), off by default
    # to keep the standard output light for MAS4TE consumers.
    if cfg.get('export_aggregate_profile'):
        profile_rows = pd.DataFrame({
            'category': 'profile_agg', 'element': None, 'metric': 'p_agg_mw',
            'value': agg_mw.values, 'timestep': range(len(agg_mw)), 'detail': None,
        })
        out = pd.concat([out, profile_rows], ignore_index=True)

    return out
