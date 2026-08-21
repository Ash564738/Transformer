# severity.py
from __future__ import annotations
import logging
from typing import Optional
import numpy as np, pandas as pd
from config import config as cfg
logger = logging.getLogger(__name__)
STANDARD = cfg.STANDARD; IEC_STANDARD = cfg.IEC_STANDARD
SEVERITY_GASES = tuple(cfg.SEVERITY_REQUIRED_GASES)
COMBUSTIBLE_GASES = ("h2", "ch4", "c2h6", "c2h4", "c2h2", "co")
IEEE_TABLE_1 = cfg.TABLE_1_90TH; IEEE_TABLE_2 = cfg.TABLE_2_95TH
IEEE_TABLE_3 = cfg.TABLE_3_DELTA_95TH; IEEE_TABLE_4 = cfg.TABLE_4_RATE_95TH
STATUS_LABELS = {0: "INSUFFICIENT_DATA", 1: "STATUS_1", 2: "STATUS_2", 3: "STATUS_3"}
STATUS_ACTIONS = {0: "REVIEW_DATA", 1: "ROUTINE_DGA_SURVEILLANCE", 2: "CONFIRM_AND_INCREASE_SURVEILLANCE", 3: "INVESTIGATE_AND_INCREASE_SURVEILLANCE"}
STATUS_DESCRIPTIONS = {0: "Required DGA inputs are insufficient for complete IEEE C57.104-2019 screening.", 1: "DGA Status 1: low gas levels and no applicable higher-status trigger.", 2: "DGA Status 2: intermediate gas level and/or possible gassing evidence.", 3: "DGA Status 3: high gas level and/or probable active-gassing evidence."}

def _finite(value) -> Optional[float]:
    try: x = float(value)
    except (TypeError, ValueError): return None
    return x if np.isfinite(x) else None

def _positive(value):
    x = _finite(value)
    if x is None or x < 0: return None
    return x

def _safe_ratio(a, b):
    x, y = _finite(a), _finite(b)
    if x is None or y is None or y <= 0: return np.nan
    return x / y

def _safe_age(row):
    for column in ("transformer_age_years", "age_years"):
        value = _positive(row.get(column, np.nan))
        if value is not None: return value
    sample_year = _positive(row.get("sample_year", np.nan)); energized_year = _positive(row.get("year_energized", np.nan))
    if sample_year is not None and energized_year is not None and sample_year >= energized_year: return sample_year - energized_year
    sample_day = pd.to_datetime(row.get("sample_day", pd.NaT), errors="coerce"); energized_day = pd.to_datetime(row.get("energized_day", pd.NaT), errors="coerce")
    if pd.notna(sample_day) and pd.notna(energized_day):
        days = (sample_day - energized_day).days
        if days >= 0: return days / 365.25
    return np.nan

def _age_bucket(age):
    if not np.isfinite(age) or age <= 0: return "unknown"
    if age < 10: return "1_9"
    if age <= 30: return "10_30"
    return "gt_30"

def _o2_n2_ratio(row):
    for column in ("o2_n2_ratio", "o2n2_ratio"):
        value = _positive(row.get(column, np.nan))
        if value is not None: return value
    return _safe_ratio(row.get("o2", np.nan), row.get("n2", np.nan))

def _gas_snapshot(row): return {gas: _positive(row.get(gas, np.nan)) for gas in SEVERITY_GASES}
def _complete(values): return all(values.get(gas) is not None for gas in SEVERITY_GASES)

def _numeric_tdcg(values):
    vals = [values.get(gas) for gas in COMBUSTIBLE_GASES]
    if not all(value is not None for value in vals): return np.nan
    return float(sum(vals))

def _previous_valid_sample(group, current_position):
    if current_position <= 0: return None
    for position in range(current_position - 1, -1, -1):
        row = group.iloc[position]
        if pd.notna(row.get("sample_day", pd.NaT)): return row
    return None

def _single_sample_evidence(values, section, age_bucket):
    t1 = IEEE_TABLE_1[section][age_bucket]; t2 = IEEE_TABLE_2[section][age_bucket]
    t1_exceeding, t2_exceeding = [], []
    t1_ratios, t2_ratios = {}, {}
    for gas in SEVERITY_GASES:
        value = values.get(gas)
        if value is None: continue
        t1_limit = float(t1[gas]); t2_limit = float(t2[gas])
        t1_ratios[gas] = value / t1_limit if t1_limit > 0 else np.nan
        t2_ratios[gas] = value / t2_limit if t2_limit > 0 else np.nan
        if value > t1_limit: t1_exceeding.append(gas)
        if value > t2_limit: t2_exceeding.append(gas)
    return {"table1": dict(t1), "table2": dict(t2), "table1_exceeding_gases": t1_exceeding, "table2_exceeding_gases": t2_exceeding, "table1_exceedance_count": len(t1_exceeding), "table2_exceedance_count": len(t2_exceeding), "table1_max_ratio": max([r for r in t1_ratios.values() if np.isfinite(r) and r > 1.0], default=1.0), "table2_max_ratio": max([r for r in t2_ratios.values() if np.isfinite(r) and r > 1.0], default=1.0), "table1_ratios": t1_ratios, "table2_ratios": t2_ratios}

def _compute_delta(current, previous, section):
    if previous is None: return {"deltas": {}, "exceeding": [], "available": False, "max_ratio": 1.0, "numeric_exceeding_count": 0, "any_exceeding": False, "any_increase_only": False}
    limits = IEEE_TABLE_3[section]
    deltas, exceeding = {}, []
    numeric_ratios = []
    numeric_count = 0
    any_increase_only = False
    for gas in SEVERITY_GASES:
        current_value = _positive(current.get(gas, np.nan)); previous_value = _positive(previous.get(gas, np.nan))
        if current_value is None or previous_value is None: continue
        delta = current_value - previous_value; deltas[gas] = delta
        limit = limits.get(gas)
        if limit is None:
            if gas == "c2h2" and delta > 0: exceeding.append(gas); any_increase_only = True
            continue
        ratio = abs(delta) / float(limit) if float(limit) > 0 else np.nan
        if np.isfinite(ratio): numeric_ratios.append(ratio)
        if abs(delta) > float(limit): exceeding.append(gas); numeric_count += 1
    return {"deltas": deltas, "exceeding": exceeding, "available": True, "max_ratio": max([r for r in numeric_ratios if r > 1.0], default=1.0), "numeric_exceeding_count": numeric_count, "any_exceeding": bool(exceeding), "any_increase_only": any_increase_only}

def _candidate_rate_windows(group, current_position):
    work = group.iloc[: current_position + 1].copy(); work["_sample_day_dt"] = pd.to_datetime(work["sample_day"], errors="coerce"); work = work.dropna(subset=["_sample_day_dt"])
    if len(work) < cfg.RATE_MIN_POINTS: return []
    candidates = []; max_points = min(cfg.RATE_MAX_POINTS, len(work))
    for point_count in range(max_points, cfg.RATE_MIN_POINTS - 1, -1):
        window = work.iloc[-point_count:].copy()
        span_months = (window["_sample_day_dt"].iloc[-1] - window["_sample_day_dt"].iloc[0]).total_seconds() / 86400.0 / 30.4375
        if cfg.RATE_MIN_MONTHS < span_months < cfg.RATE_MAX_MONTHS: candidates.append((point_count, span_months, window))
    return candidates

def _linear_slope(days, values):
    if len(days) < 3 or np.ptp(days) <= 0: return np.nan
    x = (days - days[0]) / 365.25
    try: slope = np.polyfit(x, values, 1)[0]
    except Exception: return np.nan
    return float(slope) if np.isfinite(slope) else np.nan

def _compute_rate(group, current_position, section, all_below_table1):
    if not all_below_table1: return {"rates": {}, "window": None, "exceeding": [], "available": False, "max_ratio": 1.0, "numeric_exceeding_count": 0, "any_increase_only": False}
    candidates = _candidate_rate_windows(group, current_position)
    if not candidates: return {"rates": {}, "window": None, "exceeding": [], "available": False, "max_ratio": 1.0, "numeric_exceeding_count": 0, "any_increase_only": False}
    point_count, span_months, window = candidates[0]
    days = pd.to_datetime(window["sample_day"], errors="coerce").astype("int64").to_numpy(dtype=float) / 86_400_000_000_000.0
    period_bucket = "4_9" if span_months <= 9.0 else "10_24"; limits = IEEE_TABLE_4[section][period_bucket]
    rates, exceeding = {}, {}
    ratios = []; numeric_exceeding_count = 0; any_increase_only = False
    for gas in SEVERITY_GASES:
        values = pd.to_numeric(window[gas], errors="coerce").to_numpy(dtype=float); valid = np.isfinite(values)
        if valid.sum() < 3: rates[gas] = np.nan; continue
        slope = _linear_slope(days[valid], values[valid]); rates[gas] = slope
        if not np.isfinite(slope) or slope <= 0: continue
        limit = limits.get(gas)
        if limit is None:
            if gas == "c2h2": exceeding.append(gas); any_increase_only = True
            continue
        ratio = slope / float(limit) if float(limit) > 0 else np.nan
        if np.isfinite(ratio): ratios.append(ratio)
        if slope > float(limit): exceeding.append(gas); numeric_exceeding_count += 1
    return {"rates": rates, "window": {"points": int(point_count), "span_months": float(span_months)}, "exceeding": exceeding, "available": True, "max_ratio": max([r for r in ratios if r > 1.0], default=1.0), "numeric_exceeding_count": numeric_exceeding_count, "any_increase_only": any_increase_only}

def _iec_ratios(values):
    return {"ch4_h2": _safe_ratio(values.get("ch4"), values.get("h2")), "c2h4_c2h6": _safe_ratio(values.get("c2h4"), values.get("c2h6")), "c2h2_c2h4": _safe_ratio(values.get("c2h2"), values.get("c2h4")), "c2h2_ch4": _safe_ratio(values.get("c2h2"), values.get("ch4")), "c2h6_c2h2": _safe_ratio(values.get("c2h6"), values.get("c2h2")), "c2h2_h2": _safe_ratio(values.get("c2h2"), values.get("h2")), "co2_co": _safe_ratio(values.get("co2"), values.get("co"))}

def _iec_ratio_interpretation(values):
    ratios = _iec_ratios(values)
    return {"standard": IEC_STANDARD, "available_ratio_count": int(sum(np.isfinite(v) for v in ratios.values())), "ratios": ratios, "interpretation_flags": ["IEC_60599_2022_COMPLEMENTARY_INTERPRETATION_ONLY", "RATIOS_DO_NOT_OVERRIDE_IEEE_STATUS"], "overrides_ieee_status": False}

def _empty_evidence():
    return {"table1_exceeding_gases": [], "table2_exceeding_gases": [], "table3_exceeding_gases": [], "table4_exceeding_gases": [], "table1_max_ratio": 1.0, "table2_max_ratio": 1.0, "table3_max_ratio": 1.0, "table4_max_ratio": 1.0, "table3_available": False, "table4_available": False, "table3_any_increase_only": False, "table4_any_increase_only": False, "table1_exceedance_count": 0, "table2_exceedance_count": 0, "table3_numeric_exceedance_count": 0, "table4_numeric_exceedance_count": 0, "all_below_table1": False}

def _evaluate_row(row, history, position):
    values = _gas_snapshot(row); ratio = _o2_n2_ratio(row); age = _safe_age(row); age_bucket = _age_bucket(age); complete = _complete(values)
    if not complete or not np.isfinite(ratio):
        status = 0; evidence = _empty_evidence(); selected_section = None; iec = _iec_ratio_interpretation(values)
    else:
        selected_section = "LE_0_2" if ratio <= 0.2 else "GT_0_2"; single = _single_sample_evidence(values, selected_section, age_bucket); previous = _previous_valid_sample(history, position); delta = _compute_delta(row, previous, selected_section); all_below_table1 = single["table1_exceedance_count"] == 0; rate = _compute_rate(history, position, selected_section, all_below_table1)
        if single["table2_exceeding_gases"] or rate["exceeding"]: status = 3
        elif single["table1_exceeding_gases"] or delta["exceeding"]: status = 2
        else: status = 1
        evidence = {"table1_exceeding_gases": single["table1_exceeding_gases"], "table2_exceeding_gases": single["table2_exceeding_gases"], "table3_exceeding_gases": delta["exceeding"], "table4_exceeding_gases": rate["exceeding"], "table1_max_ratio": single["table1_max_ratio"], "table2_max_ratio": single["table2_max_ratio"], "table3_max_ratio": delta["max_ratio"], "table4_max_ratio": rate["max_ratio"], "table3_available": delta["available"], "table4_available": rate["available"], "table3_any_increase_only": delta["any_increase_only"], "table4_any_increase_only": rate["any_increase_only"], "table1_exceedance_count": single["table1_exceedance_count"], "table2_exceedance_count": single["table2_exceedance_count"], "table3_numeric_exceedance_count": delta["numeric_exceeding_count"], "table4_numeric_exceedance_count": rate["numeric_exceeding_count"], "all_below_table1": all_below_table1, "deltas": delta["deltas"], "rates": rate["rates"], "rate_window": rate["window"]}
        iec = _iec_ratio_interpretation(values)
    basis = []
    if evidence["table2_exceeding_gases"]: basis.append("TABLE_2_ABOVE_95TH")
    if evidence["table4_exceeding_gases"]: basis.append("TABLE_4_RATE_ABOVE_95TH")
    if evidence["table1_exceeding_gases"]: basis.append("TABLE_1_ABOVE_90TH")
    if evidence["table3_exceeding_gases"]: basis.append("TABLE_3_DELTA_ABOVE_95TH")
    if status == 1: basis.append("ALL_APPLICABLE_IEEE_LIMITS_NOT_EXCEEDED")
    if not complete: basis.append("MANDATORY_GAS_MISSING")
    if not np.isfinite(ratio): basis.append("O2_N2_RATIO_MISSING")
    standard_trigger_count = sum(bool(x) for x in (evidence["table1_exceeding_gases"], evidence["table2_exceeding_gases"], evidence["table3_exceeding_gases"], evidence["table4_exceeding_gases"]))
    status3_standardized_exceedance = max(evidence["table2_max_ratio"], evidence["table4_max_ratio"], 1.0)
    if evidence["table2_exceeding_gases"] or evidence["table4_exceeding_gases"]: max_standardized_exceedance = status3_standardized_exceedance
    elif evidence["table1_exceeding_gases"] or evidence["table3_exceeding_gases"]: max_standardized_exceedance = max(evidence["table1_max_ratio"], evidence["table3_max_ratio"], 1.0)
    else: max_standardized_exceedance = 1.0
    return {"ieee_dga_status": int(status), "ieee_dga_status_label": STATUS_LABELS[status], "severity_label_text": STATUS_LABELS[status], "severity_source": STANDARD, "severity_score_type": "IEEE_ORDINAL_STATUS_PLUS_STANDARD_EVIDENCE_VECTOR", "severity_is_not_a_health_score": True, "severity_is_failure_probability": False, "severity_composite_weighted": False, "severity_uses_manual_weights": False, "severity_manual_weights": False, "severity_weighted_sum_used": False, "severity_anomaly_used": False, "severity_nei_used": False, "severity_inference_stage": "IEEE_C57_104_2019_RULE_ENGINE", "ieee_dga_status_description": STATUS_DESCRIPTIONS[status], "ieee_dga_status_reason": " | ".join(basis), "ieee_recommended_action": STATUS_ACTIONS[status], "ieee_confirmation_required": bool(status in {2, 3}), "ieee_o2_n2_ratio": ratio, "ieee_o2_n2_section": selected_section, "ieee_o2_n2_unknown": not np.isfinite(ratio), "ieee_transformer_age_years": age, "ieee_age_bucket": age_bucket, "ieee_table1_exceeding_gases": evidence["table1_exceeding_gases"], "ieee_table2_exceeding_gases": evidence["table2_exceeding_gases"], "ieee_table3_exceeding_gases": evidence["table3_exceeding_gases"], "ieee_table4_exceeding_gases": evidence["table4_exceeding_gases"], "ieee_table1_max_exceedance_ratio": evidence["table1_max_ratio"], "ieee_table2_max_exceedance_ratio": evidence["table2_max_ratio"], "ieee_table3_max_exceedance_ratio": evidence["table3_max_ratio"], "ieee_table4_max_exceedance_ratio": evidence["table4_max_ratio"], "ieee_max_standardized_exceedance": max_standardized_exceedance, "ieee_max_status3_standardized_exceedance": status3_standardized_exceedance, "ieee_standard_trigger_count": int(standard_trigger_count), "ieee_table3_available": evidence["table3_available"], "ieee_table4_available": evidence["table4_available"], "ieee_delta_available": evidence["table3_available"], "ieee_rate_available": evidence["table4_available"], "ieee_rate_window_points": int(evidence["rate_window"]["points"]) if evidence.get("rate_window") else 0, "ieee_rate_span_months": float(evidence["rate_window"]["span_months"]) if evidence.get("rate_window") else np.nan, "ieee_delta": evidence.get("deltas", {}), "ieee_gas_rate_ppm_per_year": evidence.get("rates", {}), "ieee_tdcg_ppm": _numeric_tdcg(values), "ieee_mandatory_gas_complete": complete, "ieee_status_basis": basis, "ieee_standard_evidence_vector": {"status": int(status), "max_standardized_exceedance": float(max_standardized_exceedance), "max_status3_standardized_exceedance": float(status3_standardized_exceedance), "delta_exceedance_flag": int(bool(evidence["table3_exceeding_gases"])), "standard_trigger_count": int(standard_trigger_count)}, "iec_60599_standard": IEC_STANDARD, "iec_60599_ratio_available": iec["available_ratio_count"] >= 1, "iec_60599_ratio_count": int(iec["available_ratio_count"]), "iec_60599_ratios": iec["ratios"], "iec_60599_interpretation_flags": iec["interpretation_flags"], "iec_60599_overrides_ieee_status": False, "severity_engine_version": "IEEE_C57_104_2019_PRIMARY_v3_STANDARD_VECTOR_PLUS_OPERATIONAL_EXTREME_FLAG"}

def apply_severity(df: pd.DataFrame, nei_reference=None) -> pd.DataFrame:
    if df is None or df.empty: return df.copy() if df is not None else pd.DataFrame()
    required = {"transformer_id", "sample_day", *SEVERITY_GASES}; missing = sorted(required - set(df.columns))
    if missing: raise ValueError(f"IEEE C57.104-2019 severity requires columns: {missing}")
    out = df.copy(); out["sample_day"] = pd.to_datetime(out["sample_day"], errors="coerce")
    for gas in SEVERITY_GASES:
        out[gas] = pd.to_numeric(out[gas], errors="coerce"); out.loc[out[gas] < 0, gas] = np.nan
    out = out.sort_values(["transformer_id", "sample_day"], kind="mergesort").copy(); result_rows = []
    for transformer_id, group in out.groupby("transformer_id", sort=False):
        group = group.reset_index(drop=False).rename(columns={"index": "_orig_index"})
        for position in range(len(group)):
            result = _evaluate_row(group.iloc[position], group, position); result["_orig_index"] = group.iloc[position]["_orig_index"]; result_rows.append(result)
    result_df = pd.DataFrame(result_rows)
    if not result_df.empty:
        result_df = result_df.set_index("_orig_index").reindex(out.index)
        for column in result_df.columns:
            if column not in out.columns: out[column] = result_df[column]
        out["severity_status"] = out["ieee_dga_status"]; out["severity_label"] = out["ieee_dga_status_label"]; out["severity_condition"] = out["ieee_dga_status"]; out["severity_condition_label"] = out["ieee_dga_status_label"]; out["severity_action"] = out["ieee_recommended_action"]; out["severity_status_description"] = out["ieee_dga_status_description"]; out["severity_standard"] = STANDARD; out["severity_is_weighted"] = False; out["severity_is_not_a_health_score"] = True
    logger.info("IEEE C57.104-2019 severity distribution: %s", out["ieee_dga_status"].value_counts().sort_index().to_dict() if "ieee_dga_status" in out.columns else {})
    return out

def add_nei(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "nei_oil" not in out.columns: out["nei_oil"] = np.nan
    out["nei_used_for_severity"] = False
    return out

def fit_nei_reference(df: pd.DataFrame): return None

__all__ = ["apply_severity", "add_nei", "fit_nei_reference", "STANDARD", "IEC_STANDARD"]