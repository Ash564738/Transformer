# inference_service.py
from __future__ import annotations
import json, logging, shutil, tempfile, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from clean_dataset import clean_dataset
from consensus import apply_consensus, normalize_fault, unify_fault
from feature_engineering import build_training_features_from_clean
from logging_config import init_logging
from ranking import build_transformer_ranking, classify_fault_criticality, fault_criticality_source, log_ranking_diagnostics
from severity import apply_severity
from weak_supervision import SNORKEL_AVAILABLE, weak_supervision_pipeline, save_weak_supervision_artifacts
from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg
init_logging()
logger = logging.getLogger(__name__)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATASET_DIR / "processed"
STUDENT_MODEL_PATH = MODEL_DIR / "fault_classifier.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_ensemble.joblib"
UNLABELED_PATH = PROCESSED_DIR / "dga_unlabeled.parquet"
RANKING_PATH = PROCESSED_DIR / "transformer_ranking.parquet"
PROCESSED_OUTPUT_PATH = PROCESSED_DIR / "dga_unlabeled_processed.parquet"
STUDENT_COMPARISON_PATH = REPORT_DIR / "student_vs_traditional_by_transformer.csv"
INFERENCE_METADATA_PATH = PROCESSED_DIR / "dga_inference_metadata.json"

def _safe_int(value, default=0):
    try:
        if pd.isna(value): return int(default)
    except Exception: pass
    try: return int(value)
    except (TypeError, ValueError): return int(default)

def _log_stage(stage: int, total: int, title: str):
    logger.info("=" * 110)
    logger.info("PIPELINE STEP %d/%d: %s", stage, total, title)
    logger.info("=" * 110)

def _save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved CSV | %s | rows=%d | columns=%d", path, len(df), len(df.columns))

def _save_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved Parquet | %s | rows=%d | columns=%d", path, len(df), len(df.columns))

def _run_external_research_benchmark(operational_df: pd.DataFrame, weak_students: dict, seed: int):
    logger.info("RESEARCH BENCHMARK START")
    try:
        from train_unsupervised_models import (BENCHMARK_DIR, benchmark_traditional_combinations, benchmark_traditional_individual, benchmark_traditional_ppm_coverage, benchmark_weak_label_model_transfer, benchmark_weak_transfer, benchmark_weak_traditional_hybrids, diagnostic_method_summary, load_labeled_csv_data)
        from consensus import apply_consensus, pairwise_label_agreement
        labeled_raw = load_labeled_csv_data()
        labeled = apply_consensus(labeled_raw)
        logger.info("External labeled benchmark loaded | rows=%d | datasets=%s", len(labeled), labeled["source_dataset"].value_counts().to_dict() if "source_dataset" in labeled.columns else {})
        traditional_individual = benchmark_traditional_individual(labeled)
        traditional_combinations = benchmark_traditional_combinations(labeled, None)
        traditional_ppm = benchmark_traditional_ppm_coverage(labeled)
        pairwise = pairwise_label_agreement(labeled)
        method_summary = diagnostic_method_summary(labeled)
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        _save_csv(pairwise, BENCHMARK_DIR / "traditional_pairwise_agreement.csv")
        _save_csv(method_summary, BENCHMARK_DIR / "traditional_method_summary.csv")
        logger.info("Traditional benchmark complete | rows=%d", len(traditional_individual))
        try:
            from train_unsupervised_models import benchmark_supervised_models
            supervised = benchmark_supervised_models(labeled, seed)
        except Exception:
            logger.exception("Supervised benchmark failed")
            supervised = pd.DataFrame()
        if not supervised.empty: _save_csv(supervised, BENCHMARK_DIR / "supervised_fault_benchmark.csv")
        weak_label_transfer = pd.DataFrame()
        try:
            from weak_supervision import load_weak_supervision_artifact
            weak_payloads = {g: load_weak_supervision_artifact(g) for g in ("coarse", "fine")}
            weak_label_transfer = benchmark_weak_label_model_transfer(labeled, weak_payloads)
        except Exception:
            logger.exception("Direct weak-label transfer benchmark failed")
        if not weak_label_transfer.empty: _save_csv(weak_label_transfer, BENCHMARK_DIR / "weak_label_model_transfer_fault_benchmark.csv")
        if weak_students:
            try: weak_transfer = benchmark_weak_transfer(labeled, weak_students, seed)
            except Exception:
                logger.exception("Weak-transfer benchmark failed")
                weak_transfer = pd.DataFrame()
        else: weak_transfer = pd.DataFrame()
        if not weak_transfer.empty: _save_csv(weak_transfer, BENCHMARK_DIR / "weak_transfer_fault_benchmark.csv")
        try:
            hybrid = benchmark_weak_traditional_hybrids(labeled, weak_students, seed) if weak_students else pd.DataFrame()
        except Exception:
            logger.exception("Weak + traditional hybrid benchmark failed")
            hybrid = pd.DataFrame()
        if not hybrid.empty: _save_csv(hybrid, BENCHMARK_DIR / "weak_traditional_hybrid_benchmark.csv")
        logger.info("Benchmark outputs | traditional=%d rows | combinations=%d | supervised=%d | weak_label_transfer=%d | weak_transfer=%d | hybrid=%d", len(traditional_individual), len(traditional_combinations), len(supervised), len(weak_label_transfer), len(weak_transfer), len(hybrid))
        return {"status": "completed", "error": None, "traditional_individual": traditional_individual, "traditional_combinations": traditional_combinations, "traditional_ppm": traditional_ppm, "pairwise": pairwise, "method_summary": method_summary, "supervised": supervised, "weak_label_transfer": weak_label_transfer, "weak_transfer": weak_transfer, "hybrid": hybrid}
    except Exception:
        logger.exception("External research benchmark failed")
        return {"status": "failed", "error": "External research benchmark failed.", "traditional_individual": pd.DataFrame(), "traditional_combinations": pd.DataFrame(), "traditional_ppm": pd.DataFrame(), "pairwise": pd.DataFrame(), "method_summary": pd.DataFrame(), "supervised": pd.DataFrame(), "weak_label_transfer": pd.DataFrame(), "weak_transfer": pd.DataFrame(), "hybrid": pd.DataFrame()}

def _run_weak_students(df_coarse, df_fine, seed):
    try:
        from train_unsupervised_models import _train_weak_students
    except Exception:
        logger.exception("Cannot import research student trainer")
        return {"coarse": {}, "fine": {}}
    students = {"coarse": {}, "fine": {}}
    try:
        students["coarse"] = _train_weak_students(df_coarse, "coarse", seed)
        logger.info("Coarse student training complete | models=%d", len(students["coarse"]))
    except Exception: logger.exception("Coarse student training failed")
    try:
        students["fine"] = _train_weak_students(df_fine, "fine", seed)
        logger.info("Fine student training complete | models=%d", len(students["fine"]))
    except Exception: logger.exception("Fine student training failed")
    return students

def _decode_student_predictions(prediction, labels, default="ABSTAIN"):
    arr = np.asarray(prediction)
    if arr.ndim == 0: arr = arr.reshape(1)
    elif arr.ndim > 1: arr = arr.reshape(arr.shape[0], -1)[:, 0]
    decoded = []
    for value in arr:
        try:
            scalar = np.asarray(value).reshape(-1)[0]
            class_index = int(scalar)
            if 0 <= class_index < len(labels): decoded.append(labels[class_index])
            else: decoded.append(default)
        except Exception: decoded.append(default)
    return decoded

def _apply_research_students(df: pd.DataFrame, weak_students: dict):
    try:
        from train_unsupervised_models import _align_feature_frame
    except Exception:
        logger.exception("Cannot import student feature aligner.")
        return df
    out = df.copy()
    total_models = 0
    failed_models = 0
    for granularity, artifacts in weak_students.items():
        for key, artifact in artifacts.items():
            try:
                feature_names = artifact["features"]
                X = _align_feature_frame(out, feature_names, granularity)
                model = artifact["model"]
                labels = list(artifact["labels"])
                raw_prediction = model.predict(X)
                decoded = _decode_student_predictions(raw_prediction, labels, default="ABSTAIN")
                column = f"weak_student_{granularity}_{key}"
                out[column] = decoded
                if hasattr(model, "predict_proba"):
                    try:
                        proba = np.asarray(model.predict_proba(X), dtype=float)
                        if proba.ndim == 1: confidence = np.abs(proba)
                        elif proba.ndim == 2: confidence = np.max(proba, axis=1)
                        else: confidence = np.full(len(out), np.nan, dtype=float)
                        out[column + "_confidence"] = confidence
                    except Exception:
                        logger.warning("Student confidence failed | %s | %s", granularity, key, exc_info=True)
                total_models += 1
            except Exception:
                failed_models += 1
                logger.exception("Student prediction failed | %s | %s", granularity, key)
    logger.info("Applied research student models | success=%d | failed=%d", total_models, failed_models)
    return out

def _build_research_excel():
    try:
        from experiment import build_excel_report
        output_path = REPORT_DIR / "dga_research_report.xlsx"
        result = build_excel_report(REPORT_DIR, PROCESSED_DIR, output_path)
        logger.info("Excel research report saved | %s", result)
        return result
    except Exception as exc:
        logger.exception("Excel report generation failed")
        return None

def _safe_float(value, default=np.nan):
    try: x = float(value)
    except (TypeError, ValueError): return default
    return x if np.isfinite(x) else default

def _ui_status(status: int) -> str:
    return {0: "Insufficient data", 1: "Normal", 2: "Watch", 3: "High"}.get(_safe_int(status), "Insufficient data")

def _ui_severity_label(value):
    if value is None: return "INSUFFICIENT_DATA"
    text = str(value).strip().upper()
    if text in cfg.SEVERITY_ORDER: return text
    return cfg.ORDINAL_TO_SEVERITY.get(_safe_int(value), "INSUFFICIENT_DATA")

def _first(row, names, default=None):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if pd.notna(value): return value
    return default

def _normalize_series(series): return series.map(normalize_fault).fillna("ABSTAIN").astype(str)
def _student_feature_columns(): return list(cfg.COMMON_BENCHMARK_GASES)

def _prepare_student_matrix(df, feature_columns):
    feature_columns = list(feature_columns)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing: raise ValueError(f"Missing student-model features: {missing}")
    return df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _validate_student_artifact(artifact):
    if not isinstance(artifact, dict):
        return False
    if artifact.get("model") is None or not artifact.get("labels"):
        return False
    if str(artifact.get("granularity", "")).lower() != "fine":
        return False
    features = artifact.get("feature_cols", artifact.get("features", cfg.COMMON_BENCHMARK_GASES))
    return bool(features)

def _load_student_model():
    if not STUDENT_MODEL_PATH.exists(): return None
    try: artifact = joblib.load(STUDENT_MODEL_PATH)
    except Exception as exc:
        logger.warning("Stored student model cannot be loaded; rebuilding. Error=%s", exc)
        return None
    if not _validate_student_artifact(artifact):
        logger.warning("Stored student artifact is incompatible with current pipeline.")
        return None
    return artifact

def _train_student_fallback(df_weak: pd.DataFrame):
    target = "weak_fine_fault"
    abstain_column = "weak_fine_is_ABSTAIN"
    if target not in df_weak.columns:
        raise ValueError("No weak fine target column found for production fallback.")
    clean = df_weak.copy()
    if abstain_column in clean.columns:
        clean = clean[~clean[abstain_column].astype(bool)]
    clean[target] = clean[target].map(normalize_fault)
    clean = clean[clean[target].isin(cfg.BENCHMARK_FINE_CLASSES)].copy()
    if clean.empty:
        raise ValueError("No usable weak fine labels remain for student-model training.")
    if clean[target].nunique() < 2:
        raise ValueError("Production fine student requires at least two classes.")
    feature_columns = _student_feature_columns()
    X = _prepare_student_matrix(clean, feature_columns)
    y = clean[target].astype(str).to_numpy()
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X, y)
    classifier = model.named_steps["classifier"]
    classes = [str(v) for v in classifier.classes_]
    artifact = {
        "model": model,
        "granularity": "fine",
        "feature_mode": "gas_only",
        "feature_cols": feature_columns,
        "features": feature_columns,
        "diagnostic_methods": list(cfg.DIAGNOSTIC_METHODS),
        "labels": classes,
        "model_name": "extra_trees",
        "training_type": "weak_supervision_runtime_fallback",
        "training_dataset": "operational_unlabeled_dataset",
        "target_column": target,
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "uses_manual_weights": False,
        "uses_diagnostic_method_weights": False,
    }
    joblib.dump(artifact, STUDENT_MODEL_PATH)
    logger.info("Built runtime fine weak-label student model | rows=%d | classes=%s", len(clean), classes)
    return artifact

def _apply_student(df: pd.DataFrame, artifact: dict):
    out = df.copy()
    feature_columns = list(artifact.get("feature_cols", artifact.get("features", cfg.COMMON_BENCHMARK_GASES)))
    X = _prepare_student_matrix(out, feature_columns)
    model = artifact["model"]
    try:
        proba = np.asarray(model.predict_proba(X), dtype=float)
        estimator = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
        model_classes = getattr(estimator, "classes_", None)
        labels = [normalize_fault(v) for v in (model_classes if model_classes is not None else artifact["labels"])]
        if proba.ndim != 2 or proba.shape[1] != len(labels): raise ValueError("Student probability matrix does not match model classes.")
        best = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        pred = [labels[int(i)] for i in best]
    except Exception:
        logger.exception("Student predict_proba failed; falling back to predict().")
        pred = [normalize_fault(v) for v in model.predict(X)]
        confidence = np.full(len(out), np.nan, dtype=float)
    out["student_fault_label"] = pred
    out["student_fault_group"] = [unify_fault(v) for v in pred]
    out["student_fault_confidence"] = confidence
    out["student_model_name"] = artifact.get("model_name", "UNKNOWN")
    out["student_training_type"] = artifact.get("training_type", "UNKNOWN")
    out["student_feature_set"] = ",".join(feature_columns)
    return out

def _combine_consensus_and_student(df):
    out = df.copy()
    weak_fine = _normalize_series(out["weak_fine_fault"]) if "weak_fine_fault" in out.columns else pd.Series("ABSTAIN", index=out.index)
    traditional = _normalize_series(out["consensus_fault"]) if "consensus_fault" in out.columns else pd.Series("ABSTAIN", index=out.index)
    student = _normalize_series(out["student_fault_label"]) if "student_fault_label" in out.columns else pd.Series("ABSTAIN", index=out.index)
    weak_active = weak_fine != "ABSTAIN"
    traditional_active = traditional != "ABSTAIN"
    student_active = student != "ABSTAIN"
    final_fine = weak_fine.copy()
    source = pd.Series("ABSTAIN", index=out.index, dtype=object)
    use_traditional = ~weak_active & traditional_active
    use_student = ~weak_active & ~traditional_active & student_active
    final_fine.loc[use_traditional] = traditional.loc[use_traditional]
    final_fine.loc[use_student] = student.loc[use_student]
    source.loc[weak_active] = "weak_fine_label_model"
    source.loc[use_traditional] = "traditional_consensus_after_weak_abstain"
    source.loc[use_student] = "student_last_resort"
    weak_group = weak_fine.map(unify_fault)
    traditional_group = traditional.map(unify_fault)
    student_group = student.map(unify_fault)
    physical_conflict = weak_active & traditional_active & (weak_group != traditional_group)
    same_group_different_fine = weak_active & traditional_active & ~physical_conflict & (weak_fine != traditional)
    out["final_fault"] = final_fine
    out["final_fault_group"] = final_fine.map(unify_fault).fillna("ABSTAIN").astype(str)
    out["final_fault_source"] = source
    out["final_fault_conflict"] = physical_conflict
    out["final_fault_same_coarse_different_fine"] = same_group_different_fine
    out["final_fault_conflict_level"] = np.select([physical_conflict, same_group_different_fine, weak_active, traditional_active, student_active], ["PHYSICAL_GROUP_CONFLICT", "SAME_GROUP_FINE_DISAGREEMENT", "WEAK_FINE_MODEL", "TRADITIONAL_CONSENSUS", "STUDENT_FALLBACK"], default="ABSTAIN")
    out["student_used_as_fallback"] = use_student
    out["final_fault_is_weak_supervision"] = weak_active
    out["student_vs_traditional_coarse_agreement"] = traditional_active & student_active & traditional_group.eq(student_group)
    out["fault_criticality_class"] = final_fine.map(classify_fault_criticality)
    out["fault_criticality_source"] = fault_criticality_source()
    return out

def _load_or_fit_anomaly(df):
    from anomaly import UnsupervisedEnsemble
    gases = list(cfg.ALL_DGA_GASES)
    X = df.reindex(columns=gases, fill_value=np.nan).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    if ANOMALY_MODEL_PATH.exists():
        try:
            model = joblib.load(ANOMALY_MODEL_PATH)
            return model, model.predict(X)
        except Exception: logger.exception("Stored anomaly model failed; refitting.")
    model = UnsupervisedEnsemble(random_state=cfg.RANDOM_STATE)
    model.fit(X, feature_names=gases)
    scores = model.predict(X)
    joblib.dump(model, ANOMALY_MODEL_PATH)
    return model, scores

def create_payload(df, ranking_df, comparison_df=None):
    rows = []
    ordered = df.sort_values(["transformer_id", "sample_day"], ascending=[True, False], kind="mergesort")
    export_fields = ["transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg", "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2", "ieee_dga_status", "ieee_dga_status_label", "ieee_dga_status_reason", "severity_label_text", "severity_source", "severity_score_type", "severity_is_failure_probability", "severity_composite_weighted", "severity_uses_manual_weights", "severity_anomaly_used", "severity_nei_used", "severity_is_not_a_health_score", "ieee_max_standardized_exceedance", "ieee_max_status3_standardized_exceedance", "ieee_table1_concentration_ratio_all", "ieee_table2_concentration_ratio_all", "ieee_table3_delta_ratio_all", "ieee_table4_rate_ratio_all", "ieee_continuous_evidence_ratio", "ieee_continuous_evidence_basis", "ieee_standard_trigger_count", "ieee_table1_max_exceedance_ratio", "ieee_table2_max_exceedance_ratio", "ieee_table3_max_exceedance_ratio", "ieee_table4_max_exceedance_ratio", "ieee_maintenance_priority_hint", "ieee_maintenance_priority_hint_reason", "consensus_fault", "consensus_fault_traditional", "consensus_fault_group", "diagnostic_agreement_ratio", "diagnostic_coverage", "diagnostic_confidence", "weak_fine_fault", "weak_fine_fault_group", "weak_fine_posterior_max", "weak_fine_entropy", "weak_fine_lf_active_count", "weak_fine_lf_coverage", "weak_coarse_fault", "weak_coarse_fault_group", "weak_coarse_posterior_max", "weak_coarse_entropy", "final_fault", "final_fault_group", "final_fault_source", "final_fault_conflict", "final_fault_same_coarse_different_fine", "fault_criticality_class", "fault_criticality_source", "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault", "duval_triangle_fault", "duval_pentagon_p1_fault", "duval_pentagon_p2_fault", "fault_p1", "fault_p2", "student_fault_label", "student_fault_group", "student_fault_confidence", "student_model_name", "student_training_type", "student_feature_set", "student_used_as_fallback", "anomaly_percentile", "anomaly_is_severity_input", "ieee_confirmation_required", "ieee_delta_available", "ieee_rate_available", "ieee_rate_span_months", "ieee_table1_exceeding_gases", "ieee_table2_exceeding_gases", "ieee_table3_exceeding_gases", "ieee_table4_exceeding_gases", "ieee_delta", "ieee_gas_rate_ppm_per_year", "iec_60599_ratios", "ieee_o2_n2_ratio", "ieee_age_bucket"]
    ranking_lookup = {str(r.get("transformer_id")): r for _, r in ranking_df.iterrows()}
    predictions = []
    for idx, row in ordered.iterrows():
        status = _safe_int(row.get("ieee_dga_status", 0))
        severity_label = _ui_severity_label(row.get("severity_label_text", "INSUFFICIENT_DATA"))
        rank_info = ranking_lookup.get(str(row.get("transformer_id")), {})
        current_priority = rank_info.get("maintenance_priority", "DATA_REVIEW")
        current_critical = bool(rank_info.get("critical_front_flag", False))
        current_priority_reason = rank_info.get("maintenance_priority_reason", "")
        critical_ratio = _safe_float(row.get("ieee_max_status3_standardized_exceedance", np.nan))
        fault_type = row.get("final_fault", row.get("consensus_fault", "ABSTAIN"))
        fault_group = row.get("final_fault_group", row.get("consensus_fault_group", "ABSTAIN"))
        predictions.append({"row_index": int(idx), "transformer_id": row.get("transformer_id"), "pred_ensemble": status, "ieee_status": status, "ieee_status_label": severity_label, "status": _ui_status(status), "severity": _ui_status(status), "severity_label": severity_label, "maintenance_priority": current_priority, "critical_front": current_critical, "maintenance_priority_reason": current_priority_reason, "critical_evidence_ratio": critical_ratio, "continuous_evidence_ratio": _safe_float(row.get("ieee_continuous_evidence_ratio", np.nan)), "continuous_evidence_basis": row.get("ieee_continuous_evidence_basis", "NO_CONTINUOUS_EVIDENCE"), "fault_type": fault_type, "fault_group": fault_group, "fault_criticality_class": classify_fault_criticality(fault_type), "fault_source": row.get("final_fault_source", "ABSTAIN"), "fault_confidence": _safe_float(row.get("weak_fine_posterior_max", np.nan)), "fault_entropy": _safe_float(row.get("weak_fine_entropy", np.nan)), "fault_evidence_level": "FINE_WEAK_LABEL_MODEL" if row.get("weak_fine_fault", "ABSTAIN") != "ABSTAIN" else "FALLBACK", "reason": row.get("ieee_dga_status_reason", ""), "confirmation_required": bool(row.get("ieee_confirmation_required", False)), "anomaly_percentile": _safe_float(row.get("anomaly_percentile", np.nan)), "top_features": []})
        rows.append({key: row[key] for key in export_fields if key in row.index})
    transformer_summary = []
    for _, rank_row in ranking_df.iterrows():
        status = _safe_int(_first(rank_row, ["transformer_overall_severity_level"], 0))
        fault_type = _first(rank_row, ["current_fault", "history_dominant_fault"], "ABSTAIN")
        fault_group = _first(rank_row, ["current_fault_group"], "ABSTAIN")
        priority = rank_row.get("maintenance_priority", "DATA_REVIEW")
        transformer_summary.append({"rank": _safe_int(rank_row.get("rank", 0)), "maintenance_rank": _safe_int(rank_row.get("maintenance_priority_rank", rank_row.get("rank", 0))), "rank_tie": bool(rank_row.get("rank_tie", False)), "rank_group_size": _safe_int(rank_row.get("rank_group_size", 1), 1), "transformer_id": rank_row.get("transformer_id"), "latest_sample_day": str(rank_row.get("sample_day", "")), "maintenance_priority": priority, "maintenance_priority_ordinal": _safe_int(rank_row.get("maintenance_priority_ordinal", 0)), "maintenance_priority_reason": rank_row.get("maintenance_priority_reason", ""), "critical_front": bool(rank_row.get("critical_front_flag", False)), "critical_rule": rank_row.get("critical_rule", cfg.CRITICAL_RULE), "critical_reference": rank_row.get("critical_reference", cfg.CRITICAL_REFERENCE), "critical_evidence_table": rank_row.get("critical_evidence_table"), "critical_evidence_gas": rank_row.get("critical_evidence_gas"), "critical_evidence_ratio": _safe_float(rank_row.get("critical_evidence_ratio", np.nan)), "critical_evidence_scope": "LATEST_SAMPLE", "historical_max_standardized_exceedance": _safe_float(rank_row.get("historical_max_standardized_exceedance", np.nan)), "ieee_status": status, "ieee_status_label": _ui_severity_label(rank_row.get("transformer_overall_severity_label", status)), "status": _ui_status(status), "severity": _ui_status(status), "severity_label": _ui_severity_label(rank_row.get("transformer_overall_severity_label", status)), "fault_type": fault_type, "fault_group": fault_group, "fault_criticality_class": rank_row.get("fault_criticality_class", classify_fault_criticality(fault_type)), "fault_criticality_source": rank_row.get("fault_criticality_source", fault_criticality_source()), "recommended_action": rank_row.get("recommended_action", "REVIEW_DATA"), "reason": rank_row.get("maintenance_priority_reason", ""), "current_standardized_exceedance": _safe_float(rank_row.get("current_standardized_exceedance", np.nan)), "current_status3_standardized_exceedance": _safe_float(rank_row.get("current_status3_standardized_exceedance", np.nan)), "current_delta_exceedance": _safe_int(rank_row.get("current_delta_exceedance", 0)), "current_standard_trigger_count": _safe_int(rank_row.get("current_standard_trigger_count", 0)), "historical_max_standardized_exceedance": _safe_float(rank_row.get("historical_max_standardized_exceedance", np.nan)), "history_max_status_before_current": _safe_int(rank_row.get("history_max_status_before_current", 0)), "history_record_count": _safe_int(rank_row.get("history_record_count", 0)), "history_worsening_transition_ratio": _safe_float(rank_row.get("history_worsening_transition_ratio", np.nan)), "history_recurrent_fault_fraction": _safe_float(rank_row.get("history_current_fault_recurrence_fraction", np.nan)), "pareto_dominance_count": _safe_int(rank_row.get("pareto_dominance_count", 0)), "pareto_front": bool(rank_row.get("pareto_front", False)), "severity_evidence_vector": rank_row.get("severity_evidence_vector", ""), "maintenance_priority_rank_percentile": _safe_float(rank_row.get("maintenance_priority_rank_percentile", np.nan)), "priority_score": _safe_float(rank_row.get("transformer_overall_severity_score", rank_row.get("priority_score", np.nan))), "priority_score_type": rank_row.get("priority_score_type", "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER"), "ranking_policy": rank_row.get("ranking_policy", ""), "ranking_is_weighted": bool(rank_row.get("ranking_is_weighted", False)), "ranking_is_health_score": bool(rank_row.get("ranking_is_health_score", False)), "loc": _first(rank_row, ["loc"], "") or "", "name": _first(rank_row, ["name"], "") or "", "features": {}})
    timeseries = {}
    critical_lookup = {str(r.get("transformer_id")): r for _, r in ranking_df.iterrows()} if ranking_df is not None and not ranking_df.empty else {}
    for transformer_id, group in ordered.groupby("transformer_id", sort=False):
        series = []
        group_sorted = group.sort_values("sample_day")
        latest_index = group_sorted.index[-1] if len(group_sorted) else None
        rank_info = critical_lookup.get(str(transformer_id), {})
        transformer_is_critical = bool(rank_info.get("critical_front_flag", False))
        for row_index, row in group_sorted.iterrows():
            status = _safe_int(row.get("ieee_dga_status", 0))
            critical_front = bool(transformer_is_critical and row_index == latest_index and status == 3)
            critical_ratio = _safe_float(row.get("ieee_max_status3_standardized_exceedance", np.nan))
            continuous_ratio = _safe_float(row.get("ieee_continuous_evidence_ratio", np.nan))
            concentration_ratio = _safe_float(row.get("ieee_table2_concentration_ratio_all", np.nan))
            delta_ratio = _safe_float(row.get("ieee_table3_delta_ratio_all", np.nan))
            rate_ratio = _safe_float(row.get("ieee_table4_rate_ratio_all", np.nan))
            fault = row.get("final_fault", row.get("consensus_fault", "ABSTAIN"))
            series.append({"Sample Day": str(row["sample_day"]), "H2": _safe_float(row.get("h2", np.nan), 0.0), "C2H2": _safe_float(row.get("c2h2", np.nan), 0.0), "TDCG": _safe_float(row.get("tdcg", row.get("ieee_tdcg_ppm", np.nan)), 0.0), "pred_ensemble": status, "ieee_status": status, "ieee_status_label": row.get("ieee_dga_status_label", "INSUFFICIENT_DATA"), "status": _ui_status(status), "fault_type": fault, "fault_group": row.get("final_fault_group", row.get("consensus_fault_group", "ABSTAIN")), "fault_criticality_class": classify_fault_criticality(fault), "severity": row.get("severity_label_text", "INSUFFICIENT_DATA"), "critical_front": critical_front, "critical_evidence_ratio": critical_ratio, "continuous_evidence_ratio": continuous_ratio, "continuous_evidence_basis": row.get("ieee_continuous_evidence_basis", "NO_CONTINUOUS_EVIDENCE"), "continuous_evidence_is_score": False, "continuous_evidence_reference": "IEEE threshold ratio; per-sample diagnostic evidence, not a weighted severity score", "table2_concentration_ratio": concentration_ratio, "table3_delta_ratio": delta_ratio, "table4_rate_ratio": rate_ratio, "confirmation_required": bool(row.get("ieee_confirmation_required", False))})
        timeseries[str(transformer_id)] = series
    status_series = pd.to_numeric(df.get("ieee_dga_status", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    priority_counts = ranking_df["transformer_overall_severity_label"].value_counts().reindex(["STATUS_3", "STATUS_2", "STATUS_1", "INSUFFICIENT_DATA"]).fillna(0).astype(int).to_dict() if "transformer_overall_severity_label" in ranking_df.columns else {}
    fault_context_counts = ranking_df["fault_criticality_class"].value_counts().to_dict() if "fault_criticality_class" in ranking_df.columns else {}
    top_queue = []
    if ranking_df is not None and not ranking_df.empty:
        for _, r in ranking_df.head(20).iterrows():
            top_queue.append({
                "rank": _safe_int(r.get("rank", 0)),
                "transformer_id": r.get("transformer_id"),
                "maintenance_priority": r.get("maintenance_priority", "DATA_REVIEW"),
                "severity_rank_within_class": _safe_int(r.get("severity_rank_within_class", 0)),
                "severity_class_size": _safe_int(r.get("severity_class_size", 0)),
                "ieee_status": _safe_int(r.get("transformer_overall_severity_level", 0)),
                "current_standardized_exceedance": _safe_float(r.get("current_standardized_exceedance", np.nan)),
                "current_status3_standardized_exceedance": _safe_float(r.get("current_status3_standardized_exceedance", np.nan)),
                "table2_exceed_count": _safe_int(r.get("table2_exceed_count", 0)),
                "table4_exceed_count": _safe_int(r.get("table4_exceed_count", 0)),
                "table3_exceed_count": _safe_int(r.get("table3_exceed_count", 0)),
                "fault_type": r.get("current_fault", "ABSTAIN"),
                "fault_group": r.get("current_fault_group", "ABSTAIN"),
                "recommended_action": r.get("recommended_action", ""),
            })
    dataset_summary = {
        "total_transformers": int(df["transformer_id"].nunique()),
        "total_rows": int(len(df)),
        "severity_status_1": int((status_series == 1).sum()),
        "severity_status_2": int((status_series == 2).sum()),
        "severity_status_3": int((status_series == 3).sum()),
        "severity_insufficient_data": int((status_series == 0).sum()),
        "maintenance_priority_counts": priority_counts,
        "high_risk_transformer_count": int(priority_counts.get("STATUS_3", 0)),
        "watch_transformer_count": int(priority_counts.get("STATUS_2", 0)),
        "normal_transformer_count": int(priority_counts.get("STATUS_1", 0)),
        "first_priority_transformer_id": (top_queue[0]["transformer_id"] if top_queue else None),
        "first_priority_rank": (top_queue[0]["rank"] if top_queue else None),
        "maintenance_queue_top20": top_queue,
        "critical_rule": "NOT_USED",
        "critical_reference": "No additional Status-4 severity class is used.",
        "fault_criticality_context_counts": fault_context_counts,
        "fault_criticality_source": fault_criticality_source(),
        "traditional_abstain_rows": int((_normalize_series(df.get("consensus_fault", pd.Series("ABSTAIN", index=df.index))) == "ABSTAIN").sum()),
        "student_fallback_rows": int(df.get("student_used_as_fallback", pd.Series(False, index=df.index)).sum()),
        "student_traditional_physical_conflicts": int(df.get("final_fault_conflict", pd.Series(False, index=df.index)).sum()),
    }
    return {"predictions": predictions, "rows": rows, "preview_rows": rows[:20], "transformer_summary": transformer_summary, "transformer_timeseries": timeseries, "dataset_summary": dataset_summary, "student_traditional_comparison": [] if comparison_df is None or comparison_df.empty else comparison_df.to_dict(orient="records"), "chat_context_payload": {"transformer_summary": transformer_summary, "dataset_summary": dataset_summary}}

def _build_student_comparison(df):
    required = {"transformer_id", "student_fault_label", "consensus_fault"}
    if not required.issubset(df.columns): return pd.DataFrame()
    work = df.copy()
    work["_consensus_fine"] = _normalize_series(work["consensus_fault"])
    work["_student_fine"] = _normalize_series(work["student_fault_label"])
    work["_consensus_group"] = work["_consensus_fine"].map(unify_fault).fillna("ABSTAIN").astype(str)
    work["_student_group"] = work["_student_fine"].map(unify_fault).fillna("ABSTAIN").astype(str)
    rows = []
    for transformer_id, group in work.groupby("transformer_id", sort=False):
        joint = (group["_consensus_group"] != "ABSTAIN") & (group["_student_group"] != "ABSTAIN")
        n_joint = int(joint.sum())
        coarse_agreement = float((group.loc[joint, "_consensus_group"] == group.loc[joint, "_student_group"]).mean()) if n_joint else np.nan
        fine_agreement = float((group.loc[joint, "_consensus_fine"] == group.loc[joint, "_student_fine"]).mean()) if n_joint else np.nan
        rows.append({"transformer_id": transformer_id, "n_samples": int(len(group)), "n_joint_active": n_joint, "coarse_agreement_rate": coarse_agreement, "fine_agreement_rate": fine_agreement, "traditional_abstain_count": int((group["_consensus_fine"] == "ABSTAIN").sum()), "student_abstain_count": int((group["_student_fine"] == "ABSTAIN").sum()), "student_used_as_fallback_count": int(((group["_consensus_fine"] == "ABSTAIN") & (group["_student_fine"] != "ABSTAIN")).sum()), "physical_conflict_count": int((joint & (group["_consensus_group"] != group["_student_group"])).sum())})
    return pd.DataFrame(rows)

def _write_inference_metadata(df, artifact, weak_metadata, elapsed_seconds):
    metadata = {"pipeline_type": "operational_unlabeled_inference", "n_rows": int(len(df)), "n_transformers": int(df["transformer_id"].nunique()), "diagnostic_methods": list(cfg.DIAGNOSTIC_METHODS), "diagnostic_consensus": "unweighted_majority_with_abstain", "diagnostic_method_weights": None, "severity_standard": cfg.STANDARD, "severity_is_weighted": False, "severity_uses_manual_weights": False, "severity_uses_nei": False, "severity_uses_anomaly": False, "severity_type": "IEEE_ORDINAL_STATUS", "ranking_policy": list(cfg.RANKING_POLICY), "ranking_is_weighted": False, "ranking_is_health_score": False, "ranking_uses_manual_weights": False, "ranking_uses_fault_criticality_as_severity": False, "ranking_score_type": "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER", "maintenance_priority_extension": "NONE; IEEE_STATUS_1_2_3_ONLY", "critical_rule": "NOT_USED", "critical_reference": cfg.CRITICAL_REFERENCE, "fault_criticality_source": fault_criticality_source(), "student_model_name": artifact.get("model_name", "UNKNOWN"), "student_training_type": artifact.get("training_type", "UNKNOWN"), "student_features": artifact.get("feature_cols", cfg.COMMON_BENCHMARK_GASES), "weak_supervision_backend": weak_metadata.get("backend") if isinstance(weak_metadata, dict) else None, "weak_supervision_granularity": weak_metadata.get("granularity") if isinstance(weak_metadata, dict) else None, "processing_seconds": float(elapsed_seconds)}
    INFERENCE_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

def process_dataframe(uploaded_df: pd.DataFrame):
    start = time.time()
    total_steps = 14
    tmp_dir = Path(tempfile.mkdtemp())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("\n\n" + "#" * 110 + "\n" + "# FULL DGA PIPELINE START\n" + "# uploaded rows=%d\n" + "#" * 110, len(uploaded_df))
        input_path = tmp_dir / "uploaded_input.xlsx"
        uploaded_df.to_excel(input_path, index=False, engine="openpyxl")
        _log_stage(1, total_steps, "Cleaning dataset")
        df_clean, _ = clean_dataset(input_file=input_path, output_dir=PROCESSED_DIR)
        if df_clean.empty: raise ValueError("Cleaning produced an empty dataset.")
        logger.info("Cleaning complete | rows=%d | transformers=%d | columns=%d", len(df_clean), df_clean["transformer_id"].nunique() if "transformer_id" in df_clean.columns else 0, len(df_clean.columns))
        _log_stage(2, total_steps, "Building DGA features")
        df_features = build_training_features_from_clean(df_clean)
        if df_features.empty: raise ValueError("Feature engineering produced an empty dataset.")
        logger.info("Feature engineering complete | rows=%d | columns=%d", len(df_features), len(df_features.columns))
        _log_stage(3, total_steps, "Traditional diagnostics + unweighted consensus")
        df_labeled = apply_consensus(df_features)
        logger.info("Traditional diagnostics complete")
        if "consensus_fault" in df_labeled.columns:
            logger.info("Traditional consensus distribution | %s", df_labeled["consensus_fault"].value_counts(dropna=False).to_dict())
        _log_stage(4, total_steps, "Weak supervision - coarse")
        df_weak_coarse, coarse_model, coarse_groups, coarse_metadata, coarse_L, coarse_probabilities, coarse_pairwise = weak_supervision_pipeline(df_labeled, use_snorkel=SNORKEL_AVAILABLE, random_state=cfg.RANDOM_STATE, granularity="coarse")
        _log_stage(5, total_steps, "Weak supervision - fine")
        df_weak_fine, fine_model, fine_groups, fine_metadata, fine_L, fine_probabilities, fine_pairwise = weak_supervision_pipeline(df_labeled, use_snorkel=SNORKEL_AVAILABLE, random_state=cfg.RANDOM_STATE, granularity="fine")
        weak_report_dir = REPORT_DIR / "benchmark"
        weak_report_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(coarse_pairwise, weak_report_dir / "weak_lf_pairwise_agreement_coarse.csv")
        _save_csv(fine_pairwise, weak_report_dir / "weak_lf_pairwise_agreement_fine.csv")
        save_weak_supervision_artifacts(df_weak_coarse, coarse_model, coarse_groups, coarse_metadata, granularity="coarse")
        save_weak_supervision_artifacts(df_weak_fine, fine_model, fine_groups, fine_metadata, granularity="fine")
        logger.info("Weak supervision complete | coarse_backend=%s | fine_backend=%s", coarse_metadata.get("backend"), fine_metadata.get("backend"))
        merge_keys = ["transformer_id", "sample_day"]
        coarse_columns = [c for c in df_weak_coarse.columns if c.startswith("weak_")]
        fine_columns = [c for c in df_weak_fine.columns if c.startswith("weak_")]
        coarse_subset = df_weak_coarse[merge_keys + coarse_columns].drop_duplicates(subset=merge_keys, keep="last")
        fine_subset = df_weak_fine[merge_keys + fine_columns].drop_duplicates(subset=merge_keys, keep="last")
        df_labeled = df_labeled.merge(coarse_subset, on=merge_keys, how="left", suffixes=("", "_weak"))
        df_labeled = df_labeled.merge(fine_subset, on=merge_keys, how="left", suffixes=("", "_fine"))
        weak_metadata = {"backend": {"coarse": coarse_metadata.get("backend"), "fine": fine_metadata.get("backend")}, "granularity": "coarse+fine", "coarse": coarse_metadata, "fine": fine_metadata, "uses_manual_lf_weights": False}
        del coarse_model, fine_model, coarse_groups, fine_groups, coarse_L, fine_L, coarse_probabilities, fine_probabilities
        _log_stage(6, total_steps, "Training weak-label student models")
        weak_students = _run_weak_students(df_weak_coarse, df_weak_fine, cfg.RANDOM_STATE)
        df_labeled = _apply_research_students(df_labeled, weak_students)
        _log_stage(7, total_steps, "Applying production student model")
        artifact = _load_student_model()
        if artifact is None:
            logger.info("No compatible stored production student model.")
            artifact = _train_student_fallback(df_weak_fine)
        df_labeled = _apply_student(df_labeled, artifact)
        _log_stage(8, total_steps, "Combining traditional + weak + student evidence")
        df_labeled = _combine_consensus_and_student(df_labeled)
        comparison_df = _build_student_comparison(df_labeled)
        if not comparison_df.empty: _save_csv(comparison_df, STUDENT_COMPARISON_PATH)
        _log_stage(9, total_steps, "Independent anomaly detection")
        _, anomaly_scores = _load_or_fit_anomaly(df_labeled)
        df_labeled["anomaly_percentile"] = np.asarray(anomaly_scores, dtype=float)
        df_labeled["anomaly_is_severity_input"] = False
        df_labeled["anomaly_interpretation"] = "Relative anomaly position only; not IEEE severity or maintenance rank input."
        logger.info("Anomaly detection complete | min=%.2f | median=%.2f | max=%.2f", float(np.nanmin(anomaly_scores)), float(np.nanmedian(anomaly_scores)), float(np.nanmax(anomaly_scores)))
        _log_stage(10, total_steps, "IEEE C57.104-2019 severity")
        df_labeled = apply_severity(df_labeled, nei_reference=None)
        df_labeled["severity_inference_stage"] = "STANDARD_RULE_ENGINE"
        df_labeled["severity_manual_weights"] = False
        df_labeled["severity_weighted_sum_used"] = False
        df_labeled["severity_anomaly_used"] = False
        df_labeled["severity_nei_used"] = False
        logger.info("IEEE severity distribution | %s", df_labeled["ieee_dga_status"].value_counts().sort_index().to_dict())
        _log_stage(11, total_steps, "Transformer maintenance ranking")
        ranking_df = build_transformer_ranking(df_labeled)
        log_ranking_diagnostics(ranking_df, top_n=20)
        if not ranking_df.empty:
            _save_parquet(ranking_df, RANKING_PATH)
            _save_csv(ranking_df, REPORT_DIR / "transformer_ranking.csv")
        _log_stage(12, total_steps, "Saving operational artifacts")
        _save_parquet(df_labeled, UNLABELED_PATH)
        _save_parquet(df_labeled, PROCESSED_OUTPUT_PATH)
        _log_stage(13, total_steps, "External labeled benchmark + weak-transfer evaluation")
        research_results = _run_external_research_benchmark(df_labeled, weak_students, cfg.RANDOM_STATE)
        research_status = research_results.get("status", "completed")
        _log_stage(14, total_steps, "Building Excel report + saving database + final response")
        elapsed = time.time() - start
        excel_path = _build_research_excel()
        excel_status = "completed" if excel_path is not None else "failed"
        _write_inference_metadata(df_labeled, artifact, weak_metadata, elapsed)
        payload = create_payload(df_labeled, ranking_df, comparison_df)
        payload["pipeline"] = {"status": "completed" if (research_status == "completed" and excel_status == "completed") else "completed_with_warnings", "rows": int(len(df_labeled)), "transformers": int(df_labeled["transformer_id"].nunique()), "elapsed_seconds": float(elapsed), "weak_backend": weak_metadata["backend"], "student_models": {"coarse": len(weak_students["coarse"]), "fine": len(weak_students["fine"])}, "research_benchmark": {"status": research_status, "traditional_rows": int(len(research_results["traditional_individual"])), "combination_rows": int(len(research_results["traditional_combinations"])), "supervised_rows": int(len(research_results["supervised"])), "weak_label_transfer_rows": int(len(research_results.get("weak_label_transfer", pd.DataFrame()))), "weak_transfer_rows": int(len(research_results["weak_transfer"])), "hybrid_rows": int(len(research_results.get("hybrid", pd.DataFrame())))}, "excel": {"status": excel_status}, "files": {"processed_parquet": str(PROCESSED_OUTPUT_PATH), "ranking_parquet": str(RANKING_PATH), "ranking_csv": str(REPORT_DIR / "transformer_ranking.csv"), "excel_report": str(excel_path) if excel_path else None, "student_comparison": str(STUDENT_COMPARISON_PATH), "weak_coarse_parquet": str(PROCESSED_DIR / "dga_weak_labels_coarse.parquet"), "weak_fine_parquet": str(PROCESSED_DIR / "dga_weak_labels_fine.parquet")}}
        from data_store import save_payload_to_db
        save_payload_to_db(payload)
        logger.info("=" * 110)
        logger.info("FULL DGA PIPELINE COMPLETED")
        logger.info("Rows              : %d", len(df_labeled))
        logger.info("Transformers      : %d", df_labeled["transformer_id"].nunique())
        logger.info("Elapsed seconds    : %.2f", elapsed)
        logger.info("Weak backend       : %s", weak_metadata["backend"])
        logger.info("Student models     : coarse=%d | fine=%d", len(weak_students["coarse"]), len(weak_students["fine"]))
        logger.info("Severity           : %s", df_labeled["ieee_dga_status"].value_counts().sort_index().to_dict())
        if not ranking_df.empty:
            logger.info("IEEE status counts : %s", ranking_df["transformer_overall_severity_label"].value_counts().to_dict())
            logger.info("First fleet priority: rank=%d transformer=%s", int(ranking_df.iloc[0]["rank"]), str(ranking_df.iloc[0]["transformer_id"]))
        logger.info("Excel report       : %s", excel_path)
        logger.info("Benchmark outputs  : %s", REPORT_DIR / "benchmark")
        logger.info("Processed data     : %s", PROCESSED_OUTPUT_PATH)
        logger.info("=" * 110)
        return payload
    except Exception:
        logger.exception("FATAL ERROR IN FULL DGA PIPELINE")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

__all__ = ["process_dataframe", "create_payload"]