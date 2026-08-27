# weak_supervision.py
from __future__ import annotations
import importlib.util, json, logging
from pathlib import Path
import joblib, numpy as np, pandas as pd
from config import DATASET_DIR, MODEL_DIR, config as cfg
from consensus import ABSTAIN as ABSTAIN_TEXT, normalize_fault, pairwise_label_agreement, unify_fault

logger = logging.getLogger(__name__)
ABSTAIN = cfg.WEAK_ABSTAIN_LABEL
SNORKEL_AVAILABLE = False
LabelModel = None
try:
    if importlib.util.find_spec("snorkel") is not None:
        from snorkel.labeling.model import LabelModel as _LabelModel
        LabelModel = _LabelModel
        SNORKEL_AVAILABLE = True
except Exception as exc:
    logger.warning("Snorkel unavailable: %s", exc)

DEFAULT_WEAK_METHODS = dict(cfg.DIAGNOSTIC_METHOD_TO_COLUMN)

def _validate_groups(groups):
    groups = [str(value).strip().upper() for value in groups]
    if len(groups) < 2:
        logger.error("_validate_groups: need at least two classes")
        raise ValueError("Need at least two weak-supervision classes.")
    if len(set(groups)) != len(groups):
        logger.error("_validate_groups: duplicate groups found")
        raise ValueError("Weak-supervision classes must be unique.")
    return groups

def _normalize_coarse_vote(raw_label):
    fine = normalize_fault(raw_label)
    if fine == ABSTAIN_TEXT:
        return ABSTAIN_TEXT
    coarse = unify_fault(fine)
    if coarse in cfg.WEAK_COARSE_GROUPS:
        return coarse
    return ABSTAIN_TEXT

def _normalize_fine_vote(raw_label):
    fine = normalize_fault(raw_label)
    if fine in cfg.BENCHMARK_FINE_CLASSES:
        return fine
    return ABSTAIN_TEXT

def build_label_matrix(df, label_columns=None, groups=None, granularity="coarse"):
    methods = dict(label_columns or DEFAULT_WEAK_METHODS)
    granularity = str(granularity).strip().lower()
    if granularity == "coarse":
        default_groups = cfg.WEAK_COARSE_GROUPS
        normalizer = _normalize_coarse_vote
    elif granularity == "fine":
        default_groups = cfg.WEAK_FINE_GROUPS
        normalizer = _normalize_fine_vote
    else:
        logger.error("build_label_matrix: invalid granularity %s", granularity)
        raise ValueError("granularity must be 'coarse' or 'fine'.")
    groups_list = _validate_groups(groups or default_groups)
    group_to_int = {group: idx for idx, group in enumerate(groups_list)}
    L = np.full((len(df), len(methods)), ABSTAIN, dtype=np.int64)
    for j, column in enumerate(methods.values()):
        if column not in df.columns:
            continue
        values = df[column].map(normalizer)
        L[:, j] = values.map(lambda value: group_to_int.get(value, ABSTAIN)).to_numpy(dtype=np.int64)
    logger.debug("build_label_matrix: L shape=%s non-abstain rate=%.3f", L.shape, (L != ABSTAIN).mean())
    return L, list(methods.keys()), groups_list

def fit_label_model_backend(
    L,
    cardinality,
    backend="snorkel",
    random_state=42,
    snorkel_epochs=500,
):
    """Fit the Snorkel LabelModel used by the weak-supervision pipeline.

    The project intentionally has one weak-supervision backend: Snorkel.
    A silent EM fallback is not used because backend changes would alter the
    research method while making the run metadata look identical.
    """
    if not SNORKEL_AVAILABLE or LabelModel is None:
        raise RuntimeError(
            "Snorkel is required for weak supervision but is not available in "
            "the active Python environment. Install a compatible snorkel "
            "package in .venv before running the unlabeled experiment."
        )

    requested = str(backend or "snorkel").strip().lower()
    if requested not in {"snorkel", "snorkel_only"}:
        raise ValueError(
            "Only the Snorkel weak-supervision backend is supported. "
            f"Received backend={backend!r}."
        )

    L = np.asarray(L, dtype=np.int64)
    if L.ndim != 2 or len(L) == 0 or L.shape[1] == 0:
        raise ValueError("Label matrix L must be a non-empty 2D array.")
    if int(cardinality) < 2:
        raise ValueError("Snorkel LabelModel cardinality must be at least 2.")

    logger.info(
        "Weak supervision backend START: Snorkel LabelModel "
        "| epochs=%d | rows=%d | LFs=%d | seed=%d",
        int(snorkel_epochs),
        len(L),
        L.shape[1],
        int(random_state),
    )

    # IMPORTANT: current Snorkel exposes LabelModel from snorkel.labeling.model.
    # Keep the import above; newer Snorkel versions expose LabelModel in the
    # `snorkel.labeling.model` module rather than the package root.
    label_model = LabelModel(
        cardinality=int(cardinality),
        verbose=False,
    )
    # Snorkel requires log_freq > 0.  Zero reaches the internal logger as
    # `unit_count % log_freq` and raises ZeroDivisionError in current releases.
    # Logging frequency is a runtime/logging parameter; it does not alter the
    # LabelModel objective or learned parameters.
    epochs = int(snorkel_epochs)
    if epochs < 1:
        raise ValueError("snorkel_epochs must be >= 1.")
    log_freq = max(1, min(100, epochs))

    label_model.fit(
        L_train=L,
        n_epochs=epochs,
        seed=int(random_state),
        log_freq=log_freq,
    )
    probabilities = label_model.predict_proba(L=L)

    return label_model, probabilities, "snorkel"


def fit_label_model(L, cardinality, use_snorkel=True, random_state=42):
    """Compatibility wrapper used by the existing training pipeline.

    There is deliberately no EM fallback. The unlabeled research experiment
    must record exactly which weak-supervision method produced its labels.
    """
    if not use_snorkel:
        raise RuntimeError(
            "This research pipeline requires Snorkel LabelModel. "
            "Run with --use-snorkel. An EM fallback is intentionally disabled."
        )
    return fit_label_model_backend(
        L,
        cardinality,
        backend="snorkel",
        random_state=random_state,
    )


def predict_from_label_model(model, df, methods, groups, granularity):
    method_mapping = {method: cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method] for method in methods}
    L, _, _ = build_label_matrix(df, method_mapping, groups, granularity)
    probabilities = model.predict_proba(L)
    out = df.copy()
    for index, group in enumerate(groups):
        col_name = f"weak_prob_{granularity}_{group.lower()}"
        out[col_name] = probabilities[:, index]
    labels = np.asarray([groups[int(index)] for index in probabilities.argmax(axis=1)], dtype=object)
    active_count = (L != ABSTAIN).sum(axis=1)
    labels[active_count == 0] = ABSTAIN_TEXT
    out[f"weak_{granularity}_fault"] = labels
    out[f"weak_{granularity}_fault_group"] = [unify_fault(value) for value in labels]
    out[f"weak_{granularity}_posterior_max"] = probabilities.max(axis=1)
    entropy = (-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=1) / np.log(max(len(groups), 2)))
    out[f"weak_{granularity}_entropy"] = entropy
    out[f"weak_{granularity}_lf_active_count"] = active_count
    out[f"weak_{granularity}_lf_coverage"] = active_count / max(L.shape[1], 1) * 100.0
    out[f"weak_{granularity}_is_ABSTAIN"] = (active_count == 0)
    out[f"weak_{granularity}_backend"] = type(model).__name__
    logger.debug("predict_from_label_model: active LF mean=%.2f, abstain rate=%.2f%%", active_count.mean(), (active_count==0).mean()*100)
    return out

def create_student_training_targets(df, target_column, abstain_column, groups):
    if target_column not in df.columns:
        logger.error("create_student_training_targets: target column missing")
        raise ValueError(f"Missing weak target column: {target_column}")
    work = df.copy()
    target = work[target_column].astype(str).str.upper().str.strip()
    if abstain_column in work.columns:
        keep = ~work[abstain_column].astype(bool)
    else:
        keep = target != ABSTAIN_TEXT
    allowed = {str(value).upper() for value in groups}
    keep &= target.isin(allowed)
    clean = work.loc[keep].copy()
    if clean.empty:
        logger.error("create_student_training_targets: no usable weak labels remain")
        raise ValueError("No usable weak labels remain.")
    present = [group for group in groups if group in set(target.loc[keep])]
    if len(present) < 2:
        logger.error("create_student_training_targets: only %d classes present", len(present))
        raise ValueError("Student requires at least two classes.")
    mapping = {group: index for index, group in enumerate(present)}
    y = target.loc[keep].map(mapping).to_numpy(dtype=np.int64)
    logger.debug("create_student_training_targets: clean rows=%d, present classes=%s", len(clean), present)
    return clean.reset_index(drop=True), y, present

def weak_supervision_pipeline(df, label_columns=None, groups=None, use_snorkel=True, random_state=None, granularity="coarse"):
    seed = cfg.RANDOM_STATE if random_state is None else int(random_state)
    if granularity == "coarse":
        default_groups = cfg.WEAK_COARSE_GROUPS
    elif granularity == "fine":
        default_groups = cfg.WEAK_FINE_GROUPS
    else:
        logger.error("weak_supervision_pipeline: invalid granularity %s", granularity)
        raise ValueError("granularity must be coarse or fine.")
    groups = list(groups or default_groups)
    logger.debug("Weak supervision START | granularity=%s | rows=%d | LFs=%d", granularity, len(df), len(label_columns or DEFAULT_WEAK_METHODS))
    L, methods, groups = build_label_matrix(df, label_columns, groups, granularity)
    model, probabilities, backend = fit_label_model(
        L,
        len(groups),
        use_snorkel=use_snorkel,
        random_state=seed,
    )
    out = predict_from_label_model(model, df, methods, groups, granularity)
    pairwise = pairwise_label_agreement(df, methods)
    active_count = (L != ABSTAIN).sum(axis=1)
    metadata = {
        "backend": backend,
        "granularity": granularity,
        "groups": list(groups),
        "methods": list(methods),
        "n_rows": int(len(df)),
        "n_lfs": int(L.shape[1]),
        "rows_with_at_least_one_lf": int((active_count > 0).sum()),
        "abstain_rate": float((active_count == 0).mean()) if len(df) else 0.0,
        "mean_active_lf_count": float(active_count.mean()) if len(df) else 0.0,
        "uses_manual_lf_weights": False,
        "pairwise": pairwise.to_dict(orient="records"),
    }
    logger.debug("Weak supervision COMPLETE | granularity=%s | backend=%s | abstain=%.2f%% | active_LF_mean=%.2f", granularity, backend, metadata["abstain_rate"] * 100.0, metadata["mean_active_lf_count"])
    return out, model, groups, metadata, L, probabilities, pairwise

def save_weak_supervision_artifacts(df, model, groups, metadata, output_path=None, granularity="coarse"):
    output_path = Path(output_path or DATASET_DIR / "processed" / f"dga_weak_labels_{granularity}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.parent / f"dga_weak_label_metadata_{granularity}.json"
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        logger.warning(
            "save_weak_supervision_artifacts: duplicate columns detected; "
            "keeping first occurrence: %s",
            duplicated,
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    if df.empty:
        raise ValueError(
            f"Cannot save empty weak-supervision output for granularity={granularity}"
        )
    df.to_parquet(output_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "groups": list(groups), "metadata": metadata}, MODEL_DIR / f"weak_label_model_{granularity}.joblib")
    logger.debug("save_weak_supervision_artifacts: saved parquet=%s, metadata=%s, model=%s", output_path, metadata_path, MODEL_DIR / f"weak_label_model_{granularity}.joblib")
    return output_path, metadata_path

def load_weak_supervision_artifact(granularity="fine"):
    path = MODEL_DIR / f"weak_label_model_{granularity}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing weak-label model artifact: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid weak-label artifact: {path}")
    return payload