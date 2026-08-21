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
    if len(groups) < 2: raise ValueError("Need at least two weak-supervision classes.")
    if len(set(groups)) != len(groups): raise ValueError("Weak-supervision classes must be unique.")
    return groups

def _normalize_coarse_vote(raw_label):
    fine = normalize_fault(raw_label)
    if fine == ABSTAIN_TEXT: return ABSTAIN_TEXT
    coarse = unify_fault(fine)
    if coarse in cfg.WEAK_COARSE_GROUPS: return coarse
    return ABSTAIN_TEXT

def _normalize_fine_vote(raw_label):
    fine = normalize_fault(raw_label)
    if fine in cfg.BENCHMARK_FINE_CLASSES: return fine
    return ABSTAIN_TEXT

def build_label_matrix(df, label_columns=None, groups=None, granularity="coarse"):
    methods = dict(label_columns or DEFAULT_WEAK_METHODS)
    granularity = str(granularity).strip().lower()
    if granularity == "coarse": default_groups = cfg.WEAK_COARSE_GROUPS; normalizer = _normalize_coarse_vote
    elif granularity == "fine": default_groups = cfg.WEAK_FINE_GROUPS; normalizer = _normalize_fine_vote
    else: raise ValueError("granularity must be 'coarse' or 'fine'.")
    groups_list = _validate_groups(groups or default_groups)
    group_to_int = {group: idx for idx, group in enumerate(groups_list)}
    L = np.full((len(df), len(methods)), ABSTAIN, dtype=np.int64)
    for j, column in enumerate(methods.values()):
        if column not in df.columns: continue
        values = df[column].map(normalizer)
        L[:, j] = values.map(lambda value: group_to_int.get(value, ABSTAIN)).to_numpy(dtype=np.int64)
    return L, list(methods.keys()), groups_list

class EMLabelModel:
    def __init__(self, cardinality, random_state=42, max_iter=500, tol=1e-7, smoothing=1e-3):
        self.cardinality = int(cardinality); self.random_state = int(random_state); self.max_iter = int(max_iter); self.tol = float(tol); self.smoothing = float(smoothing)
        self.class_prior_ = None; self.confusion_ = None; self.n_iter_ = 0; self.fitted_ = False

    def _initialize(self, L):
        n, m = L.shape; k = self.cardinality
        rng = np.random.default_rng(self.random_state)
        counts = np.array([(L == class_id).sum() for class_id in range(k)], dtype=float)
        prior = counts + self.smoothing
        self.class_prior_ = prior / prior.sum()
        self.confusion_ = np.full((m, k, k + 1), self.smoothing, dtype=float)
        for j in range(m):
            for latent in range(k):
                self.confusion_[j, latent, latent] += 1.0
        self.confusion_[:, :, k] += 0.25
        self.confusion_ += rng.uniform(0.0, 1e-9, size=self.confusion_.shape)
        self.confusion_ /= self.confusion_.sum(axis=2, keepdims=True)

    def _log_joint(self, L):
        n, m = L.shape; k = self.cardinality
        result = np.tile(np.log(np.clip(self.class_prior_, 1e-12, None)), (n, 1))
        for j in range(m):
            for latent in range(k):
                for observed in range(k + 1):
                    observed_value = ABSTAIN if observed == k else observed
                    mask = L[:, j] == observed_value
                    if not np.any(mask): continue
                    result[mask, latent] += np.log(np.clip(self.confusion_[j, latent, observed], 1e-12, None))
        return result

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)

    def fit(self, L):
        L = np.asarray(L, dtype=np.int64)
        if L.ndim != 2 or len(L) == 0 or L.shape[1] == 0: raise ValueError("L must be a non-empty 2D array.")
        self._initialize(L)
        previous = None
        for iteration in range(self.max_iter):
            posterior = self._softmax(self._log_joint(L))
            prior = np.clip(posterior.mean(axis=0), self.smoothing, None); prior /= prior.sum()
            k = self.cardinality
            confusion = np.full_like(self.confusion_, self.smoothing)
            for j in range(L.shape[1]):
                for latent in range(k):
                    for observed in range(k):
                        mask = L[:, j] == observed
                        if np.any(mask): confusion[j, latent, observed] += posterior[mask, latent].sum()
                    abstain_mask = L[:, j] == ABSTAIN
                    if np.any(abstain_mask): confusion[j, latent, k] += posterior[abstain_mask, latent].sum()
            confusion /= confusion.sum(axis=2, keepdims=True)
            self.class_prior_ = prior; self.confusion_ = confusion
            log_joint = self._log_joint(L)
            maximum = log_joint.max(axis=1)
            objective = float(np.mean(np.log(np.clip(np.exp(log_joint - maximum[:, None]).sum(axis=1), 1e-12, None)) + maximum))
            self.n_iter_ = iteration + 1
            if previous is not None and abs(objective - previous) < self.tol: break
            previous = objective
        self.fitted_ = True
        return self

    def predict_proba(self, L):
        if not self.fitted_: raise RuntimeError("Model is not fitted.")
        return self._softmax(self._log_joint(np.asarray(L, dtype=np.int64)))

    def predict(self, L): return self.predict_proba(L).argmax(axis=1)

def fit_label_model(L, cardinality, use_snorkel=True, random_state=42):
    L = np.asarray(L, dtype=np.int64)
    if L.ndim != 2 or L.shape[1] == 0: raise ValueError("L must be a non-empty 2D matrix.")
    if use_snorkel and SNORKEL_AVAILABLE:
        try:
            logger.info("Weak supervision backend: Snorkel LabelModel")
            model = LabelModel(cardinality=cardinality, verbose=False)
            model.fit(L_train=L, n_epochs=500, log_freq=100, seed=random_state)
            return model, model.predict_proba(L), "snorkel"
        except Exception as exc:
            logger.warning("Snorkel failed; using EM fallback: %s", exc)
    logger.info("Weak supervision backend: EM fallback")
    model = EMLabelModel(cardinality=cardinality, random_state=random_state)
    model.fit(L)
    return model, model.predict_proba(L), "em"

def predict_from_label_model(model, df, methods, groups, granularity):
    method_mapping = {method: cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method] for method in methods}
    L, _, _ = build_label_matrix(df, method_mapping, groups, granularity)
    probabilities = model.predict_proba(L)
    out = df.copy()
    for index, group in enumerate(groups): out[f"weak_prob_{granularity}_{group.lower()}"] = probabilities[:, index]
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
    return out

def create_student_training_targets(df, target_column, abstain_column, groups):
    if target_column not in df.columns: raise ValueError(f"Missing weak target column: {target_column}")
    work = df.copy()
    target = work[target_column].astype(str).str.upper().str.strip()
    if abstain_column in work.columns: keep = ~work[abstain_column].astype(bool)
    else: keep = target != ABSTAIN_TEXT
    allowed = {str(value).upper() for value in groups}
    keep &= target.isin(allowed)
    clean = work.loc[keep].copy()
    if clean.empty: raise ValueError("No usable weak labels remain.")
    present = [group for group in groups if group in set(target.loc[keep])]
    if len(present) < 2: raise ValueError("Student requires at least two classes.")
    mapping = {group: index for index, group in enumerate(present)}
    y = target.loc[keep].map(mapping).to_numpy(dtype=np.int64)
    return clean.reset_index(drop=True), y, present

def weak_supervision_pipeline(df, label_columns=None, groups=None, use_snorkel=True, random_state=None, granularity="coarse"):
    seed = cfg.RANDOM_STATE if random_state is None else int(random_state)
    if granularity == "coarse": default_groups = cfg.WEAK_COARSE_GROUPS
    elif granularity == "fine": default_groups = cfg.WEAK_FINE_GROUPS
    else: raise ValueError("granularity must be coarse or fine.")
    groups = list(groups or default_groups)
    logger.info("Weak supervision START | granularity=%s | rows=%d | LFs=%d", granularity, len(df), len(label_columns or DEFAULT_WEAK_METHODS))
    L, methods, groups = build_label_matrix(df, label_columns, groups, granularity)
    model, probabilities, backend = fit_label_model(L, len(groups), use_snorkel=use_snorkel, random_state=seed)
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
    logger.info("Weak supervision COMPLETE | granularity=%s | backend=%s | abstain=%.2f%% | active_LF_mean=%.2f", granularity, backend, metadata["abstain_rate"] * 100.0, metadata["mean_active_lf_count"])
    return out, model, groups, metadata, L, probabilities, pairwise

def save_weak_supervision_artifacts(df, model, groups, metadata, output_path=None, granularity="coarse"):
    output_path = Path(output_path or DATASET_DIR / "processed" / f"dga_weak_labels_{granularity}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.parent / f"dga_weak_label_metadata_{granularity}.json"
    df.to_parquet(output_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "groups": list(groups), "metadata": metadata}, MODEL_DIR / f"weak_label_model_{granularity}.joblib")
    return output_path, metadata_path