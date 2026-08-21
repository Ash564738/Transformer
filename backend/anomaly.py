# anomaly.py
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

class AutoencoderAnomaly:
    def __init__(self, hidden_units: int = 8, random_state: int = 42, max_iter: int = 500):
        self.hidden_units = int(hidden_units); self.random_state = int(random_state); self.max_iter = int(max_iter)
        self.model: Optional[MLPRegressor] = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("AutoencoderAnomaly.fit requires a 2D array with >=2 samples.")
        self.model = MLPRegressor(hidden_layer_sizes=(self.hidden_units,), activation="relu", solver="adam", max_iter=self.max_iter, random_state=self.random_state, early_stopping=False)
        self.model.fit(X, X)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("AutoencoderAnomaly must be fitted.")
        X = np.asarray(X, dtype=np.float64)
        reconstructed = self.model.predict(X)
        return np.mean((X - reconstructed) ** 2, axis=1)

class UnsupervisedEnsemble:
    def __init__(self, random_state: int = 42, contamination="auto", ocsvm_nu: float = 0.1,
                 quantile_range: Tuple[float, float] = (10.0, 90.0), min_lof_neighbors: int = 5,
                 max_lof_neighbors: int = 20, autoencoder_hidden_units: int = 8):
        self.random_state = int(random_state); self.contamination = contamination; self.ocsvm_nu = float(ocsvm_nu)
        self.quantile_range = quantile_range; self.min_lof_neighbors = int(min_lof_neighbors)
        self.max_lof_neighbors = int(max_lof_neighbors); self.autoencoder_hidden_units = int(autoencoder_hidden_units)
        self.detectors: Dict[str, object] = {}; self.calibration_bins: Dict[str, np.ndarray] = {}
        self.scaler = RobustScaler(quantile_range=quantile_range)
        self._feature_names = None; self._last_diagnostics = None; self._is_fitted = False

    @staticmethod
    def _prepare_array(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError("Anomaly model expects a non-empty 2D matrix.")
        return X

    @staticmethod
    def _safe_transform(X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sign(X) * np.log1p(np.abs(X))

    def _preprocess(self, X, fit=False):
        X = self._prepare_array(X); X = self._safe_transform(X)
        if fit:
            return self.scaler.fit_transform(X)
        if not hasattr(self.scaler, "center_"):
            raise RuntimeError("Scaler has not been fitted.")
        return self.scaler.transform(X)

    def _build_detectors(self, n_samples: int):
        if n_samples < 2:
            raise ValueError("At least two training samples are required.")
        detectors = {
            "iforest": IsolationForest(n_estimators=300, contamination=self.contamination, random_state=self.random_state, n_jobs=-1),
            "ocsvm": OneClassSVM(kernel="rbf", gamma="scale", nu=min(max(self.ocsvm_nu, 1.0 / n_samples), 0.5)),
            "autoencoder": AutoencoderAnomaly(hidden_units=self.autoencoder_hidden_units, random_state=self.random_state)
        }
        if n_samples >= 3:
            n_neighbors = min(self.max_lof_neighbors, n_samples - 1)
            n_neighbors = max(2, min(self.min_lof_neighbors, n_neighbors))
            if n_neighbors < n_samples:
                detectors["lof"] = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=self.contamination, n_jobs=-1)
        return detectors

    @staticmethod
    def _raw_score(name, detector, X_scaled):
        if name == "autoencoder":
            return detector.predict(X_scaled)
        return -np.asarray(detector.decision_function(X_scaled), dtype=float)

    def _rank_calibrate(self, name: str, scores: np.ndarray):
        bins = self.calibration_bins.get(name, np.array([]))
        if len(bins) == 0:
            return np.zeros(len(scores), dtype=float)
        percentile = np.searchsorted(bins, scores, side="right") / float(len(bins))
        return 100.0 * percentile

    def fit(self, X, feature_names=None):
        X = self._prepare_array(X)
        if X.shape[0] < 2:
            raise ValueError("UnsupervisedEnsemble.fit requires >=2 samples.")
        self._feature_names = list(feature_names) if feature_names is not None else None
        X_scaled = self._preprocess(X, fit=True)
        candidate_detectors = self._build_detectors(X.shape[0])
        self.detectors = {}; self.calibration_bins = {}
        for name, detector in candidate_detectors.items():
            logger.info("Fitting anomaly detector: %s", name)
            try:
                detector.fit(X_scaled)
                if name == "lof" and hasattr(detector, "negative_outlier_factor_"):
                    raw = -np.asarray(detector.negative_outlier_factor_, dtype=float)
                else:
                    raw = self._raw_score(name, detector, X_scaled)
                raw = np.nan_to_num(raw, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
                self.detectors[name] = detector
                self.calibration_bins[name] = np.sort(raw)
            except Exception:
                logger.exception("Anomaly detector failed: %s", name)
        if not self.detectors:
            raise RuntimeError("No anomaly detector could be fitted.")
        self._is_fitted = True
        return self

    def predict(self, X, return_diagnostics=False):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._preprocess(X, fit=False)
        calibrated = {}
        for name, detector in self.detectors.items():
            raw = self._raw_score(name, detector, X_scaled)
            raw = np.nan_to_num(raw, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
            calibrated[name] = self._rank_calibrate(name, raw)
        matrix = np.column_stack([calibrated[name] for name in self.detectors])
        score = np.median(matrix, axis=1)
        score = np.clip(score, 0.0, 100.0)
        self._last_diagnostics = {
            "per_detector": calibrated,
            "aggregation": "median",
            "feature_names": self._feature_names,
            "scale": "0-100 percentile",
            "interpretation": "relative anomaly position only"
        }
        if return_diagnostics:
            return score, self._last_diagnostics
        return score

    def fit_predict(self, X, feature_names=None, return_diagnostics=False):
        self.fit(X, feature_names=feature_names)
        return self.predict(X, return_diagnostics=return_diagnostics)

    def get_last_diagnostics(self):
        return self._last_diagnostics