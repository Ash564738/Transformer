# anomaly.py
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

class AutoencoderAnomaly:
    def __init__(self, hidden_units=3, random_state=42):
        self.hidden_units = hidden_units
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        logger.debug("Training Autoencoder (MLP) with hidden_units=%d on %d samples",
                     self.hidden_units, X.shape[0])
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_units,),
            activation='relu', solver='adam',
            max_iter=400, random_state=self.random_state,
        )
        self.model.fit(X, X)
        logger.debug("Autoencoder training completed")
        return self

    def predict(self, X):
        X_pred = self.model.predict(X)
        errors = np.mean(np.square(X - X_pred), axis=1)
        logger.debug("Autoencoder reconstruction error: mean=%.4f, std=%.4f",
                     np.mean(errors), np.std(errors))
        return errors


class UnsupervisedEnsemble:
    """
    Unsupervised anomaly ensemble (IForest + LOF + OCSVM + Autoencoder).

    No ground truth is assumed anywhere. Instead of fixed equal weights,
    each detector's ensemble weight is derived from how much it *agrees*
    with the others on the training batch (agreement-based weighting) —
    a detector that disagrees wildly with the rest of the ensemble gets
    down-weighted automatically, without needing labels.
    """

    def __init__(
        self,
        random_state=42,
        contamination='auto',
        ocsvm_nu=0.1,
        quantile_range=(10.0, 90.0),
        min_lof_neighbors=5,
        max_lof_neighbors=20,
    ):
        self.random_state = random_state
        self.contamination = contamination
        self.ocsvm_nu = ocsvm_nu
        self.quantile_range = quantile_range
        self.min_lof_neighbors = min_lof_neighbors
        self.max_lof_neighbors = max_lof_neighbors

        self.detectors = None
        self.calibration_bins = {}
        self.weights = {}
        self.scaler = RobustScaler(quantile_range=quantile_range)
        self._last_diagnostics = None
        self._feature_names = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, a_min=0.0, a_max=None)
        X_log = np.log1p(X)
        if fit:
            logger.debug("Preprocessing (fit): RobustScaler on log1p-transformed data")
            return self.scaler.fit_transform(X_log)
        return self.scaler.transform(X_log)

    def _build_detectors(self, n_samples):
        n_neighbors = int(np.clip(n_samples - 1, self.min_lof_neighbors, self.max_lof_neighbors))
        logger.info("Building detectors: IForest, LOF (n_neighbors=%d), OCSVM, Autoencoder", n_neighbors)
        return {
            'iforest': IsolationForest(
                n_estimators=100, contamination=self.contamination,
                random_state=self.random_state,
            ),
            'lof': LocalOutlierFactor(
                n_neighbors=n_neighbors, novelty=True,
                contamination=self.contamination, metric='minkowski',
            ),
            'ocsvm': OneClassSVM(kernel='rbf', gamma='scale', nu=self.ocsvm_nu),
            'autoencoder': AutoencoderAnomaly(random_state=self.random_state),
        }

    def _raw_scores(self, name, det, X_scaled):
        if name == 'autoencoder':
            return det.predict(X_scaled)          # reconstruction error, higher = anomalous
        return -det.decision_function(X_scaled)    # invert so higher = anomalous

    def _rank_calibrate(self, name, scores):
        bins = self.calibration_bins[name]
        return np.searchsorted(bins, scores, side='right') / len(bins)

    def _agreement_weights(self, calibrated_train):
        """
        calibrated_train: dict[name] -> 1D array of rank-calibrated scores
        on the training batch. Weight each detector by its average
        correlation with all other detectors (min-clipped at 0 so a
        detector that's purely noise doesn't get a negative weight).
        """
        names = list(calibrated_train.keys())
        n = len(names)
        if n <= 1:
            w = {names[0]: 1.0} if names else {}
            logger.info("Agreement weights: single detector, weight=1.0")
            return w

        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    si, sj = calibrated_train[names[i]], calibrated_train[names[j]]
                    if np.std(si) < 1e-9 or np.std(sj) < 1e-9:
                        c = 0.0
                    else:
                        c = np.corrcoef(si, sj)[0, 1]
                    corr[i, j] = 0.0 if np.isnan(c) else max(c, 0.0)

        avg_agreement = corr.mean(axis=1)
        logger.debug("Detector agreement matrix:\n%s", corr)
        logger.debug("Average agreement: %s", dict(zip(names, avg_agreement)))

        if avg_agreement.sum() <= 0:
            logger.warning("All detectors have zero agreement; falling back to equal weights.")
            return {name: 1.0 / n for name in names}

        weights = avg_agreement / avg_agreement.sum()
        weight_dict = {name: float(w) for name, w in zip(names, weights)}
        logger.info("Agreement-based weights: %s", weight_dict)
        return weight_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, feature_names=None):
        logger.info("Fitting UnsupervisedEnsemble on %d samples, %d features",
                    X.shape[0], X.shape[1] if X.ndim > 1 else 1)
        X = np.asarray(X, dtype=float)
        self._feature_names = feature_names
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("UnsupervisedEnsemble.fit needs at least 2 samples.")

        X_scaled = self._preprocess(X, fit=True)
        self.detectors = self._build_detectors(n_samples)

        calibrated_train = {}
        for name, det in self.detectors.items():
            logger.info("Fitting detector: %s", name)
            det.fit(X_scaled)
            scores = self._raw_scores(name, det, X_scaled)
            self.calibration_bins[name] = np.sort(scores)
            calibrated_train[name] = self._rank_calibrate(name, scores)
            logger.debug("Detector %s: raw score range [%.4f, %.4f]", name, np.min(scores), np.max(scores))

        self.weights = self._agreement_weights(calibrated_train)
        logger.info("Ensemble fitting complete.")
        return self

    def predict(self, X, return_diagnostics=False):
        if self.detectors is None:
            raise RuntimeError("Call fit() before predict().")

        X_scaled = self._preprocess(X, fit=False)

        calibrated = np.zeros((X_scaled.shape[0], len(self.detectors)))
        per_detector = {}
        names = list(self.detectors.keys())
        for i, name in enumerate(names):
            scores = self._raw_scores(name, self.detectors[name], X_scaled)
            ranks = self._rank_calibrate(name, scores)
            calibrated[:, i] = ranks
            per_detector[name] = ranks

        weights = np.array([self.weights.get(name, 0.0) for name in names])
        if weights.sum() <= 0:
            weights = np.full(len(names), 1.0 / len(names))
        else:
            weights = weights / weights.sum()

        ensemble_score = np.dot(calibrated, weights)
        self._last_diagnostics = {
            "per_detector": per_detector,
            "weights": dict(zip(names, weights.tolist())),
        }
        logger.debug("Prediction completed for %d samples", X_scaled.shape[0])
        if return_diagnostics:
            return ensemble_score, self._last_diagnostics
        return ensemble_score

    def fit_predict(self, X, feature_names=None, return_diagnostics=False):
        self.fit(X, feature_names=feature_names)
        return self.predict(X, return_diagnostics=return_diagnostics)

    def get_last_diagnostics(self):
        """Per-detector rank scores + weights from the most recent predict()
        call — use this to explain WHY a row scored high (which detector(s)
        flagged it), instead of only seeing the blended number."""
        return self._last_diagnostics