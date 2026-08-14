# anomaly.py
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


# ============================================================================
# Autoencoder
# ============================================================================

class AutoencoderAnomaly:
    def __init__(
        self,
        hidden_units: int = 8,
        random_state: int = 42,
        max_iter: int = 500,
    ):
        self.hidden_units = hidden_units
        self.random_state = random_state
        self.max_iter = max_iter
        self.model: Optional[MLPRegressor] = None

    def fit(self, X):
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "AutoencoderAnomaly.fit expects a 2D array."
            )

        if X.shape[0] < 2:
            raise ValueError(
                "AutoencoderAnomaly.fit requires at least 2 samples."
            )

        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_units,),
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
        )

        self.model.fit(X, X)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(
                "AutoencoderAnomaly must be fitted before predict()."
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "AutoencoderAnomaly.predict expects a 2D array."
            )

        reconstructed = self.model.predict(X)

        return np.mean(
            np.square(
                X - reconstructed
            ),
            axis=1,
        )


# ============================================================================
# Ensemble
# ============================================================================

class UnsupervisedEnsemble:
    """
    Unsupervised anomaly ensemble:

        Isolation Forest
        LOF
        One-Class SVM
        MLP Autoencoder

    Output:
        anomaly score in [0, 1].

    Important:
        This is an anomaly percentile/ranking score.
        It is NOT a probability of transformer failure.
    """

    def __init__(
        self,
        random_state: int = 42,
        contamination="auto",
        ocsvm_nu: float = 0.1,
        quantile_range=(10.0, 90.0),
        min_lof_neighbors: int = 5,
        max_lof_neighbors: int = 20,
        autoencoder_hidden_units: int = 8,
    ):
        self.random_state = random_state
        self.contamination = contamination
        self.ocsvm_nu = ocsvm_nu
        self.quantile_range = quantile_range
        self.min_lof_neighbors = min_lof_neighbors
        self.max_lof_neighbors = max_lof_neighbors
        self.autoencoder_hidden_units = autoencoder_hidden_units

        self.detectors: Optional[Dict[str, object]] = None
        self.calibration_bins: Dict[str, np.ndarray] = {}
        self.weights: Dict[str, float] = {}

        self.scaler = RobustScaler(
            quantile_range=quantile_range
        )

        self._feature_names = None
        self._last_diagnostics = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_array(X):
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "Anomaly model expects a 2D feature matrix."
            )

        if X.shape[0] < 1:
            raise ValueError(
                "At least one sample is required."
            )

        return X

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        X,
        fit=False,
    ):
        X = self._prepare_array(X)

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Signed log transform allows negative engineered deltas/rates.
        X_log = (
            np.sign(X)
            * np.log1p(
                np.abs(X)
            )
        )

        if fit:
            return self.scaler.fit_transform(
                X_log
            )

        if not hasattr(
            self.scaler,
            "center_",
        ):
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        return self.scaler.transform(
            X_log
        )

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _build_detectors(
        self,
        n_samples,
    ):
        if n_samples < 2:
            raise ValueError(
                "At least 2 training samples are required."
            )

        n_neighbors = min(
            self.max_lof_neighbors,
            n_samples - 1,
        )

        n_neighbors = max(
            self.min_lof_neighbors,
            n_neighbors,
        )

        n_neighbors = min(
            n_neighbors,
            n_samples - 1,
        )

        return {
            "iforest": IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "lof": LocalOutlierFactor(
                n_neighbors=n_neighbors,
                novelty=True,
                contamination=self.contamination,
                metric="minkowski",
                n_jobs=-1,
            ),
            "ocsvm": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.ocsvm_nu,
            ),
            "autoencoder": AutoencoderAnomaly(
                hidden_units=self.autoencoder_hidden_units,
                random_state=self.random_state,
            ),
        }

    # ------------------------------------------------------------------
    # Training scores
    # ------------------------------------------------------------------

    def _training_raw_score(
        self,
        name,
        detector,
        X_scaled,
    ):
        if name == "autoencoder":
            return detector.predict(
                X_scaled
            )

        if name == "lof":
            # Correct training-set LOF score.
            # scikit-learn documents that novelty=True
            # decision_function/predict should only be used on unseen data.
            return -np.asarray(
                detector.negative_outlier_factor_,
                dtype=float,
            )

        return -np.asarray(
            detector.decision_function(
                X_scaled
            ),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Unseen-data scores
    # ------------------------------------------------------------------

    def _prediction_raw_score(
        self,
        name,
        detector,
        X_scaled,
    ):
        if name == "autoencoder":
            return detector.predict(
                X_scaled
            )

        return -np.asarray(
            detector.decision_function(
                X_scaled
            ),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _rank_calibrate(
        self,
        name,
        scores,
    ):
        bins = self.calibration_bins[name]

        if len(bins) == 0:
            return np.zeros(
                len(scores),
                dtype=float,
            )

        return (
            np.searchsorted(
                bins,
                scores,
                side="right",
            )
            / float(len(bins))
        )

    # ------------------------------------------------------------------
    # Agreement weights
    # ------------------------------------------------------------------

    @staticmethod
    def _agreement_weights(
        calibrated_train,
    ):
        names = list(
            calibrated_train.keys()
        )

        n = len(names)

        if n == 0:
            return {}

        if n == 1:
            return {
                names[0]: 1.0
            }

        corr = np.eye(
            n,
            dtype=float,
        )

        for i in range(n):
            for j in range(i + 1, n):

                a = calibrated_train[
                    names[i]
                ]

                b = calibrated_train[
                    names[j]
                ]

                if (
                    np.std(a) < 1e-12
                    or np.std(b) < 1e-12
                ):
                    c = 0.0
                else:
                    c = np.corrcoef(
                        a,
                        b,
                    )[0, 1]

                    if not np.isfinite(c):
                        c = 0.0

                c = max(
                    float(c),
                    0.0,
                )

                corr[i, j] = c
                corr[j, i] = c

        avg = (
            corr.sum(axis=1) - 1.0
        ) / max(
            n - 1,
            1,
        )

        if (
            not np.isfinite(avg).all()
            or avg.sum() <= 0
        ):
            weights = np.full(
                n,
                1.0 / n,
            )
        else:
            weights = (
                avg / avg.sum()
            )

        return {
            name: float(weight)
            for name, weight in zip(
                names,
                weights,
            )
        }

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        feature_names=None,
    ):
        X = self._prepare_array(X)

        if X.shape[0] < 2:
            raise ValueError(
                "UnsupervisedEnsemble.fit requires at least 2 samples."
            )

        self._feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        X_scaled = self._preprocess(
            X,
            fit=True,
        )

        self.detectors = (
            self._build_detectors(
                X.shape[0]
            )
        )

        calibrated_train = {}

        for name, detector in (
            self.detectors.items()
        ):

            logger.info(
                "Fitting anomaly detector: %s",
                name,
            )

            detector.fit(
                X_scaled
            )

            scores = self._training_raw_score(
                name,
                detector,
                X_scaled,
            )

            scores = np.nan_to_num(
                np.asarray(
                    scores,
                    dtype=float,
                ),
                nan=0.0,
                posinf=np.finfo(
                    float
                ).max,
                neginf=np.finfo(
                    float
                ).min,
            )

            self.calibration_bins[name] = (
                np.sort(scores)
            )

            calibrated_train[name] = (
                self._rank_calibrate(
                    name,
                    scores,
                )
            )

        self.weights = (
            self._agreement_weights(
                calibrated_train
            )
        )

        self._is_fitted = True

        logger.info(
            "Unsupervised anomaly ensemble fitted. weights=%s",
            self.weights,
        )

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X,
        return_diagnostics=False,
    ):
        if not self._is_fitted:
            raise RuntimeError(
                "Call fit() before predict()."
            )

        X = self._prepare_array(X)

        X_scaled = self._preprocess(
            X,
            fit=False,
        )

        names = list(
            self.detectors.keys()
        )

        calibrated = np.zeros(
            (
                len(X_scaled),
                len(names),
            ),
            dtype=float,
        )

        per_detector = {}

        for i, name in enumerate(
            names
        ):

            raw = self._prediction_raw_score(
                name,
                self.detectors[name],
                X_scaled,
            )

            raw = np.nan_to_num(
                raw,
                nan=0.0,
                posinf=np.finfo(
                    float
                ).max,
                neginf=np.finfo(
                    float
                ).min,
            )

            rank = self._rank_calibrate(
                name,
                raw,
            )

            calibrated[:, i] = rank
            per_detector[name] = rank

        weights = np.asarray(
            [
                self.weights.get(
                    name,
                    0.0,
                )
                for name in names
            ],
            dtype=float,
        )

        if (
            not np.isfinite(weights).all()
            or weights.sum() <= 0
        ):
            weights = np.full(
                len(names),
                1.0 / len(names),
            )
        else:
            weights /= weights.sum()

        score = np.clip(
            calibrated.dot(
                weights
            ),
            0.0,
            1.0,
        )

        self._last_diagnostics = {
            "per_detector": per_detector,
            "weights": {
                name: float(weight)
                for name, weight in zip(
                    names,
                    weights,
                )
            },
            "feature_names": self._feature_names,
        }

        if return_diagnostics:
            return (
                score,
                self._last_diagnostics,
            )

        return score

    # ------------------------------------------------------------------
    # Fit predict
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        X,
        feature_names=None,
        return_diagnostics=False,
    ):
        self.fit(
            X,
            feature_names=feature_names,
        )

        return self.predict(
            X,
            return_diagnostics=return_diagnostics,
        )

    def get_last_diagnostics(self):
        return self._last_diagnostics