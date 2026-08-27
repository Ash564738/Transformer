# research_analysis.py
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kendalltau, ks_2samp, spearmanr

logger = logging.getLogger(__name__)

CROSS_TRANSFER_CLASSES = ("NORMAL", "PD", "D1", "D2", "T1_T2", "T3")

DEFAULT_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
RATIO_FEATURES = [
    "pct5_h2", "pct5_c2h6", "pct5_ch4", "pct5_c2h4", "pct5_c2h2",
    "pct_ch4_tri", "pct_c2h4_tri", "pct_c2h2_tri",
    "ratio_ch4_h2", "ratio_c2h2_c2h4", "ratio_c2h4_c2h6",
]
RAW_LOG_FEATURES = [f"log1p_{g}" for g in DEFAULT_GASES]
EPS = 1e-6


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    return x.where(x >= 0)


def add_ratio_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a small, auditable scale-invariant feature set for domain-gap ablation.

    This is kept separate from the production feature builder. It is used only
    for research analysis and model ablation, so changing it cannot silently
    change the production inference feature contract.
    """
    out = df.copy()

    for gas in DEFAULT_GASES:
        x = _numeric_series(out, gas)
        out[f"log1p_{gas}"] = np.log1p(x)

    five = [_numeric_series(out, g) for g in ["h2", "c2h6", "ch4", "c2h4", "c2h2"]]
    total5 = sum(five).replace(0, np.nan)
    for gas in ["h2", "c2h6", "ch4", "c2h4", "c2h2"]:
        out[f"pct5_{gas}"] = _numeric_series(out, gas) / total5 * 100.0

    tri_total = (
        _numeric_series(out, "ch4")
        + _numeric_series(out, "c2h4")
        + _numeric_series(out, "c2h2")
    ).replace(0, np.nan)

    out["pct_ch4_tri"] = _numeric_series(out, "ch4") / tri_total * 100.0
    out["pct_c2h4_tri"] = _numeric_series(out, "c2h4") / tri_total * 100.0
    out["pct_c2h2_tri"] = _numeric_series(out, "c2h2") / tri_total * 100.0

    out["ratio_ch4_h2"] = _numeric_series(out, "ch4") / (
        _numeric_series(out, "h2") + EPS
    )
    out["ratio_c2h2_c2h4"] = _numeric_series(out, "c2h2") / (
        _numeric_series(out, "c2h4") + EPS
    )
    out["ratio_c2h4_c2h6"] = _numeric_series(out, "c2h4") / (
        _numeric_series(out, "c2h6") + EPS
    )

    return out


def build_domain_gap_analysis(
    labeled_frames: Iterable[tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> pd.DataFrame:
    """Compare absolute concentration vs scale-invariant DGA representations.

    The KS statistic is a descriptive domain-gap diagnostic only. It is not a
    training objective and it is never used to choose deployment weights.
    """
    frames = []
    for source_name, frame in labeled_frames:
        tmp = frame.copy()
        tmp["source_dataset"] = source_name
        frames.append(tmp)

    if len(frames) != 2:
        raise ValueError("build_domain_gap_analysis currently expects exactly two datasets.")

    a_name, a = frames[0]["source_dataset"].iloc[0], frames[0]
    b_name, b = frames[1]["source_dataset"].iloc[0], frames[1]

    a = add_ratio_analysis_features(a)
    b = add_ratio_analysis_features(b)

    rows = []
    for feature in RAW_LOG_FEATURES + RATIO_FEATURES:
        xa = pd.to_numeric(a[feature], errors="coerce").dropna()
        xb = pd.to_numeric(b[feature], errors="coerce").dropna()
        if len(xa) == 0 or len(xb) == 0:
            continue

        statistic, p_value = ks_2samp(xa, xb)
        rows.append(
            {
                "representation": "absolute_log_concentration"
                if feature in RAW_LOG_FEATURES
                else "scale_invariant_ratio_percentage",
                "feature": feature,
                "dataset_a": a_name,
                "dataset_b": b_name,
                "n_a": int(len(xa)),
                "n_b": int(len(xb)),
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "median_a": float(np.median(xa)),
                "median_b": float(np.median(xb)),
                "median_abs_gap": float(abs(np.median(xa) - np.median(xb))),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(
            output_dir / "domain_gap_absolute_vs_ratio.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_rows = []
        for rep, sub in result.groupby("representation", sort=False):
            summary_rows.append(
                {
                    "representation": rep,
                    "features_evaluated": int(len(sub)),
                    "mean_ks": float(sub["ks_statistic"].mean()),
                    "median_ks": float(sub["ks_statistic"].median()),
                    "max_ks": float(sub["ks_statistic"].max()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "domain_gap_representation_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    logger.info(
        "Domain-gap analysis complete | %s vs %s | features=%d",
        a_name,
        b_name,
        len(result),
    )
    return result


def build_rank_correlation_analysis(
    ranking: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fleet order against independent descriptive baselines.

    TDCG is included only as a baseline comparator. It is NOT a ranking input
    and this analysis does not replace the documented IEEE evidence ordering.
    """
    work = ranking.copy()

    if "tdcg" in work.columns:
        tdcg = pd.to_numeric(work["tdcg"], errors="coerce")
    elif "tdcg_raw" in work.columns:
        tdcg = pd.to_numeric(work["tdcg_raw"], errors="coerce")
    else:
        tdcg = pd.Series(np.nan, index=work.index)

    work["tdcg_baseline"] = tdcg

    candidates = {
        "Fleet rank (lower = higher priority)": pd.to_numeric(
            work.get("rank"), errors="coerce"
        ),
        "Current concentration evidence": pd.to_numeric(
            work.get("current_concentration_exceedance_ratio"),
            errors="coerce",
        ),
        "Current rate evidence": pd.to_numeric(
            work.get("current_rate_exceedance_ratio"),
            errors="coerce",
        ),
        "Current delta evidence": pd.to_numeric(
            work.get("current_delta_exceedance_ratio"),
            errors="coerce",
        ),
        "TDCG baseline": pd.to_numeric(work["tdcg_baseline"], errors="coerce"),
    }

    names = list(candidates)
    rho_rows = []
    tau_rows = []

    for left, right in combinations(names, 2):
        x = candidates[left]
        y = candidates[right]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 3:
            rho = np.nan
            tau = np.nan
            n = int(mask.sum())
        else:
            rho, _ = spearmanr(x[mask], y[mask])
            tau, _ = kendalltau(x[mask], y[mask])
            n = int(mask.sum())

        rho_rows.append({"metric_a": left, "metric_b": right, "n": n, "spearman_rho": rho})
        tau_rows.append({"metric_a": left, "metric_b": right, "n": n, "kendall_tau": tau})

    spearman = pd.DataFrame(rho_rows)
    kendall = pd.DataFrame(tau_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    spearman.to_csv(
        output_dir / "rank_correlation_spearman.csv",
        index=False,
        encoding="utf-8-sig",
    )
    kendall.to_csv(
        output_dir / "rank_correlation_kendall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Rank correlation analysis complete | transformers=%d", len(work))
    return spearman, kendall


__all__ = [
    "RATIO_FEATURES",
    "build_domain_gap_analysis",
    "build_rank_correlation_analysis",
    "add_ratio_analysis_features",
]


def cross_dataset_transfer_grid(
    source_frames: Iterable[tuple[str, pd.DataFrame]],
    build_feature_frame,
    build_models,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Train on one labeled dataset and transfer to the other.

    This implements the useful part of the teammate's protocol: a genuinely
    cross-domain experiment, rather than evaluating only inside a pooled
    benchmark. No result from the target dataset is used for model selection.
    The output is descriptive research evidence; it does not select the
    production weak student.
    """
    frames = list(source_frames)
    if len(frames) != 2:
        raise ValueError("cross_dataset_transfer_grid expects exactly two datasets.")

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        label_col = "fault_type_label"
        if label_col not in out.columns:
            raise ValueError(f"Missing {label_col}")
        labels = out[label_col].astype(str).str.upper().str.strip()
        # Existing project normalization has already harmonized aliases before
        # this function is called; retain only the common cross-domain taxonomy.
        out["_cross_label"] = labels
        out = out[out["_cross_label"].isin(CROSS_TRANSFER_CLASSES)].copy()
        return out

    normalized = [(name, normalize(frame)) for name, frame in frames]
    rows = []

    for (train_name, train_df), (test_name, test_df) in (
        (normalized[0], normalized[1]),
        (normalized[1], normalized[0]),
    ):
        if train_df.empty or test_df.empty:
            continue

        common = sorted(
            set(train_df["_cross_label"].unique())
            & set(test_df["_cross_label"].unique())
        )
        train_use = train_df[train_df["_cross_label"].isin(common)].reset_index(drop=True)
        test_use = test_df[test_df["_cross_label"].isin(common)].reset_index(drop=True)

        if len(common) < 2:
            continue

        encoder = LabelEncoder().fit(common)
        y_train = encoder.transform(train_use["_cross_label"])
        y_test = encoder.transform(test_use["_cross_label"])

        for feature_mode in (
            "gas_only",
            "ratio_only",
            "gas_plus_ratio",
            "gas_plus_traditional",
        ):
            X_train = build_feature_frame(train_use, feature_mode, "fine")
            X_test = build_feature_frame(test_use, feature_mode, "fine")

            for model_name, model in build_models(seed).items():
                try:
                    model.fit(X_train, y_train)
                    pred = np.asarray(model.predict(X_test)).reshape(-1)
                    rows.append(
                        {
                            "train_dataset": train_name,
                            "test_dataset": test_name,
                            "feature_mode": feature_mode,
                            "model": model_name,
                            "n_train": int(len(train_use)),
                            "n_test": int(len(test_use)),
                            "classes_common": "|".join(common),
                            "accuracy": float(accuracy_score(y_test, pred)),
                            "balanced_accuracy": float(
                                balanced_accuracy_score(y_test, pred)
                            ),
                            "macro_f1": float(
                                f1_score(
                                    y_test,
                                    pred,
                                    labels=list(range(len(common))),
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                            "evaluation": "cross_dataset_transfer",
                            "selection_used_target_labels": False,
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Cross-dataset transfer failed | train=%s test=%s mode=%s model=%s | %s",
                        train_name,
                        test_name,
                        feature_mode,
                        model_name,
                        exc,
                    )

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_dir / "cross_dataset_transfer_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result