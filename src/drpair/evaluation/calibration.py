"""Patient cross-fitted calibration and training-only clinical operating points."""

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .ordinal import project


def fit_calibrator(q, y, method):
    if method not in ("raw", "platt", "isotonic"):
        raise ValueError(f"Unknown calibration method: {method}")
    if method == "raw":
        return {"method": method, "boundaries": []}
    records = []
    for b in range(4):
        target = np.asarray(y) > b
        x = np.clip(q[:, b], 1e-6, 1 - 1e-6)
        if len(np.unique(target)) < 2:
            records.append({"constant": float(target.mean())})
        elif method == "platt":
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(logit(x)[:, None], target)
            records.append(
                {"coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
            )
        else:
            model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, target)
            records.append({"x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()})
    return {"method": method, "boundaries": records}


def apply_calibrator(q, fitted):
    if fitted["method"] == "raw":
        return project(q)
    out = np.empty_like(q, dtype=float)
    for b, params in enumerate(fitted["boundaries"]):
        x = np.clip(q[:, b], 1e-6, 1 - 1e-6)
        if "constant" in params:
            out[:, b] = params["constant"]
        elif fitted["method"] == "platt":
            out[:, b] = expit(params["coefficient"] * logit(x) + params["intercept"])
        else:
            out[:, b] = np.interp(x, params["x"], params["y"])
    return project(out)


def specificity_threshold(y_positive, scores, specificity):
    """Conservative >= threshold rule; fit on training negatives, never heldout labels."""
    if not 0 < specificity <= 1:
        raise ValueError("Specificity must be in (0, 1]")
    negative = np.sort(np.asarray(scores, dtype=float)[~np.asarray(y_positive, bool)])
    if not len(negative):
        return None
    allowed_false_positive = int(np.floor((1 - specificity) * len(negative) + 1e-9))
    return float(np.nextafter(negative[-(allowed_false_positive + 1)], np.inf))


def crossfit_calibration(q, y, patient, folds, method, specificities):
    q, y, patient, folds = map(np.asarray, (q, y, patient, folds))
    for p in np.unique(patient):
        if len(np.unique(folds[patient == p])) != 1:
            raise ValueError("Calibration patient leakage across folds")
    out = np.full_like(q, np.nan, dtype=float)
    thresholds = np.full((len(q), 2, len(specificities)), np.nan, dtype=float)
    records = []
    for fold in np.unique(folds):
        train, held = folds != fold, folds == fold
        if not train.any() or not held.any():
            raise ValueError("Calibration needs nonempty disjoint folds")
        fitted = fit_calibrator(q[train], y[train], method)
        calibrated_train = apply_calibrator(q[train], fitted)
        out[held] = apply_calibrator(q[held], fitted)
        record = {
            "fold": int(fold),
            "training_patients": np.unique(patient[train]).tolist(),
            "heldout_patients": np.unique(patient[held]).tolist(),
            "calibrator": fitted,
            "clinical_thresholds": [],
        }
        for j, grade in enumerate((2, 3)):
            for k, specificity in enumerate(specificities):
                threshold = specificity_threshold(
                    y[train] >= grade, calibrated_train[:, grade - 1], specificity
                )
                thresholds[held, j, k] = np.nan if threshold is None else threshold
                record["clinical_thresholds"].append(
                    {"grade": grade, "target_specificity": specificity, "threshold": threshold}
                )
        records.append(record)
    return out.astype(np.float32), thresholds, records
