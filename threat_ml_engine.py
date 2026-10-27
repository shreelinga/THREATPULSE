"""
threat_ml_engine.py

Inference module for predicting stock price reaction from cyber threat features:
  - severity_score (1.0 to 10.0)
  - users_affected_millions (>= 0.0)
  - data_sensitivity_score (1.0 to 10.0)

Uses scikit-learn Pipeline with bundled ColumnTransformer (log1p user scaling)
from threat_impact_model.joblib, ensuring exact preprocessing parity with training.
Provides transparent baseline comparisons, honest dispersion intervals, and robust fallbacks.
"""

import os
import json
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np

MODEL_VERSION = "v2.2.0-leakage-safe-pipeline"
TARGET_DEFINITION = "Post-Disclosure Stock Price Change (% from T-1 Close to T+2 Close)"
PREDICTION_HORIZON = "1D-2D Post-Disclosure Window"

MODEL_LIMITATION = (
    "Production model evaluated on n=14 historical incidents using Leave-One-Out Cross-Validation (85.7% legacy directional base rate). "
    "In Phase 5 candidate evaluation on benchmark-adjusted 1-day abnormal returns (N=21), the Historical Median Baseline (LOOCV MAE 3.66%) "
    "outperforms all machine learning models. Random Forest (LOOCV MAE 4.18%) and Huber are experimental "
    "comparison models, not proven superior predictors. Predictions represent exploratory statistical estimates, "
    "not causal certainty or financial advice. Calibrated incident-level prediction intervals are unavailable due to sample size constraints."
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "threat_impact_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

_LOADED_BUNDLE = None
_LOADED_METRICS = None


def load_model_metrics() -> Dict[str, Any]:
    """Loads out-of-sample metrics from model_metrics.json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_model_bundle() -> Optional[Dict[str, Any]]:
    """Loads the serialized model bundle if available, auto-training if missing or outdated."""
    global _LOADED_BUNDLE
    if _LOADED_BUNDLE is not None:
        return _LOADED_BUNDLE

    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("model_version") == MODEL_VERSION:
                _LOADED_BUNDLE = bundle
                return _LOADED_BUNDLE
        except Exception as e:
            print(f"[ML Engine] Notice loading bundle: {e}")

    try:
        from train_model import run_staged_model_evaluation
        print(f"[ML Engine] Auto-training staged model bundle {MODEL_VERSION}...")
        run_staged_model_evaluation()
        import joblib
        _LOADED_BUNDLE = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[ML Engine] Warning: Could not auto-train {MODEL_PATH}: {e}")
        _LOADED_BUNDLE = None
    return _LOADED_BUNDLE


def reload_model():
    """Forces reloading the model bundle and metrics from disk."""
    global _LOADED_BUNDLE, _LOADED_METRICS
    _LOADED_BUNDLE = None
    _LOADED_METRICS = None
    return load_model_bundle()


def validate_ml_inputs(
    severity_score: Any,
    users_affected_millions: Any,
    data_sensitivity_score: Any
) -> Tuple[bool, Optional[str], Optional[Tuple[float, float, float]]]:
    """
    Validates ML feature inputs strictly against defined domains:
      - severity_score: finite float between 1.0 and 10.0 (or 0.0 for zero baseline)
      - users_affected_millions: non-negative finite float
      - data_sensitivity_score: finite float between 1.0 and 10.0 (or 0.0 for zero baseline)
    """
    try:
        if severity_score is None or data_sensitivity_score is None or users_affected_millions is None:
            return False, "Required input parameters cannot be None", None

        sev = float(severity_score)
        users = float(users_affected_millions)
        sens = float(data_sensitivity_score)

        if not (math.isfinite(sev) and math.isfinite(users) and math.isfinite(sens)):
            return False, "Inputs must be finite numbers (no NaN or Inf)", None

        if sev < 1.0 or sev > 10.0:
            return False, f"severity_score must be between 1.0 and 10.0 (got {sev})", None

        if sens < 1.0 or sens > 10.0:
            return False, f"data_sensitivity_score must be between 1.0 and 10.0 (got {sens})", None

        if users < 0.0:
            return False, f"users_affected_millions cannot be negative (got {users})", None

        return True, None, (sev, users, sens)
    except (ValueError, TypeError) as err:
        return False, f"Invalid numerical inputs: {err}", None


def predict_stock_impact(
    severity_score: float,
    users_affected_millions: float,
    data_sensitivity_score: float
) -> Dict[str, Any]:
    """
    Predicts stock price reaction given threat features.
    Features: [severity_score, users_affected_millions, data_sensitivity_score]
    """
    # 1. Strict input validation
    valid, err_msg, cleaned = validate_ml_inputs(
        severity_score, users_affected_millions, data_sensitivity_score
    )
    if not valid:
        return {
            "status": "UNAVAILABLE",
            "error": err_msg,
            "predicted_change_percent": None,
            "predicted_direction": None,
            "expected_range_percent": None,
            "tree_output_spread": None,
            "confidence": None,
            "confidence_basis": "unavailable: invalid input parameters",
            "target_type": "raw_return",
            "target_definition": TARGET_DEFINITION,
            "prediction_horizon": PREDICTION_HORIZON,
            "model_limitation": MODEL_LIMITATION,
            "predicted_impact_class": "UNAVAILABLE",
            "model_used": "None",
            "is_fallback": False,
            "primary_benchmark": None,
            "experimental_model": None,
            "experimental_huber": None,
            "incident_prediction_interval": {
                "status": "UNAVAILABLE",
                "interval": None,
                "reason": "Input parameters invalid"
            }
        }

    sev, users, sens = cleaned

    bundle = load_model_bundle()
    metrics_data = load_model_metrics()

    # 2. Model inference via bundled Pipeline or Regressor
    if bundle is not None:
        try:
            raw_input = np.array([[sev, users, sens]])

            if "pipeline" in bundle and bundle["pipeline"] is not None:
                pipeline = bundle["pipeline"]
                predicted_drop = float(pipeline.predict(raw_input)[0])

                reg = pipeline.named_steps["regressor"]
                X_trans = pipeline.named_steps["preprocessor"].transform(raw_input)
                tree_predictions = [float(tree.predict(X_trans)[0]) for tree in reg.estimators_]
            elif "regressor" in bundle and bundle["regressor"] is not None:
                reg = bundle["regressor"]
                log_users = float(np.log1p(users))
                X_legacy = np.array([[sev, log_users, sens]])
                predicted_drop = float(reg.predict(X_legacy)[0])
                tree_predictions = [float(tree.predict(X_legacy)[0]) for tree in reg.estimators_]
            else:
                raise KeyError("Neither 'pipeline' nor 'regressor' found in model bundle")

            p10 = round(float(np.percentile(tree_predictions, 10)), 2)
            p90 = round(float(np.percentile(tree_predictions, 90)), 2)

            dispersion_basis = (
                f"Tree-output dispersion across {len(tree_predictions)} estimators "
                "(descriptive spread: 10th to 90th percentile, NOT a calibrated confidence interval)"
            )

            expected_range = {
                "low": p10,
                "high": p90,
                "basis": dispersion_basis
            }

            tree_spread = {
                "low": p10,
                "high": p90,
                "description": dispersion_basis,
                "basis": dispersion_basis
            }

            if predicted_drop <= -5.0:
                pred_class = "SEVERE_DROP"
            elif predicted_drop <= -1.5:
                pred_class = "MODERATE_DROP"
            else:
                pred_class = "MILD_DROP"

            feature_importance = bundle.get("feature_importances", {
                "severity_score": 0.515,
                "users_affected_millions": 0.268,
                "data_sensitivity_score": 0.218
            })

            primary_bm = metrics_data.get("primary_benchmark", {
                "model_name": "Historical Median Baseline",
                "target_horizon": "1-Day Post-Disclosure",
                "target_definition": "Benchmark-Adjusted 1-Day Abnormal Return (Stock Return - Benchmark Return)",
                "market_benchmarks": "SPY (US Equities) / ^NSEI (Indian Equities)",
                "eval_sample_count": 21,
                "loocv_mae": 3.66,
                "loocv_mae_ci_95": [2.33, 5.27],
                "loocv_rmse": 5.05,
                "logo_mae": 3.72,
                "logo_rmse": 5.06,
                "directional_accuracy": 0.762,
                "balanced_accuracy": 0.500,
                "median_abnormal_return": -2.00,
                "median_abnormal_return_pct": -2.00,
                "status": "RECOMMENDED_PRIMARY_BENCHMARK",
                "validation_methodology": "Leave-One-Out (21 folds) and Company-Grouped (17 folds)"
            })

            experimental_comp = {
                "name": "RandomForestRegressor (Regularized Pipeline)",
                "role": "EXPERIMENTAL_COMPARISON",
                "predicted_change_percent": round(predicted_drop, 2),
                "loocv_mae_pct": 4.18,
                "logo_cv_mae_pct": 4.25,
                "status": "NOT_PROVEN_SUPERIOR",
                "note": "Experimental comparison model; not proven superior to Historical Median baseline."
            }

            # Experimental Huber Robust Regressor (derived from audited N=21 benchmark-adjusted dataset)
            log_u = float(np.log1p(users))
            z_sev = (sev - 8.4047619) / 0.89885822
            z_users = (log_u - 2.76611384) / 2.27899144
            z_sens = (sens - 8.4047619) / 1.04944983
            huber_raw = -2.85057723 + 0.46980910 * z_sev - 1.28906872 * z_users - 1.17160709 * z_sens
            huber_drop = round(float(np.clip(huber_raw, -25.0, 5.0)), 2)

            experimental_huber = {
                "name": "HuberRegressor (Robust Linear)",
                "model_name": "Huber Robust Regression",
                "role": "EXPERIMENTAL_COMPARISON",
                "predicted_change_percent": huber_drop,
                "loocv_mae": 4.65,
                "loocv_mae_pct": 4.65,
                "logo_mae": 4.69,
                "logo_cv_mae_pct": 4.86,
                "loocv_rmse_pct": 6.27,
                "directional_accuracy_pct": 76.2,
                "balanced_accuracy_pct": 56.9,
                "status": "NOT_PROVEN_SUPERIOR",
                "formula": "AR = -2.85 + 0.47*z(sev) - 1.29*z(log_users) - 1.17*z(sens)",
                "note": "Experimental outlier-robust linear model; highest balanced accuracy (56.9%) but higher LOOCV MAE (4.65%) than Historical Median."
            }

            incident_pred_interval = {
                "status": "UNAVAILABLE",
                "interval": None,
                "reason": "Defensible incident-level prediction intervals cannot be estimated from the available data (N=21 incidents).",
                "note": "The current dataset contains only 21 incidents; out-of-sample performance and individual incident impact remain highly uncertain."
            }

            eval_meta = {
                "sample_size": 21,
                "validation_methods": ["LOOCV (Leave-One-Out)", "LOGO-CV (Leave-One-Group-Out by Company)"],
                "production_model_target": "raw_return (stock only from T-1 Close to T+2 Close)",
                "target_horizon": "1-Day Post-Disclosure",
                "target_definition": "Benchmark-Adjusted 1-Day Abnormal Return (AR = R_stock - R_benchmark)",
                "market_benchmarks": ["S&P 500 (^GSPC)", "Nifty 50 (^NSEI)"],
                "confidence_interval_distinction": "Reported bootstrap confidence intervals apply to sample average MAE, NOT incident-level prediction intervals.",
                "sample_size_warning": "Small evaluation sample size (N=21 incidents across 17 companies). Performance on future incidents is uncertain."
            }

            return {
                "status": "SUCCESS",
                "model_version": bundle.get("model_version", MODEL_VERSION),
                "target_type": "raw_return",
                "target_definition": bundle.get("target_definition", TARGET_DEFINITION),
                "prediction_horizon": bundle.get("prediction_horizon", PREDICTION_HORIZON),
                "predicted_change_percent": round(predicted_drop, 2),
                "raw_return_predicted": round(predicted_drop, 2),
                "predicted_direction": "UP" if predicted_drop > 0 else "DOWN" if predicted_drop < 0 else "FLAT",
                "expected_range_percent": expected_range,
                "tree_output_spread": tree_spread,
                "confidence": None,
                "confidence_basis": "unavailable: small sample (n=14) prohibits calibrated confidence intervals",
                "model_limitation": bundle.get("limitations", MODEL_LIMITATION),
                "predicted_impact_class": pred_class,
                "model_used": "RandomForestRegressor (Regularized Pipeline, LOOCV, log1p user scaling)",
                "is_fallback": False,
                "feature_importance": feature_importance,
                "out_of_sample_comparison": metrics_data.get("staged_comparison", {}),
                "primary_benchmark": primary_bm,
                "experimental_model": experimental_comp,
                "experimental_huber": experimental_huber,
                "incident_prediction_interval": incident_pred_interval,
                "evaluation_metadata": eval_meta,
                "candidate_evaluation_abnormal_1d": metrics_data.get("candidate_evaluation_abnormal_1d", {})
            }
        except Exception as e:
            print(f"[ML Engine] Prediction error: {e}, using heuristic fallback")

    # 3. Transparent heuristic fallback when trained bundle is missing or fails
    log_users_fallback = float(np.log1p(users))
    base_drop = -0.35 * sev - 0.25 * sens - 0.3 * log_users_fallback
    heuristic_drop = round(max(-18.0, min(1.0, base_drop)), 2)

    if heuristic_drop <= -5.0:
        h_class = "SEVERE_DROP"
    elif heuristic_drop <= -1.5:
        h_class = "MODERATE_DROP"
    else:
        h_class = "MILD_DROP"

    return {
        "status": "FALLBACK",
        "model_version": "v1.0.0-heuristic",
        "target_type": "raw_return",
        "target_definition": "Heuristic proxy for expected price reaction (%)",
        "prediction_horizon": "1D-2D Reaction (Heuristic Proxy)",
        "predicted_change_percent": heuristic_drop,
        "predicted_direction": "UP" if heuristic_drop > 0 else "DOWN" if heuristic_drop < 0 else "FLAT",
        "expected_range_percent": {
            "low": None,
            "high": None,
            "basis": "unavailable: heuristic fallback has no statistical distribution"
        },
        "tree_output_spread": None,
        "confidence": None,
        "confidence_basis": "unavailable for heuristic fallback",
        "model_limitation": MODEL_LIMITATION,
        "predicted_impact_class": h_class,
        "model_used": "Deterministic Rule Heuristic (ML bundle offline)",
        "is_fallback": True,
        "feature_importance": {
            "severity_score": 0.45,
            "data_sensitivity_score": 0.35,
            "users_affected_millions": 0.20
        },
        "out_of_sample_comparison": {},
        "primary_benchmark": metrics_data.get("primary_benchmark", None),
        "experimental_model": None,
        "experimental_huber": None,
        "incident_prediction_interval": {
            "status": "UNAVAILABLE",
            "interval": None,
            "reason": "ML bundle offline; heuristic fallback active"
        },
        "evaluation_metadata": {
            "sample_size": 21,
            "validation_methods": ["LOOCV", "LOGO-CV"]
        }
    }


if __name__ == "__main__":
    result = predict_stock_impact(severity_score=9.0, users_affected_millions=54.3, data_sensitivity_score=9.0)
    print("Test prediction (T-Mobile breach profile):", json.dumps(result, indent=2))
    result_small = predict_stock_impact(severity_score=5.0, users_affected_millions=0.01, data_sensitivity_score=4.0)
    print("Test prediction (Low severity profile):", json.dumps(result_small, indent=2))
