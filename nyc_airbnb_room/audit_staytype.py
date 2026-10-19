from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "nyc.csv"
FEATURES_PATH = ROOT / "nyc_features.py"
TRAIN_PATH = ROOT / "train_model.py"
APP_PATH = ROOT / "app.py"
MODEL_PATH = ROOT / "models" / "Best_Model_Pipeline.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"

# Keep project module imports consistent with the saved pipeline.
sys.path.insert(0, str(ROOT))

results = []
findings = []

def report(section, status, detail):
    results.append({
        "section": section,
        "status": status,
        "detail": str(detail),
    })
    print(f"[{status}] {section}: {detail}")

def finding(severity, title, detail):
    findings.append({
        "severity": severity,
        "title": title,
        "detail": detail,
    })
    print(f"  {severity}: {title} — {detail}")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def source_audit(path, label):
    if not path.exists():
        report(label, "MISSING", f"File not found: {path}")
        return None

    source = path.read_text(encoding="utf-8", errors="replace")
    report(label, "OK", f"{len(source.splitlines())} source lines")

    try:
        tree = ast.parse(source)
        classes = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        functions = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        report(
            f"{label} syntax",
            "OK",
            f"AST parsed; classes={classes}; functions={functions}",
        )
    except SyntaxError as exc:
        report(label + " syntax", "FAIL", str(exc))
        finding("HIGH", f"{label} has invalid Python syntax", str(exc))

    return source

def scan_patterns(source, label):
    if source is None:
        return

    patterns = {
        "target referenced in feature engineering": (
            r"room_type|target|y_train|y_test"
        ),
        "global aggregation / fit-before-split clues": (
            r"\.fit_transform\s*\(|groupby\s*\(|"
            r"train_test_split|StratifiedKFold|cross_val_score|"
            r"GridSearchCV|RandomizedSearchCV"
        ),
        "possible target leakage / target-derived feature clues": (
            r"target.?encoding|target_encoder|"
            r"room_type.*map|map.*room_type|"
            r"Churn|label|leakage"
        ),
        "identifier and time columns": (
            r"\bid\b|host_id|host_name|last_review|"
            r"drop\s*\(|dropna|fillna"
        ),
        "prediction and calibration logic": (
            r"predict_proba|predict\s*\(|CalibratedClassifierCV|"
            r"class_weight|sample_weight|decision|"
            r"threshold|weights"
        ),
    }

    lines = source.splitlines()
    for title, pattern in patterns.items():
        matches = []
        regex = re.compile(pattern, re.IGNORECASE)
        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{i}: {line.strip()[:180]}")
        if matches:
            print(f"\n--- {label}: {title} ---")
            for item in matches[:18]:
                print(item)
            if len(matches) > 18:
                print(f"... {len(matches) - 18} additional matches")

def walk_pipeline(obj, depth=0, seen=None):
    if seen is None:
        seen = set()
    if obj is None or id(obj) in seen or depth > 6:
        return
    seen.add(id(obj))

    indent = "  " * depth
    print(f"{indent}- {type(obj).__module__}.{type(obj).__name__}")

    if hasattr(obj, "named_steps"):
        for name, step in obj.named_steps.items():
            print(f"{indent}  step[{name!r}]:")
            walk_pipeline(step, depth + 2, seen)

    if hasattr(obj, "calibrated_classifiers_"):
        for i, item in enumerate(obj.calibrated_classifiers_):
            estimator = getattr(item, "estimator", None)
            if estimator is None:
                estimator = getattr(item, "base_estimator", None)
            print(f"{indent}  calibration fold {i}:")
            walk_pipeline(estimator, depth + 2, seen)

    if hasattr(obj, "estimator_"):
        nested = getattr(obj, "estimator_", None)
        if nested is not None and nested is not obj:
            print(f"{indent}  fitted estimator_:")
            walk_pipeline(nested, depth + 1, seen)

print("=" * 72)
print("STAYTYPE AI — READ-ONLY AUDIT")
print("=" * 72)
print("Project:", ROOT)
print("Python:", sys.version.split()[0])
print("scikit-learn:", sklearn.__version__)

# 1. Check required files.
required = [
    DATA_PATH, FEATURES_PATH, TRAIN_PATH, APP_PATH, MODEL_PATH
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    for path in missing:
        report("Required file", "FAIL", path)
    raise SystemExit(
        "Audit stopped because required files are missing. "
        "No production files were changed."
    )

model_hash_before = sha256(MODEL_PATH)
report("Production model SHA-256 before", "OK", model_hash_before)

# 2. Inspect source code without executing training code.
features_source = source_audit(FEATURES_PATH, "nyc_features.py")
train_source = source_audit(TRAIN_PATH, "train_model.py")
app_source = source_audit(APP_PATH, "app.py")

scan_patterns(features_source, "nyc_features.py")
scan_patterns(train_source, "train_model.py")
scan_patterns(app_source, "app.py")

# 3. Load the dataset and inspect schema/target.
df = pd.read_csv(DATA_PATH)
report("Dataset shape", "OK", f"{df.shape[0]} rows x {df.shape[1]} columns")
report("Duplicate rows", "INFO", int(df.duplicated().sum()))
report("Missing values", "INFO", int(df.isna().sum().sum()))

target = "room_type"
if target not in df.columns:
    report("Target column", "FAIL", f"'{target}' not found")
    raise SystemExit("Cannot continue without the expected target column.")

report("Target distribution", "INFO", df[target].value_counts(dropna=False).to_dict())

identifier_candidates = [
    c for c in ["id", "name", "host_id", "host_name", "last_review"]
    if c in df.columns
]
report("Potential identifier/date columns", "INFO", identifier_candidates)

# This audit intentionally does not split, tune, or fit any estimator.
# Drop target and obvious record ID, matching the prior SHAP smoke-test input.
X = df.drop(columns=[target]).copy()
for col in list(X.columns):
    if col.strip().lower() in {"id", "listing_id"}:
        X.drop(columns=[col], inplace=True)

# 4. Load saved production pipeline.
print("\n--- SAVED MODEL STRUCTURE ---")
model = joblib.load(MODEL_PATH)
report("Model load", "OK", type(model).__name__)
report("Model classes", "INFO", getattr(model, "classes_", "not exposed"))
report("Model fitted status", "INFO", hasattr(model, "estimator_"))
walk_pipeline(model)

# Inspect the first fitted calibration fold where available.
fitted_estimator = getattr(model, "estimator_", None)
folds = getattr(fitted_estimator, "calibrated_classifiers_", [])
if folds:
    fold = folds[0]
    pipeline = getattr(fold, "estimator", None)
    if pipeline is None:
        pipeline = getattr(fold, "base_estimator", None)

    if pipeline is not None and hasattr(pipeline, "named_steps"):
        report("Fitted pipeline steps", "OK", list(pipeline.named_steps))
        feature_step = pipeline.named_steps.get("features")
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier")

        if feature_step is not None:
            report(
                "Feature engineering class",
                "INFO",
                type(feature_step).__name__,
            )
            for attr in [
                "n_features_in_", "feature_names_in_",
                "clip_caps_", "neighbourhood_medians_",
            ]:
                if hasattr(feature_step, attr):
                    value = getattr(feature_step, attr)
                    if attr == "feature_names_in_":
                        value = list(value)
                    report(f"FeatureEngineer.{attr}", "INFO", str(value)[:500])

        if preprocessor is not None:
            try:
                report(
                    "Preprocessor output feature count",
                    "OK",
                    len(preprocessor.get_feature_names_out()),
                )
            except Exception as exc:
                report("Preprocessor feature names", "WARN", exc)

        if classifier is not None:
            report("Base classifier", "INFO", type(classifier).__name__)
            for attr in [
                "n_estimators", "max_depth", "min_samples_split",
                "min_samples_leaf", "max_features", "class_weight",
            ]:
                if hasattr(classifier, attr):
                    report(
                        f"Classifier.{attr}",
                        "INFO",
                        getattr(classifier, attr),
                    )
    else:
        report("Fitted pipeline", "WARN", "No pipeline found in first calibration fold")
else:
    report(
        "Calibration folds",
        "WARN",
        "No calibrated_classifiers_ found; model structure needs manual review.",
    )

# 5. Inference smoke test. Predict only a few existing feature rows.
# No model fitting or artifact writing occurs here.
print("\n--- INFERENCE SMOKE TEST ---")
sample = X.head(min(5, len(X))).copy()

try:
    predictions_1 = model.predict(sample)
    predictions_2 = model.predict(sample)

    report("predict()", "OK", f"Returned {len(predictions_1)} predictions")
    report("Prediction classes", "INFO", list(map(str, predictions_1)))
    report(
        "Repeatability",
        "OK" if np.array_equal(predictions_1, predictions_2) else "FAIL",
        "Repeated predictions are identical"
        if np.array_equal(predictions_1, predictions_2)
        else "Repeated predictions differ",
    )

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(sample))
        report("predict_proba shape", "INFO", probabilities.shape)

        if probabilities.ndim != 2:
            finding("HIGH", "Unexpected probability dimensions", str(probabilities.shape))
        elif not np.isfinite(probabilities).all():
            finding("HIGH", "Non-finite probabilities", "NaN or infinity found")
        elif (probabilities < -1e-8).any() or (probabilities > 1 + 1e-8).any():
            finding("HIGH", "Probabilities outside [0, 1]", "Inspect wrapper and calibration")
        else:
            report("Probability bounds", "OK", "All values fall within [0, 1]")

        if probabilities.ndim == 2 and probabilities.shape[1] == len(
            getattr(model, "classes_", [])
        ):
            row_sums = probabilities.sum(axis=1)
            report("Probability row sums", "INFO", np.round(row_sums, 6).tolist())
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                finding(
                    "MEDIUM",
                    "predict_proba rows do not sum to 1",
                    "Check the model wrapper's predict_proba implementation.",
                )
    else:
        report("predict_proba()", "WARN", "Method not available")

except Exception as exc:
    report("Inference smoke test", "FAIL", f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    finding(
        "HIGH",
        "Inference smoke test failed",
        "Inspect the complete traceback and confirm the app uses the expected input schema.",
    )

# 6. Metrics file is read only.
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        report("Saved metrics file", "OK", str(METRICS_PATH))
        for key in [
            "dataset_rows", "dataset_columns",
            "training_samples", "testing_samples",
            "test_accuracy", "test_macro_f1",
            "per_class_f1", "decision_weights", "model_path",
        ]:
            if key in metrics:
                report(f"Metric: {key}", "INFO", metrics[key])
    except Exception as exc:
        report("Saved metrics file", "WARN", exc)
else:
    report("Saved metrics file", "WARN", "metrics.json not found")

# 7. Check that the model artifact was not modified.
model_hash_after = sha256(MODEL_PATH)
report("Production model SHA-256 after", "OK", model_hash_after)
unchanged = model_hash_before == model_hash_after
report(
    "Production model unchanged",
    "PASS" if unchanged else "FAIL",
    "SHA-256 matches before and after audit"
    if unchanged else "SHA-256 changed during audit; investigate immediately",
)

# Final summary. Findings are clues, not automatic proof of leakage.
print("\n" + "=" * 72)
print("AUDIT SUMMARY")
print("=" * 72)
print("Checks recorded:", len(results))
print("Findings:", len(findings))
print("Model unchanged:", unchanged)

if not findings:
    print("No automatic findings were raised. This is not proof of zero leakage.")

print("\nSeverity counts:")
for severity in ["HIGH", "MEDIUM", "LOW"]:
    count = sum(f["severity"] == severity for f in findings)
    print(f"  {severity}: {count}")

print("\nReview the source-pattern matches above manually.")
print("This script did not retrain, tune, or overwrite the production model.")
