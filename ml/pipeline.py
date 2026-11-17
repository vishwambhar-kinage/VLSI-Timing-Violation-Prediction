"""Train and serve VLSI timing-violation classifiers from the project's real CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vlsi_timing_dataset.csv"

CATEGORICAL = ["design_type", "path_type"]
NUMERIC = [
    "logic_depth", "cell_count", "fanout", "cell_delay_ns", "net_delay_ns",
    "clock_period_ns", "input_slew_ns", "output_load_pf", "setup_time_ns"
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "timing_violation"
REQUIRED = FEATURES + [TARGET]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Add data/vlsi_timing_dataset.csv before starting the app."
    )

DF = pd.read_csv(DATA_PATH)
missing = [c for c in REQUIRED if c not in DF.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
if DF[TARGET].isna().any():
    raise ValueError("Target column timing_violation contains missing values.")
labels = set(pd.to_numeric(DF[TARGET], errors="coerce").dropna().astype(int).unique())
if not labels.issubset({0, 1}) or len(labels) != 2:
    raise ValueError("timing_violation must contain both binary labels 0 and 1.")
DF[TARGET] = DF[TARGET].astype(int)

X = DF[FEATURES].copy()
Y = DF[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=Y
)

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])

MODEL_DEFINITIONS = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=180, max_depth=14, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.06, max_depth=2, random_state=42
    )
}

MODELS = {}
METRICS = {}
for name, estimator in MODEL_DEFINITIONS.items():
    model = Pipeline([
        ("preprocess", make_preprocessor()),
        ("classifier", estimator)
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    MODELS[name] = model
    METRICS[name] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    }

BEST_NAME = max(METRICS, key=lambda model_name: METRICS[model_name]["f1"])
BEST_MODEL = MODELS[BEST_NAME]
TEST_SIZE = int(len(y_test))

def get_dashboard_data():
    counts = DF[TARGET].value_counts()
    return {
        "samples": int(len(DF)),
        "features": int(len(REQUIRED) - 1),
        "columns": int(len(REQUIRED)),
        "classes": 2,
        "models": len(MODELS),
        "violations": int(counts.get(1, 0)),
        "clean": int(counts.get(0, 0)),
        "data_source": "Uploaded project dataset: data/vlsi_timing_dataset.csv",
        "best_model": BEST_NAME,
        "best_f1": METRICS[BEST_NAME]["f1"],
        "design_types": sorted(DF["design_type"].dropna().astype(str).unique().tolist()),
        "path_types": sorted(DF["path_type"].dropna().astype(str).unique().tolist())
    }

def get_model_comparison():
    return {
        "models": [
            {"name": name, "accuracy": values["accuracy"], "precision": values["precision"],
             "recall": values["recall"], "f1": values["f1"]}
            for name, values in METRICS.items()
        ],
        "best_model": BEST_NAME,
        "confusion_matrix": METRICS[BEST_NAME]["matrix"],
        "test_samples": TEST_SIZE,
        "data_source": "Uploaded project dataset"
    }

def get_analysis_data():
    counts = DF[TARGET].value_counts()
    associations = (
        DF[NUMERIC + [TARGET]].corr(numeric_only=True)[TARGET]
        .drop(TARGET).abs().sort_values(ascending=False)
    )
    return {
        "class_distribution": {
            "no_violation": int(counts.get(0, 0)),
            "violation": int(counts.get(1, 0))
        },
        "feature_association": [
            {"feature": feature, "value": round(float(value), 4)}
            for feature, value in associations.items()
        ],
        "data_source": "Uploaded project dataset"
    }

def predict_path(payload):
    values = {}
    for col in NUMERIC:
        if col not in payload:
            raise ValueError(f"Missing required field: {col}")
        try:
            values[col] = float(payload[col])
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be numeric")
        if not np.isfinite(values[col]):
            raise ValueError(f"{col} must be a finite number")
    for col in CATEGORICAL:
        value = str(payload.get(col, "")).strip()
        allowed = set(DF[col].dropna().astype(str).unique())
        if value not in allowed:
            raise ValueError(f"Choose a valid {col}: {', '.join(sorted(allowed))}")
        values[col] = value

    for col in ["logic_depth", "cell_count", "fanout"]:
        if values[col] < 0:
            raise ValueError(f"{col} cannot be negative")

    row = pd.DataFrame([{col: values[col] for col in FEATURES}], columns=FEATURES)
    prediction = int(BEST_MODEL.predict(row)[0])
    probabilities = BEST_MODEL.predict_proba(row)[0]
    confidence = float(np.max(probabilities)) * 100
    return {
        "prediction": prediction,
        "label": "Timing Violation" if prediction == 1 else "No Timing Violation",
        "confidence": round(confidence, 1),
        "model": BEST_NAME,
        "note": "ML screening estimate; confirm critical paths using sign-off Static Timing Analysis."
    }
