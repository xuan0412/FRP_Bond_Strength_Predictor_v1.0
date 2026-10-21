"""Train the 9 FRP bond-strength regressors locally under Python 3.11.

The packaged dataset was reconstructed from Appendix A of the supplied thesis.
The thesis text reports 193 valid samples after outlier removal, while 195 rows
can be recovered from Appendix A. This package deliberately trains on the 195
recoverable experimental rows until the final cleaned 193-row CSV is supplied.

Expected columns:
Surface,Rh_d,Rs_d,d,c_d,lb_d,Ar,as,Rr,Type,Ef,ffu,fc,tau
"""
from pathlib import Path
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import catboost
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frp_dataset_appendixA_195.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"This training script requires Python 3.11.x; current={sys.version.split()[0]}")

FEATURES = ["Surface","Rh_d","Rs_d","d","c_d","lb_d","Ar","as","Rr","Type","Ef","ffu","fc"]

df = pd.read_csv(DATA)
missing = [c for c in FEATURES + ["tau"] if c not in df.columns]
if missing:
    raise SystemExit("Dataset is missing columns: " + ", ".join(missing))

X = df[FEATURES].copy()
y = df["tau"].astype(float).to_numpy()
Xn = X.astype(float)

# CatBoost keeps Surface and Type as true categorical variables.
Xc = X.copy()
Xc["Surface"] = Xc["Surface"].astype(int).astype(str)
Xc["Type"] = Xc["Type"].astype(int).astype(str)

print("[TRAIN 1/9] CatBoost")
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=12,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1
)
cat_model.fit(Xc, y, cat_features=["Surface","Type"])
cat_model.save_model(str(OUT / "catboost.cbm"))

models = [
    ("XGBoost", "xgboost.joblib", XGBRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=6,
        subsample=1.0, colsample_bytree=0.5,
        random_state=42, objective="reg:squarederror", n_jobs=-1
    )),
    ("GBDT", "gbdt.joblib", GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=5,
        subsample=0.5, random_state=42
    )),
    ("Random Forest", "random_forest.joblib", RandomForestRegressor(
        n_estimators=50, max_depth=13, min_samples_leaf=1,
        max_features=1.0, random_state=42, n_jobs=-1
    )),
    ("BPNN", "bpnn.joblib", Pipeline([
        ("scale", MinMaxScaler(feature_range=(-1,1))),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64,32), activation="tanh",
            solver="adam", learning_rate_init=0.003,
            alpha=0.0001, max_iter=3000, early_stopping=False,
            random_state=42
        ))
    ])),
    ("SVR", "svr.joblib", Pipeline([
        ("scale", StandardScaler()),
        ("model", SVR(C=450, gamma="scale", epsilon=0.5, kernel="rbf"))
    ])),
    ("AdaBoost", "adaboost.joblib", AdaBoostRegressor(
        n_estimators=100, learning_rate=0.05, random_state=42
    )),
    ("MLR", "mlr.joblib", Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ])),
    ("LASSO", "lasso.joblib", Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=10000, tol=0.0001))
    ])),
]

for i, (name, filename, model) in enumerate(models, start=2):
    print(f"[TRAIN {i}/9] {name}")
    model.fit(Xn, y)
    joblib.dump(model, OUT / filename, compress=3)

meta = {
    "app_version": "1.0.0",
    "python": platform.python_version(),
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "rows": int(len(df)),
    "features": FEATURES,
    "dataset": DATA.name,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "catboost": catboost.__version__,
    "joblib": joblib.__version__,
    "models": ["CatBoost"] + [m[0] for m in models],
    "note": "195 experimental rows reconstructed from Appendix A. Thesis text reports 193 valid samples after outlier removal."
}
(OUT / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[DONE] 9 models trained locally with Python {platform.python_version()} from {len(df)} experimental rows.")
