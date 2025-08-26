

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


df = pd.read_csv("PATH/TO/usa_house_price.csv")

if not np.all([np.issubdtype(dt, np.number) for dt in df.dtypes]):
    df = pd.get_dummies(df, drop_first=True)

assert 'price' in df.columns, "Put your target column name in place of 'price'."

X = df.drop(columns=['price']).to_numpy(dtype=float)
y = df['price'].to_numpy(dtype=float).reshape(-1, 1)

def normal_eq_beta(Xb, yb, eps=1e-8):
    Xtil = np.c_[np.ones((Xb.shape[0], 1)), Xb]
    A = Xtil.T @ Xtil
    A_reg = A + eps * np.eye(A.shape[0])
    beta = np.linalg.solve(A_reg, Xtil.T @ yb)
    return beta  

def predict_with_beta(Xb, beta):
    Xtil = np.c_[np.ones((Xb.shape[0], 1)), Xb]
    return Xtil @ beta

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_beta, best_r2 = None, -np.inf

fold_id = 1
for tr_idx, te_idx in kf.split(X):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    beta = normal_eq_beta(Xtr_s, ytr)
    yhat = predict_with_beta(Xte_s, beta)
    r2 = r2_score(yte, yhat)
    print(f"Fold {fold_id}: R2 = {r2:.4f}")
    fold_id += 1

    if r2 > best_r2:
        best_r2, best_beta, best_scaler = r2, beta, scaler

print(f"Best fold R2: {best_r2:.4f}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=7, shuffle=True)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
beta_7030 = normal_eq_beta(Xtr_s, ytr)
yhat_7030 = predict_with_beta(Xte_s, beta_7030)
print(f"70/30 split R2 (Normal Equation): {r2_score(yte, yhat_7030):.4f}")


df = pd.read_csv("PATH/TO/usa_house_price.csv")
if not np.all([np.issubdtype(dt, np.number) for dt in df.dtypes]):
    df = pd.get_dummies(df, drop_first=True)
assert 'price' in df.columns

X = df.drop(columns=['price']).to_numpy(dtype=float)
y = df['price'].to_numpy(dtype=float).reshape(-1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42, shuffle=True)

scaler = StandardScaler().fit(X_train)
Xtr, Xv, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

def gd_fit(Xb, yb, lr=0.01, iters=1000):
    m, d = Xb.shape
    Xtil = np.c_[np.ones((m, 1)), Xb]
    beta = np.zeros((d + 1, 1))
    for t in range(iters):
        yhat = Xtil @ beta
        grad = (2/m) * (Xtil.T @ (yhat - yb))
        beta -= lr * grad
    return beta

def gd_predict(Xb, beta):
    Xtil = np.c_[np.ones((Xb.shape[0], 1)), Xb]
    return Xtil @ beta

alphas = [0.001, 0.01, 0.1, 1.0]
results = []
best = (-np.inf, None)

for a in alphas:
    beta = gd_fit(Xtr, y_train, lr=a, iters=1000)
    yv_hat = gd_predict(Xv, beta)
    yt_hat = gd_predict(Xte, beta)
    r2_val = r2_score(y_val, yv_hat)
    r2_test = r2_score(y_test, yt_hat)
    results.append((a, r2_val, r2_test))
    print(f"alpha={a:<5} -> R2_val={r2_val:.4f}, R2_test={r2_test:.4f}")
    if r2_val > best[0]:
        best = (r2_val, (a, beta))

best_alpha, best_beta = best[1]
print(f"Best learning rate by validation R²: {best_alpha}")
# ====== Q3: Auto Imports 1985 — preprocessing and Multiple Linear Regression ======
# Steps implemented:
# 1) Load specified columns; replace '?' with NaN.
# 2) Impute: numeric -> median; categorical -> mode. Drop rows with NaN price.
# 3) Convert 'num_doors' & 'num_cylinders' (words -> numbers).
# 4) One-hot encode: 'body_style', 'drive_wheels'.
# 5) Label-encode: 'make', 'aspiration', 'engine_location', 'fuel_type'.
#    (engine_location 'rear'->1 else 0 as a convenience binary feature too)
# 6) Standardize numeric inputs.
# 7) PCA on inputs (keep 95% variance).
# 8) Train/test split (70/30), fit LinearRegression, report metrics & plot.


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

cols = ["symboling","normalized_losses","make","fuel_type","aspiration",
        "num_doors","body_style","drive_wheels","engine_location",
        "wheel_base","length","width","height","curb_weight",
        "engine_type","num_cylinders","engine_size","fuel_system",
        "bore","stroke","compression_ratio","horsepower","peak_rpm",
        "city_mpg","highway_mpg","price"]


auto = pd.read_csv("PATH/TO/imports-85.data", names=cols, na_values=['?'])
df = auto.copy()

df = df.dropna(subset=['price']).reset_index(drop=True)

num_like = ["symboling","normalized_losses","wheel_base","length","width","height",
            "curb_weight","engine_size","bore","stroke","compression_ratio",
            "horsepower","peak_rpm","city_mpg","highway_mpg","price"]
for c in num_like:
    df[c] = pd.to_numeric(df[c], errors='coerce')

from sklearn.impute import SimpleImputer
num_cols = [c for c in df.columns if df[c].dtype.kind in "fc" and c != "price"]
cat_cols = [c for c in df.columns if c not in num_cols + ["price"]]

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

door_map = {"two":2, "four":4}
cyl_map = {"two":2,"three":3,"four":4,"five":5,"six":6,"eight":8,"twelve":12}
df["num_doors"] = df["num_doors"].map(door_map).astype(int)
df["num_cylinders"] = df["num_cylinders"].map(cyl_map).astype(int)

one_hot_cols = ["body_style","drive_wheels"]

label_cols = ["make","aspiration","engine_location","fuel_type","engine_type","fuel_system"]
for c in label_cols:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

y = df["price"].astype(float).to_numpy()
X = df.drop(columns=["price"])

pre = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first", sparse_output=False), one_hot_cols),
    ],
    remainder="passthrough"
)

pipe = Pipeline(steps=[
    ("pre", pre),
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full")),
    ("linreg", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123, shuffle=True)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

lin = pipe.named_steps["linreg"]
print("Intercept (after PCA):", lin.intercept_)
print("Number of coefficients (PCA comps):", lin.coef_.shape[0])

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Auto Imports 1985: Predicted vs Actual")
plt.show()


