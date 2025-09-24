# train_xgb_pipeline.py
import pandas as pd
import numpy as np
import joblib, pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ---------------- 1) Load ----------------
df = pd.read_excel("df_final_clean.xlsx")
df.columns = df.columns.str.strip()

# ถ้ายังไม่มีคอลัมน์ปี/เดือน/ฤดูกาล ให้สร้าง (เผื่อไฟล์คุณมีอยู่แล้วจะไม่กระทบ)
if "date" in df.columns:
    df["date"]  = pd.to_datetime(df["date"])
    if "ปี" not in df.columns:
        df["ปี"] = df["date"].dt.year
    if "เดือน" not in df.columns:
        df["เดือน"] = df["date"].dt.month
if "ฤดูกาล" not in df.columns:
    def month_to_season(m: int) -> int:
        if m in [3,4,5]: return 1
        elif m in [6,7,8,9,10]: return 2
        else: return 3
    df["ฤดูกาล"] = df["เดือน"].apply(month_to_season)

# ---------------- 2) Encode หมวดหมู่ ----------------
cat_cols = [c for c in ["อาชีพ","ตำบล","อำเภอ"] if c in df.columns]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ---------------- 3) X / y ----------------
drop_if_exist = ["cases","date","เพศ","อายุ(ปี)","ไตรมาส","humid_15d_avg"]
features = [c for c in df.columns if c not in drop_if_exist]
X = df[features].copy()
y = df["cases"].astype(float)

print("✅ Features used:", features)

# ---------------- 4) Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- 5) Model ----------------
xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# ---------------- 6) Evaluate ----------------
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
print(f"✅ RMSE: {rmse:.2f} | R²: {r2:.3f} | MAE: {mae:.2f}")

# ---------------- 7) Save ----------------
joblib.dump(xgb_model, "xgb_regressor.pkl")
joblib.dump(X.columns.tolist(), "xgb_features.pkl")
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("💾 Saved: xgb_regressor.pkl, xgb_features.pkl, label_encoders.pkl")
