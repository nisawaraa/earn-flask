# train_xgb_forecast.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ------------------ โหลดข้อมูล ------------------
df = pd.read_excel("df_final_clean.xlsx")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# สร้างซีรีส์รายเดือน
monthly = (
    df.groupby(["year", "month"], as_index=False)["cases"]
      .sum()
      .sort_values(["year", "month"])
      .reset_index(drop=True)
)

# ------------------ ฟีเจอร์ lag/rolling ------------------
def add_features(data):
    data = data.copy()
    data["lag1"] = data["cases"].shift(1)
    data["lag2"] = data["cases"].shift(2)
    data["lag3"] = data["cases"].shift(3)
    data["roll3"] = data["cases"].shift(1).rolling(3).mean()
    data["roll6"] = data["cases"].shift(1).rolling(6).mean()
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return data

monthly = add_features(monthly).dropna()

# ------------------ แบ่ง train/test ------------------
X = monthly[["lag1", "lag2", "lag3", "roll3", "roll6", "month", "year", "month_sin", "month_cos"]]
y = monthly["cases"]

train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# ------------------ เทรน XGBoost ------------------
final_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

final_model.fit(X_train, y_train)

# ------------------ ประเมินผล ------------------
y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5   # ✅ คำนวณเองแทน squared=False
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.3f}")

# ------------------ บันทึกโมเดล ------------------
joblib.dump(final_model, "xgb_forecast_model.pkl")
print("✅ Saved new xgb_forecast_model.pkl")
