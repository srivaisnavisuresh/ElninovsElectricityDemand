
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_excel(
    "data/elnino_electricity_1980_2025_daily.xlsx"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)

# ==========================================
# BASIC FEATURES
# ==========================================

df["Time_Index"] = np.arange(len(df))

df["Year_Num"] = (
    df["Year"]
    - df["Year"].min()
)

df["Month_Sin"] = np.sin(
    2*np.pi*df["Month"]/12
)

df["Month_Cos"] = np.cos(
    2*np.pi*df["Month"]/12
)

df["DayOfYear"] = (
    df["Date"].dt.dayofyear
)

df["Day_Sin"] = np.sin(
    2*np.pi*df["DayOfYear"]/365
)

df["Day_Cos"] = np.cos(
    2*np.pi*df["DayOfYear"]/365
)

# ==========================================
# EL NINO FEATURES
# ==========================================

df["ElNino_Temp"] = (
    df["ElNino_Index"]
    * df["Temperature_C"]
)

df["ElNino_Rain"] = (
    df["ElNino_Index"]
    * df["Rainfall_mm"]
)

df["ElNino_Humidity"] = (
    df["ElNino_Index"]
    * df["Humidity_pct"]
)

df["Temp_Squared"] = (
    df["Temperature_C"] ** 2
)

# ==========================================
# DEMAND HISTORY
# ==========================================

df["Lag1"] = (
    df["Electricity_Demand_MW"]
    .shift(1)
)

df["Lag7"] = (
    df["Electricity_Demand_MW"]
    .shift(7)
)

df["Lag30"] = (
    df["Electricity_Demand_MW"]
    .shift(30)
)

df["Lag90"] = (
    df["Electricity_Demand_MW"]
    .shift(90)
)

df["Rolling7"] = (
    df["Electricity_Demand_MW"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["Rolling30"] = (
    df["Electricity_Demand_MW"]
    .shift(1)
    .rolling(30)
    .mean()
)

df["Rolling90"] = (
    df["Electricity_Demand_MW"]
    .shift(1)
    .rolling(90)
    .mean()
)

# ==========================================
# TARGET = CHANGE IN DEMAND
# ==========================================

df["Demand_Change"] = (

    df["Electricity_Demand_MW"]

    -

    df["Electricity_Demand_MW"]
    .shift(1)

)

# ==========================================
# REMOVE NaN
# ==========================================

df = (
    df.dropna()
      .reset_index(drop=True)
)

# ==========================================
# FEATURES
# ==========================================

features = [

    "ElNino_Index",

    "Temperature_C",
    "Rainfall_mm",
    "Humidity_pct",

    "ElNino_Temp",
    "ElNino_Rain",
    "ElNino_Humidity",

    "Temp_Squared",

    "Month_Sin",
    "Month_Cos",

    "Day_Sin",
    "Day_Cos",

    "Year_Num",
    "Time_Index",

    "Lag1",
    "Lag7",
    "Lag30",
    "Lag90",

    "Rolling7",
    "Rolling30",
    "Rolling90"
]

target = "Demand_Change"

X = df[features]
y = df[target]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split = int(len(df) * 0.85)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ==========================================
# MODEL
# ==========================================

model = XGBRegressor(

    n_estimators=2000,

    learning_rate=0.01,

    max_depth=7,

    min_child_weight=3,

    subsample=0.9,

    colsample_bytree=0.9,

    gamma=0.1,

    objective="reg:squarederror",

    random_state=42
)

# ==========================================
# TRAIN
# ==========================================

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICT CHANGE
# ==========================================

pred_change = model.predict(
    X_test
)

# ==========================================
# CONVERT CHANGE -> DEMAND
# ==========================================

lag1_actual = (
    df.iloc[split:]["Lag1"]
    .values
)

pred_demand = (
    lag1_actual
    + pred_change
)

actual_demand = (
    df.iloc[split:]
    ["Electricity_Demand_MW"]
    .values
)

# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(
    actual_demand,
    pred_demand
)

rmse = np.sqrt(
    mean_squared_error(
        actual_demand,
        pred_demand
    )
)

r2 = r2_score(
    actual_demand,
    pred_demand
)

print("\n===== RESULTS =====")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
    model.feature_importances_

})

importance = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\n===== FEATURE IMPORTANCE =====")

print(
    importance.head(20)
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "electricity_demand_change_model.pkl"
)

print(
    "\nModel saved successfully."
)

# ==========================================
# SAMPLE RESULTS
# ==========================================

results = pd.DataFrame({

    "Actual_Demand":
    actual_demand,

    "Predicted_Demand":
    pred_demand

})

print(
    "\n===== SAMPLE PREDICTIONS ====="
)

print(
    results.head(20)
)