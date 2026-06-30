import io
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file

app = Flask(__name__)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel("data/elnino_electricity_1980_2025_daily.xlsx")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

model = joblib.load("electricity_demand_change_model.pkl")

FEATURES = [
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
    "Rolling90",
]


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(data):
    """Engineer time, seasonal, El Niño, and lag/rolling features."""
    data = data.copy()

    # Time-based features
    data["Time_Index"] = np.arange(len(data))
    data["Year_Num"] = data["Year"] - data["Year"].min()
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)

    data["DayOfYear"] = data["Date"].dt.dayofyear
    data["Day_Sin"] = np.sin(2 * np.pi * data["DayOfYear"] / 365)
    data["Day_Cos"] = np.cos(2 * np.pi * data["DayOfYear"] / 365)

    # El Niño interaction features
    data["ElNino_Temp"] = data["ElNino_Index"] * data["Temperature_C"]
    data["ElNino_Rain"] = data["ElNino_Index"] * data["Rainfall_mm"]
    data["ElNino_Humidity"] = data["ElNino_Index"] * data["Humidity_pct"]
    data["Temp_Squared"] = data["Temperature_C"] ** 2

    # Lag features
    data["Lag1"] = data["Electricity_Demand_MW"].shift(1)
    data["Lag7"] = data["Electricity_Demand_MW"].shift(7)
    data["Lag30"] = data["Electricity_Demand_MW"].shift(30)
    data["Lag90"] = data["Electricity_Demand_MW"].shift(90)

    # Rolling averages (based on prior-day values only)
    shifted = data["Electricity_Demand_MW"].shift(1)
    data["Rolling7"] = shifted.rolling(7).mean()
    data["Rolling30"] = shifted.rolling(30).mean()
    data["Rolling90"] = shifted.rolling(90).mean()

    return data


df = create_features(df)
df = df.dropna().reset_index(drop=True)  # drop rows with NaN lag values


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_90_days():

    history = (
        df["Electricity_Demand_MW"]
        .tolist()
    )

    forecasts = []

    last_date = df["Date"].iloc[-1]

    for i in range(90):

        next_date = last_date + timedelta(days=i + 1)

        lag1 = history[-1]
        lag7 = history[-7]
        lag30 = history[-30]
        lag90 = history[-90]

        rolling7 = np.mean(history[-7:])
        rolling30 = np.mean(history[-30:])
        rolling90 = np.mean(history[-90:])

        row = df.iloc[-1:].copy()

        row["Date"] = next_date

        row["Lag1"] = lag1
        row["Lag7"] = lag7
        row["Lag30"] = lag30
        row["Lag90"] = lag90

        row["Rolling7"] = rolling7
        row["Rolling30"] = rolling30
        row["Rolling90"] = rolling90

        dayofyear = next_date.timetuple().tm_yday

        row["Month_Sin"] = np.sin(
            2 * np.pi * next_date.month / 12
        )

        row["Month_Cos"] = np.cos(
            2 * np.pi * next_date.month / 12
        )

        row["Day_Sin"] = np.sin(
            2 * np.pi * dayofyear / 365
        )

        row["Day_Cos"] = np.cos(
            2 * np.pi * dayofyear / 365
        )

        change = model.predict(
            row[FEATURES]
        )[0]

        pred = max(
            0,
            lag1 + change
        )

        history.append(pred)

        forecasts.append(
            [next_date, pred]
        )

    pred_df = pd.DataFrame(
        forecasts,
        columns=["Date", "Demand"]
    )

    pred_df["Upper"] = (
        pred_df["Demand"] * 1.05
    )

    pred_df["Lower"] = (
        pred_df["Demand"] * 0.95
    )

    return pred_df


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def dashboard():
    history = df.tail(60)
    forecast = forecast_90_days()

    current_val = float(history["Electricity_Demand_MW"].iloc[-1])
    avg_val = float(history["Electricity_Demand_MW"].mean())
    peak_val = float(history["Electricity_Demand_MW"].max())
    forecast_val = float(forecast["Demand"].iloc[0])

    trend_pct = ((current_val - avg_val) / avg_val) * 100
    forecast_trend_pct = ((forecast_val - current_val) / current_val) * 100

    return render_template(
        "dashboard.html",
        current=int(current_val),
        avg=int(avg_val),
        peak=int(peak_val),
        forecast_value=int(forecast_val),
        trend_dir="up" if trend_pct >= 0 else "down",
        trend_pct_abs=round(abs(trend_pct), 1),
        forecast_trend_dir="up" if forecast_trend_pct >= 0 else "down",
        forecast_trend_pct_abs=round(abs(forecast_trend_pct), 1),
        last_updated=history["Date"].iloc[-1].strftime("%b %d, %Y"),
        history_dates=history["Date"].dt.strftime("%Y-%m-%d").tolist(),
        history_values=history["Electricity_Demand_MW"].tolist(),
        forecast_dates=forecast["Date"].dt.strftime("%Y-%m-%d").tolist(),
        forecast_values=forecast["Demand"].tolist(),
        upper=forecast["Upper"].tolist(),
        lower=forecast["Lower"].tolist(),
    )


@app.route("/report")
def report():
    forecast = forecast_90_days()

    peak_row = forecast.loc[forecast["Demand"].idxmax()]
    forecast_table = forecast.round(2).to_html(
        index=False,
        classes="forecast-table",
        border=0,
    )

    return render_template(
        "report.html",
        prediction=int(forecast["Demand"].iloc[0]),
        peak_value=int(peak_row["Demand"]),
        peak_date=peak_row["Date"].strftime("%Y-%m-%d"),
        forecast_table=forecast_table,
    )


@app.route("/download")
def download():

    # Generate 90-day forecast
    forecast = forecast_90_days().copy()

    # Format report
    report = pd.DataFrame({
        "Forecast_Date": forecast["Date"].dt.strftime("%Y-%m-%d"),
        "Predicted_Electricity_Demand_MW": forecast["Demand"].round(2),
        "Lower_Confidence_Bound": forecast["Lower"].round(2),
        "Upper_Confidence_Bound": forecast["Upper"].round(2)
    })

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        report.to_excel(
            writer,
            sheet_name="90-Day Forecast",
            index=False
        )

    output.seek(0)

    return send_file(
        output,
        download_name="Electricity_Demand_90_Day_Forecast.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
