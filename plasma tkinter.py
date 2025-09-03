import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.optimize import minimize
import os

# Load Data
df = pd.read_csv("plasma_spray_data_2.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df[["stand_of_distance", "plasma_power", "shroud_gas_pressure", "oxide_content"]]
df = df.dropna()

X = df[["stand_of_distance", "plasma_power", "shroud_gas_pressure"]]
y = df["oxide_content"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_scaled, y)

# Prediction Logic
def predict_oxide(params_scaled):
    df_scaled = pd.DataFrame([params_scaled], columns=["stand_of_distance", "plasma_power", "shroud_gas_pressure"])
    return model.predict(df_scaled)[0]

def optimize_for_oxide(desired_oxide=0):
    def objective(params_scaled):
        return abs(predict_oxide(params_scaled) - desired_oxide)

    bounds_real = [
        (df["stand_of_distance"].min(), df["stand_of_distance"].max()),
        (df["plasma_power"].min(), df["plasma_power"].max()),
        (df["shroud_gas_pressure"].min(), df["shroud_gas_pressure"].max())
    ]

    low_vals = scaler.transform([ [b[0] for b in bounds_real] ])[0]
    high_vals = scaler.transform([ [b[1] for b in bounds_real] ])[0]
    bounds_scaled = list(zip(low_vals, high_vals))

    best_result = None
    min_error = float("inf")

    for _ in range(10):
        x0 = np.random.uniform(*zip(*bounds_scaled))
        result = minimize(objective, x0, bounds=bounds_scaled)
        if result.success:
            pred = predict_oxide(result.x)
            error = abs(pred - desired_oxide)
            if error < min_error:
                min_error = error
                best_result = result

    if best_result:
        df_scaled = pd.DataFrame([best_result.x], columns=["stand_of_distance", "plasma_power", "shroud_gas_pressure"])
        unscaled = scaler.inverse_transform(df_scaled)[0]
        return {
            "Desired Oxide Content": desired_oxide,
            "Stand-off Distance": round(unscaled[0], 2),
            "Plasma Power": round(unscaled[1], 2),
            "Shroud Gas Pressure": round(unscaled[2], 2),
            "Predicted Oxide Content": round(predict_oxide(best_result.x), 3)
        }
    else:
        return None

# GUI Functions
last_result = {}

def predict():
    global last_result
    try:
        user_input = entry.get().strip()
        if not user_input:
            raise ValueError("Empty input")
        desired = float(user_input)
        if desired < 0:
            raise ValueError("Oxide content must be ≥ 0")

        result = optimize_for_oxide(desired)
        if result:
            last_result = result
            output = "\n".join([f"{k}: {v}" for k, v in result.items()])
            messagebox.showinfo("Prediction", output)
        else:
            messagebox.showerror("Optimization Failed", "Could not find parameters for that oxide level.")

    except Exception as e:
        messagebox.showerror("Invalid Input", f"Please enter a valid number ≥ 0.\n\nDetails: {e}")

def save():
    if last_result:
        df_out = pd.DataFrame([last_result])
        file_exists = os.path.exists("prediction_results.csv")
        df_out.to_csv("prediction_results.csv", mode="a", header=not file_exists, index=False)
        messagebox.showinfo("Saved", "Result saved to prediction_results.csv")
    else:
        messagebox.showwarning("Nothing to save", "Predict first before saving.")

# GUI Layout
root = tk.Tk()
root.title("Oxide → Spray Parameter Predictor")

tk.Label(root, text="Enter Desired Oxide Content (≥ 0)", font=("Arial", 12)).pack(pady=10)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=5)

tk.Button(root, text="Predict Parameters", font=("Arial", 12), command=predict).pack(pady=10)
tk.Button(root, text="Save Result to CSV", font=("Arial", 12), command=save).pack(pady=5)

root.geometry("420x240")
root.mainloop()