import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv(r'C:\Users\Murali\Downloads\kinathukadavu_ml_ready.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df = df.dropna()

df = df[(df['water_level_m'] > -50) & (df['water_level_m'] < 50)]

for lag in range(1,6):
    df[f'water_level_m_lag{lag}'] = df['water_level_m'].shift(lag)
df = df.dropna()

X = df.drop(columns=['datetime', 'water_level_m'])
y = df['water_level_m']
df = df.dropna()

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
Y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"GROUND WATER PREDICTIVE ANALYSIS USING RANDOM FOREST REGRESSION: ")
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Value: {r2:.4f}')

joblib.dump(model, "groundwater_model_multilag.pkl")

results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
results.to_csv("groundwater_predictions_multilag.csv", index=False)
print("Predictions saved.")

