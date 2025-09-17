import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\Murali\Downloads\kinathukadavu_ml_ready.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df = df.dropna()

X = df.drop(columns=['datetime', 'water_level_m'])
y = df['water_level_m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("GROUND WATER PREDICTION ANALYSIS USING LINEAR REGRESSION \n")
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Value: {r2:.4f}')

joblib.dump(model, "groundwater_model.pkl")
print("Model saved.")
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
results.to_csv("groundwater_predictions.csv", index=False)
print("Predictions saved.")


