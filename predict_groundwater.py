import pandas as pd
import joblib

model = joblib.load("groundwater_model.pkl")

df_new = pd.read_csv(r'C:\Users\Murali\Downloads\kinathukadavu_ml_ready.csv')
df_new['datetime'] = pd.to_datetime(df_new['datetime'], format='%d-%m-%Y %H:%M')
df_new = df_new.dropna()

X_new = df_new.drop(columns=['datetime', 'water_level_m'], errors='ignore')

predictions = model.predict(X_new)

df_new['predicted_water_level'] = predictions
df_new.to_csv('predicted_groundwater_levels.csv', index=False)

print("Prediction completed and saved to 'predicted_groundwater_levels.csv")

