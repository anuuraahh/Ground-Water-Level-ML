import pandas as pd
import joblib

model = joblib.load("groundwater_model_multilag.pkl")

df_new = pd.read_csv(r'C:\Users\Murali\Downloads\kinathukadavu_ml_ready.csv')
df_new['datetime'] = pd.to_datetime(df_new['datetime'], format='%d-%m-%Y %H:%M')
df_new = df_new[(df_new['water_level_m'] > -50) & (df_new['water_level_m'] < 50)]

for lag in range(1, 6):
    df_new[f'water_level_m_lag{lag}'] = df_new['water_level_m'].shift(lag)

df_new = df_new.dropna()

X_new = df_new.drop(columns=['datetime', 'water_level_m'])

predictions = model.predict(X_new)
df_new['predicted_water_level'] = predictions
df_new.to_csv('predicted_groundwater_levels.csv', index=False)

print("Prediction completed and saved.")


