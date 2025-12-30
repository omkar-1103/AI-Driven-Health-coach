import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

df = pd.DataFrame({

    # Time features
    'hour_of_day': np.random.randint(0, 24, n),
    'day_of_week': np.random.randint(0, 7, n),

    # Environmental
    'temperature': np.random.normal(30, 5, n),
    'humidity': np.random.normal(60, 15, n),
    'air_quality_index': np.random.normal(120, 50, n),
    'pm2_5': np.random.normal(55, 25, n),
    'noise_level': np.random.normal(65, 10, n),

    # Activity
    'activity_type': np.random.choice([0, 1, 2, 3], n),  # 0=rest,1=walk,2=workout,3=travel
    'steps': np.abs(np.random.normal(1500, 1200, n)),
    'calories_burned': np.abs(np.random.normal(70, 50, n)),

    # Stress indicators
    'oxygen_stress': np.abs(np.random.normal(0.3, 0.6, n)),
    'immune_stress': np.abs(np.random.normal(0.4, 0.7, n)),

    # Sleep
    'deep_sleep_ratio': np.clip(np.random.normal(0.20, 0.04, n), 0.05, 0.6),
    'rem_sleep_ratio': np.clip(np.random.normal(0.24, 0.05, n), 0.05, 0.6),
    'sleep_architecture': np.clip(np.random.normal(0.94, 0.03, n), 0.6, 1),
    'env_sleep_disruption': np.clip(np.random.normal(0.8, 0.25, n), 0, 2),
    'sleep_quality': np.clip(np.random.normal(33, 6, n), 1, 100),

    # Stress flags
    'high_stress_hour': np.random.choice([0, 1], n, p=[0.75, 0.25]),
    'hourly_stress': np.random.normal(41.2, 0.1, n)
})

# Inject outliers
outlier_idx = np.random.choice(df.index, size=500, replace=False)
df.loc[outlier_idx, 'air_quality_index'] *= 3
df.loc[outlier_idx, 'pm2_5'] *= 2
df.loc[outlier_idx, 'steps'] *= 5
df.loc[outlier_idx, 'calories_burned'] *= 4

for col in df.columns:
    df.loc[df.sample(frac=0.03).index, col] = np.nan

df['illness_risk'] = (
    25
    + 30 * df['immune_stress']
    + 20 * df['oxygen_stress']
    + 0.08 * df['air_quality_index']
    + 0.05 * df['pm2_5']
    - 40 * (df['deep_sleep_ratio'] + df['rem_sleep_ratio'])
    + np.random.normal(0, 10, n)   # noise
)

df['illness_risk'] = df['illness_risk'].clip(lower=0)

df['environmental_load'] = (
    0.4 * df['air_quality_index'] +
    0.4 * df['pm2_5'] +
    0.2 * df['noise_level']
)


df['sleep_efficiency'] = (
    df['deep_sleep_ratio'] + df['rem_sleep_ratio']
)

df['activity_intensity'] = (
    df['steps'] / (df['calories_burned'] + 1)
)


df['stress_load'] = (
    0.6 * df['immune_stress'] +
    0.4 * df['oxygen_stress']
)

df['circadian_risk'] = np.where(
    (df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5),
    df['hourly_stress'] * 1.2,
    df['hourly_stress']
)


from sklearn.impute import SimpleImputer

num_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

required_cols = [
    'immune_stress',
    'oxygen_stress',
    'sleep_efficiency',
    'environmental_load',
    'illness_risk'
]

missing = set(required_cols) - set(df.columns)
print("Missing columns:", missing)


X = df[
    ['immune_stress', 'oxygen_stress',
     'sleep_efficiency', 'environmental_load']
]

y = df['illness_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,#12
    min_samples_split=6,#8
    min_samples_leaf=4,
    random_state=42
    )

rf_model.fit(X_train, y_train)


y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("🔹 TRAIN METRICS")
print("MAE :", mean_absolute_error(y_train, y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("R²  :", r2_score(y_train, y_train_pred))

print("\n🔹 TEST METRICS")
print("MAE :", mean_absolute_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R²  :", r2_score(y_test, y_test_pred))


# predict for new user
new_user = pd.DataFrame({
    'immune_stress': [1.1],
    'oxygen_stress': [0.7],
    'sleep_efficiency': [0.38],
    'environmental_load': [110]
})

predicted_risk = rf_model.predict(new_user)
print("Predicted Illness Risk:", predicted_risk[0])

import os
os.getcwd()

os.chdir(r'C:\Users\adity\Downloads')
os.getcwd()

import joblib

joblib.dump(rf_model, "illness_risk.pkl")


features = [
    'immune_stress',
    'oxygen_stress',
    'sleep_efficiency',
    'environmental_load'
]

joblib.dump(features, "illness_risk_features.pkl")


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf_model, f"{model_dir}/illness_risk.pkl")
joblib.dump(features, f"{model_dir}/illness_risk_features.pkl")

rf_model = joblib.load("models/illness_risk.pkl")
features = joblib.load("models/illness_risk_features.pkl")


import pandas as pd

new_user = pd.DataFrame({
    'immune_stress': [1.2],
    'oxygen_stress': [0.7],
    'sleep_efficiency': [0.39],
    'environmental_load': [105]
})

new_user = new_user[features]

prediction = rf_model.predict(new_user)
print("Predicted Illness Risk:", prediction[0])
