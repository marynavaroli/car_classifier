import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

df = pd.read_csv('electric_vehicles_spec_2025.csv')
df.head()

X = df.drop(columns=['brand', 'model', 'battery_type', 'efficiency_wh_per_km', 'fast_charge_port', 'source_url', 'cargo_volume_l'])
y = df['efficiency_wh_per_km']
X_feature_selected = df[['top_speed_kmh', 'torque_nm', 'fast_charging_power_kw_dc', 'height_mm', 'car_body_type']]

num_cols = X_feature_selected.select_dtypes(include=np.number).columns
cat_cols = X_feature_selected.select_dtypes(include=['object']).columns

X_train, X_temp, y_train, y_temp = train_test_split(
    X_feature_selected, y, test_size=0.3, random_state=1
)

# Now split temp into validation and test (50/50 of 0.3 → 15% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=1
)

num_vals = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])
cat_vals = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown="ignore", drop = 'first'))])

preprocess = ColumnTransformer(
    transformers=[ ("num_preprocess", num_vals, num_cols),
                   ("cat_preprocess", cat_vals, cat_cols)])

lr = KNeighborsRegressor(n_neighbors=1)

bc = BaggingRegressor(estimator=lr, n_estimators=50)
bc_pipe_feature = Pipeline([
    ("preprocess", preprocess),
    ("regr", bc) 
], memory=None)
bc_pipe_feature.fit(X_train, y_train)
print("Test R^2 score:", bc_pipe_feature.score(X_test, y_test))
print("Validation R^2 score:", bc_pipe_feature.score(X_val, y_val))

def create_df(speed, torque, charging, height, body):
    df = pd.DataFrame({
        "top_speed_kmh": [speed],
        "torque_nm": [torque],  
        "fast_charging_power_kw_dc": [charging],
        "height_mm": [height],
        "car_body_type": [body]
    })
    return df

def get_prediction(speed, torque, charging, height, body):
    df = create_df(speed, torque, charging, height, body)
    pred = bc_pipe_feature.predict(df)
    return pred
