from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

df = pd.read_csv('/home/akashy/staidg/datasets/data_train.csv')
df  = df.iloc [0:, 2:8]
df['counts'] = df['counts'].fillna(0)
X = df.iloc [0:, 1:8]
y = df.iloc [0:, 0:1]
model = RandomForestRegressor(max_depth=2, random_state=0)

with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/akashy/staidg/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()


model.fit(X.values, y.values)

with open('/home/akashy/staidg/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)