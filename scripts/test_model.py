from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd

df = pd.read_csv('/home/akashy/staidg/datasets/data_test.csv')
df  = df.iloc [0:, 2:8]
df['counts'] = df['counts'].fillna(0)
X = df.iloc [0:, 1:8]
y = df.iloc [0:, 0:1]
model = RandomForestRegressor(max_depth=2, random_state=0)
with open('/home/akashy/staidg/models/data.pickle', 'rb') as f:
    model = pickle.load(f)

score = model.score(X.values, y.values)
print("score=", score)