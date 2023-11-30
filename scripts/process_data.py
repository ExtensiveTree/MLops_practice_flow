import pandas as pd
import numpy as np
from datetime import datetime
import time
df = pd.read_csv('/home/akashy/staidg/datasets/data.csv')


df = df.drop(np.where(df['index'] == '2020-09-23T14:36:00Z')[0])
df = df.reset_index(drop=True)
df = df.drop(np.where(df['index'] == '2022-12-25T12:36:15Z')[0])
df = df.reset_index(drop=True)
df['counts'] = (df['counts'] - df['counts'].min())/(df['counts'].max()-df['counts'].min())

df = df.drop(np.where(df['counts'] < 0.01)[0])
df  = df.iloc [0:, 1:3]

df['index'] = pd.to_datetime(df['index']).dt.tz_convert('US/Eastern')
df['index'] = pd.to_datetime(df['index'], unit='s')
df['year'] = df['index'].dt.year
df['month'] = df['index'].dt.month
df['day'] = df['index'].dt.day
df['hour'] = df['index'].dt.hour
df['minute'] = df['index'].dt.minute
df = df.drop(columns ="index").reset_index(drop=True)


df.to_csv('/home/akashy/staidg/datasets/data_processed.csv')