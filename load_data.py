import pandas as pd

data = pd.read_pickle('./data_full.pkl')
print(data.head())
print(data.tail())