import pandas as pd

data = pd.read_pickle('./data.pkl')
print(data.head())
print(data.tail())