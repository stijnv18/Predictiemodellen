import pandas as pd
df = pd.read_csv('customer_36.csv')
df['timestamp'] = pd.date_range(start='2010-07-01', periods=len(df), freq='H')
df.to_csv('customer_36v2.csv', index=False)