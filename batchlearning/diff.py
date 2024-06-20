import pandas as pd

# Read the CSV files
df1 = pd.read_csv('D:\bachelor\forecast.csv')
df2 = pd.read_csv('D:\bachelor\historical.csv')

# Find the rows where the dataframes differ
diff = df1.compare(df2)

# Print the rows with differences
print(diff)