import pandas as pd

# List of file names
files = ['2010_short_correct.csv', '2011_short_correct.csv', '2012_short_correct.csv']

# List to hold DataFrames
dfs = []

# Read each file and append the DataFrame to the list
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined DataFrame to a CSV file
combined_df.to_csv('combined.csv', index=False)