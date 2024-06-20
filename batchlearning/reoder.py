import pandas as pd

# Read the CSV file
df = pd.read_csv('2012_melted.csv')

# Pivot the 'Consumption Category' column
pivot_df = df.pivot_table(index=['datetime', 'Customer', 'Generator Capacity', 'Postcode'], 
                          columns='Consumption Category', 
                          values='consumption', 
                          fill_value=0)

# Reset the index to make the index columns into regular columns
pivot_df.reset_index(inplace=True)

# Rename the columns
pivot_df.columns.name = None
pivot_df.rename(columns={'GG': 'consumption gg', 'CL': 'consumption cl', 'GC': 'consumption gc'}, inplace=True)

# Write the DataFrame back to CSV
pivot_df.to_csv('2012_short_correct.csv', index=False)