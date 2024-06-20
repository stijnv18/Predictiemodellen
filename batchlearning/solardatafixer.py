# import pandas as pd
# import matplotlib.pyplot as plt

# #read the file
# df = pd.read_csv('2011-2012 Solar home electricity data v2.csv')

# # Convert 'date' column to datetime
# df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# # Define the id_vars for the melt function
# id_vars = ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date']

# # Melt the DataFrame to have a single datetime column
# df_melted = df.melt(id_vars=id_vars, var_name='time', value_name='consumption')

# # Combine the 'date' and 'time' columns into a single 'datetime' column
# df_melted['datetime'] = pd.to_datetime(df_melted['date'].dt.date.astype(str) + ' ' + df_melted['time'])

# # Drop the separate 'date' and 'time' columns
# df_melted.drop(['date', 'time'], axis=1, inplace=True)

# # Set 'datetime' as the index
# df_melted.set_index('datetime', inplace=True)

# # Sort the DataFrame by 'datetime'
# df_melted.sort_values('datetime', inplace=True)
# df_melted.to_csv('2011_correct.csv')
import pandas as pd

# Assuming df is your DataFrame
df = pd.read_csv('2011-2012 Solar home electricity data v2.csv')

# Melt the DataFrame
df_melted = df.melt(id_vars=['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date'], 
                    var_name='time', 
                    value_name='value')

# Combine the date and time columns
df_melted['date'] = pd.to_datetime(df_melted['date'].astype(str) + ' ' + df_melted['time'].astype(str))

# Drop the time column
df_melted = df_melted.drop(columns=['time'])

# Now df_melted is in the desired format

# Write the reshaped DataFrame to a new CSV file
df_melted.to_csv('2011_correct.csv', index=False)