import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('merged_data.csv')
print(df.head())

# Convert 'DateTime' to datetime and set as index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

print("describe")
print(df.describe())

print("correlation")
print(df.corr())

print("covariance")
print(df.cov())



# Select numeric columns only
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Create a separate subplot for each column for histograms
fig, axs = plt.subplots(len(numeric_df.columns), 1, figsize=(5, 5*len(numeric_df.columns)))

for i, col in enumerate(numeric_df.columns):
    axs[i].hist(numeric_df[col], bins=20, color='g', alpha=0.75)
    axs[i].set_title(col)
    axs[i].set_xlabel('Value')  # x-axis label
    axs[i].set_ylabel('Frequency')  # y-axis label

# Adjust the spacing between subplots
plt.subplots_adjust(hspace = 0.5)

# Create a separate subplot for each column for boxplots
fig, axs = plt.subplots(len(numeric_df.columns), 1, figsize=(5, 5*len(numeric_df.columns)))

for i, col in enumerate(numeric_df.columns):
    axs[i].boxplot(numeric_df[col].dropna(), vert=False)
    axs[i].set_title(col)
    axs[i].set_xlabel('Value')  # x-axis label
    axs[i].set_ylabel('Boxplot')  # y-axis label

# Adjust the spacing between subplots
plt.subplots_adjust(hspace = 0.5)

# Calculate mean
mean = numeric_df.mean()
print("Mean:\n", mean)

# Calculate median
median = numeric_df.median()
print("Median:\n", median)

# Calculate mode
mode = numeric_df.mode()
print("Mode:\n", mode)

# Calculate standard deviation
std_dev = numeric_df.std()
print("Standard Deviation:\n", std_dev)

# Calculate variance
variance = numeric_df.var()
print("Variance:\n", variance)

plt.show()