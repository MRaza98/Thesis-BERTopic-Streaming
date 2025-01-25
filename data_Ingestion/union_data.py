import pandas as pd

# Read all three datasets
df_2016 = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2016.csv')
df_2017 = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2017.csv')
df_2018 = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2018.csv')

# Concatenate all dataframes
combined_df = pd.concat([df_2016, df_2017, df_2018], ignore_index=True)

# Convert date to datetime for proper sorting (optional)
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Sort by date (optional)
combined_df = combined_df.sort_values('date')

# Save the combined dataset
combined_df.to_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2016_2018.csv', index=False)

# Print some info about the combined dataset
print(f"Total number of speeches: {len(combined_df)}")
print(f"Date range: from {combined_df['date'].min()} to {combined_df['date'].max()}")
print(f"Number of speeches per year:")
print(combined_df['date'].dt.year.value_counts().sort_index())