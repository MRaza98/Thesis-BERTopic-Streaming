"""This script samples data from the entire csv for each country
i.e. when 2010 onwards data is needed for the cross parliamentary analysis.
It is a template which can be modified to sample any ParlSpeechV2 csv.
"""

import pandas as pd

# Read the original CSV file
# csv of any of the parliaments
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/Corp_Bundestag_V2.csv')

# Convert date column to datetime if it's not already
df['date'] = pd.to_datetime(df['date'])

# Create mask for date range to sample data
mask_date_range = (df['date'].dt.year >= 2010)

# Combine masks and filter the dataframe
filtered_df = df[mask_date_range]

# Sort by date for better organization
filtered_df = filtered_df.sort_values('date')

# Write the filtered data to a new file
filtered_df.to_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/germany_speeches_2010.csv', 
                  index=False)

# Print some info about the filtered dataset
print("\nDataset Summary:")
print(f"Total number of speeches: {len(filtered_df)}")
print(f"Date range: from {filtered_df['date'].min()} to {filtered_df['date'].max()}")

# Print breakdown by year and month
print("\nBreakdown by year and month:")
monthly_counts = filtered_df.groupby([df['date'].dt.year, 
                                    df['date'].dt.month]).size()
total_speeches = sum(monthly_counts)
for (year, month), count in monthly_counts.items():
    print(f"{year}-{month:02d}: {count} speeches")
print(total_speeches)