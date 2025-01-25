import pandas as pd
import os
from pathlib import Path

# Define input and output paths
input_file = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/Corp_HouseOfCommons_V2.csv'
output_dir = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/yearly_england'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv(input_file)

# Convert date column to datetime
# Note: You might need to adjust the date format depending on your data
df['date'] = pd.to_datetime(df['date'])

# Extract year from date column
df['year'] = df['date'].dt.year

# Filter for years 2000 and after
df = df[df['year'] >= 2000]

# Group by year and save separate CSV files
for year in df['year'].unique():
    year_df = df[df['year'] == year]
    output_file = os.path.join(output_dir, f'england_speeches_{year}.csv')
    print(f"Saving speeches from {year} to {output_file}")
    year_df.to_csv(output_file, index=False)

print("Processing complete!")