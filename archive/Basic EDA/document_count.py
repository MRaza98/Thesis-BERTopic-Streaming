import pandas as pd
import os

# Base path for the files
base_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/yearly_england"

# Years we want to process (2010-2019)
years = range(2010, 2020)

# Dictionary to store results
row_counts = {}

# Process each year
for year in years:
    file_name = f"nltk_stopwords_preprocessed_england_speeches_{year}.csv"
    file_path = os.path.join(base_path, file_name)
    
    try:
        # Read and count rows
        df = pd.read_csv(file_path)
        row_count = len(df)
        row_counts[year] = row_count
        print(f"Year {year}: {row_count} rows")
    except Exception as e:
        print(f"Error processing {year}: {e}")

# Print total
total_rows = sum(row_counts.values())
print(f"\nTotal rows across all files (2010-2019): {total_rows}")