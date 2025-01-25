import pandas as pd

df = pd.read_csv('data/england_speeches_sample.csv')
# Assuming your DataFrame is called 'df' and the date column is named 'date'
df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime if it's not already
df_filtered = df[df['date'] > '2014-12-31']
#print the earliest date
print(df_filtered['date'].min())

df_filtered.to_csv('data/england_speeches_sample_2015.csv'
                   , index=False)