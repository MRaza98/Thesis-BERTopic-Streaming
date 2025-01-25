import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_text(row, max_words=384):
    words = row['text'].split()
    total_words = len(words)
    
    if total_words <= max_words:
        return [dict(row, new_terms=total_words)]
    
    result = []
    for i in range(0, total_words, max_words):
        split_words = words[i:i+max_words]
        new_row = dict(row)
        new_row['text'] = ' '.join(split_words)
        new_row['new_terms'] = len(split_words)
        result.append(new_row)
    
    return result

def process_chunk(chunk):
    results = []
    for _, row in chunk.iterrows():
        results.extend(split_text(row))
    return results

def process_csv_in_chunks(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk[chunk['chair'] == False]

if __name__ == '__main__':
    start_time = time.time()
    logging.info("Starting preprocessing of speeches")

    file_path = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/england_speeches_sample_2015.csv'
    output_path = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_sample_2015.csv'

    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    all_results = []
    total_rows = 0

    for i, chunk in enumerate(process_csv_in_chunks(file_path)):
        total_rows += len(chunk)
        logging.info(f"Processing chunk {i+1}, total rows processed: {total_rows}")
        
        df_split = np.array_split(chunk, num_cores)
        results = pool.map(process_chunk, df_split)
        all_results.extend([item for sublist in results for item in sublist])

    logging.info("Creating final dataframe")
    df_final = pd.DataFrame(all_results)
    logging.info(f"Final dataframe shape: {df_final.shape}")

    logging.info(f"Writing results to {output_path}")
    df_final.to_csv(output_path, index=False)

    end_time = time.time()
    logging.info(f"Processing complete. Total time: {end_time - start_time:.2f} seconds")