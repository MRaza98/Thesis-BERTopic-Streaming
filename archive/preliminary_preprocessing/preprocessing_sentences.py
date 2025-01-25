import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
import time
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_nltk_resources():
    """
    Ensure all required NLTK resources are downloaded
    """
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logging.error(f"Failed to download NLTK resources: {str(e)}")
        raise

def split_text(row, min_words=10):
    """
    Split text into sentences and filter out those with fewer than min_words words.
    Returns a list of dictionaries, one for each valid sentence.
    """
    sentences = sent_tokenize(row['text'])
    
    result = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) >= min_words:
            new_row = dict(row)
            new_row['text'] = sentence
            new_row['word_count'] = len(words)
            result.append(new_row)
    
    return result

def initialize_worker():
    """
    Initialize each worker process with required NLTK resources
    """
    ensure_nltk_resources()

def process_chunk(chunk):
    """
    Process a chunk of the DataFrame
    """
    results = []
    for _, row in chunk.iterrows():
        results.extend(split_text(row))
    return pd.DataFrame(results)

def process_csv_in_chunks(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk[chunk['chair'] == False]

def write_chunk_to_csv(df_chunk, output_path, first_chunk=False):
    """
    Write a chunk to CSV file, either creating a new file or appending to existing
    """
    mode = 'w' if first_chunk else 'a'
    header = first_chunk
    df_chunk.to_csv(output_path, mode=mode, header=header, index=False)

if __name__ == '__main__':
    start_time = time.time()
    logging.info("Starting preprocessing of speeches")

    # File paths
    file_path = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_sample_2015_20k.csv'
    output_path = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/sentences_england_speeches_sample_2015_20k.csv'
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize multiprocessing pool with worker initialization
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores, initializer=initialize_worker)

    total_rows = 0
    is_first_chunk = True

    try:
        for i, chunk in enumerate(process_csv_in_chunks(file_path)):
            chunk_start_time = time.time()
            total_rows += len(chunk)
            logging.info(f"Processing chunk {i+1}, total rows processed: {total_rows}")
            
            # Split the chunk for parallel processing
            df_split = np.array_split(chunk, num_cores)
            
            # Process chunks in parallel
            results = pool.map(process_chunk, df_split)
            
            # Combine results from parallel processing
            df_chunk = pd.concat(results, ignore_index=True)
            
            # Write chunk to CSV
            write_chunk_to_csv(df_chunk, output_path, first_chunk=is_first_chunk)
            is_first_chunk = False
            
            chunk_time = time.time() - chunk_start_time
            logging.info(f"Chunk {i+1} processed in {chunk_time:.2f} seconds")
            logging.info(f"Processed {len(df_chunk)} sentences in this chunk")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise
    
    finally:
        pool.close()
        pool.join()

    # Get final statistics
    try:
        final_df = pd.read_csv(output_path)
        logging.info(f"Final dataset statistics:")
        logging.info(f"Total number of sentences: {len(final_df)}")
        logging.info(f"Average sentence length: {final_df['word_count'].mean():.2f} words")
        logging.info(f"Minimum sentence length: {final_df['word_count'].min()} words")
        logging.info(f"Maximum sentence length: {final_df['word_count'].max()} words")
    except Exception as e:
        logging.error(f"Error reading final statistics: {str(e)}")

    end_time = time.time()
    logging.info(f"Processing complete. Total time: {end_time - start_time:.2f} seconds")