import pandas as pd
import numpy as np
import multiprocessing as mp
from pathlib import Path
import logging
import time
from typing import List, Dict, Generator, Any
import traceback
from functools import partial
from transformers import AutoTokenizer
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass

class MPNetChunker:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # MPNet uses <s> and </s> as special tokens
        self.special_tokens_count = 2
        logger.info(f"Initialized MPNet tokenizer: {model_name}")

    def split_text(self, row: Dict[str, Any], max_tokens: int = 384, overlap_tokens: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into chunks using MPNet tokenizer with overlap.
        
        Args:
            row: Dictionary containing text and metadata
            max_tokens: Maximum number of tokens per chunk (including special tokens)
            overlap_tokens: Number of overlapping tokens between chunks
        """
        try:
            text = row['text']
            
            # MPNet specific tokenization
            tokens = self.tokenizer(
                text,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
            input_ids = tokens['input_ids'][0]
            total_tokens = len(input_ids)
            
            # Account for MPNet special tokens (<s> and </s>)
            effective_max_tokens = max_tokens - self.special_tokens_count
            
            if total_tokens <= max_tokens:
                decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                return [dict(row, 
                           text=decoded_text,
                           chunk_id=0, 
                           total_chunks=1, 
                           n_tokens=total_tokens,
                           is_overlapping=False)]
            
            # Split into chunks with careful sentence boundary handling
            result = []
            chunk_id = 0
            stride = effective_max_tokens - overlap_tokens
            
            for start_idx in range(0, total_tokens, stride):
                end_idx = min(start_idx + effective_max_tokens, total_tokens)
                
                # Extract chunk tokens
                chunk_tokens = input_ids[start_idx:end_idx]
                
                # Skip small final chunks
                if len(chunk_tokens) < effective_max_tokens * 0.5:
                    break
                
                # Decode chunk back to text
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                
                # Find last sentence boundary if this isn't the first chunk
                if start_idx > 0:
                    # Look for sentence boundaries (., !, ?)
                    sentence_ends = [i for i, char in enumerate(chunk_text) if char in ['.', '!', '?']]
                    if sentence_ends:
                        # Take the last complete sentence
                        last_sentence_end = sentence_ends[-1] + 1
                        chunk_text = chunk_text[last_sentence_end:].strip()
                        
                # Find first sentence boundary if this isn't the last chunk
                if end_idx < total_tokens:
                    sentence_ends = [i for i, char in enumerate(chunk_text) if char in ['.', '!', '?']]
                    if sentence_ends:
                        # Take up to the last complete sentence
                        last_sentence_end = sentence_ends[-1] + 1
                        chunk_text = chunk_text[:last_sentence_end].strip()
                
                # Recompute token count for the adjusted text
                adjusted_tokens = self.tokenizer(
                    chunk_text,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                actual_token_count = len(adjusted_tokens['input_ids'][0])
                
                new_row = dict(row)
                new_row.update({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'n_tokens': actual_token_count,
                    'chunk_start_idx': start_idx,
                    'is_overlapping': start_idx > 0,
                    'token_coverage': f"{start_idx}-{end_idx}/{total_tokens}"
                })
                
                result.append(new_row)
                chunk_id += 1
            
            # Add total_chunks to all chunks
            for r in result:
                r['total_chunks'] = len(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing row: {str(e)}\n{traceback.format_exc()}")
            return []

def process_chunk(chunker: MPNetChunker, chunk: pd.DataFrame, max_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
    """Process a chunk of the dataframe"""
    try:
        results = []
        split_func = partial(chunker.split_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        
        for _, row in chunk.iterrows():
            results.extend(split_func(row.to_dict()))
        return results
    except Exception as e:
        logger.error(f"Error in process_chunk: {str(e)}\n{traceback.format_exc()}")
        return []

def process_csv_in_chunks(
    file_path: Path,
    chunk_size: int = 10000,
    filter_condition: Dict[str, Any] = None
) -> Generator[pd.DataFrame, None, None]:
    """Process CSV file in chunks with optional filtering."""
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if filter_condition:
                for col, val in filter_condition.items():
                    chunk = chunk[chunk[col] == val]
            if not chunk.empty:
                yield chunk
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}\n{traceback.format_exc()}")
        raise PreprocessingError("Failed to process CSV file")

def main(
    input_path: str,
    output_path: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    max_tokens: int = 384,
    overlap_tokens: int = 50,
    chunk_size: int = 10000,
    filter_condition: Dict[str, Any] = None
):
    """Main processing function"""
    start_time = time.time()
    logger.info("Starting preprocessing of speeches")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Initialize tokenizer
    chunker = MPNetChunker(model_name)
    
    # Validate paths
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up multiprocessing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    logger.info(f"Using {num_cores} CPU cores")
    
    pool = mp.Pool(num_cores)
    all_results = []
    total_rows = 0
    
    try:
        # Process chunks
        for i, chunk in enumerate(process_csv_in_chunks(input_path, chunk_size, filter_condition)):
            total_rows += len(chunk)
            logger.info(f"Processing chunk {i+1}, total rows processed: {total_rows}")
            
            df_split = np.array_split(chunk, num_cores)
            process_func = partial(process_chunk, chunker, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            results = pool.map(process_func, df_split)
            all_results.extend([item for sublist in results for item in sublist])
            
            # Periodic progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {total_rows} rows in {time.time() - start_time:.2f} seconds")
        
        # Create final dataframe
        logger.info("Creating final dataframe")
        df_final = pd.DataFrame(all_results)
        
        # Add metadata
        df_final['processing_timestamp'] = pd.Timestamp.now()
        df_final['source_file'] = input_path.name
        df_final['tokenizer_name'] = model_name
        
        logger.info(f"Final dataframe shape: {df_final.shape}")
        
        # Save results
        logger.info(f"Writing results to {output_path}")
        df_final.to_csv(output_path, index=False)
        
        # Log processing statistics
        logger.info(f"Processing complete:")
        logger.info(f"- Input rows: {total_rows}")
        logger.info(f"- Output rows: {len(df_final)}")
        logger.info(f"- Average chunks per document: {len(df_final) / total_rows:.2f}")
        logger.info(f"- Total processing time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        pool.close()
        pool.join()

if __name__ == '__main__':
    # Configuration
    config = {
        'input_path': '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/england_speeches_sample_2015.csv',
        'output_path': '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2015_2019.csv',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'max_tokens': 384,
        'overlap_tokens': 50,
        'chunk_size': 10000,
        'filter_condition': {'chair': False}
    }
    
    try:
        main(**config)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)