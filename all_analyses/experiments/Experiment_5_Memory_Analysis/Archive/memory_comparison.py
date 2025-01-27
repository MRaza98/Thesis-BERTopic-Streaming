import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
import psutil
import os
import time
from typing import Dict, List, Tuple
import gc
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import traceback
import torch
from typing import Dict, List, Tuple, Union

class ResourceExperiment:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.coherence_data_path = os.path.join(base_path, 'nltk_stopwords_preprocessed_england_speeches_2015_2019.csv')
        self.results = {
            'batch': {'memory': [], 'time': []},
            'incremental': {'memory': [], 'time': []}
        }
        self.results_dir = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/Iteration_2'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def save_run_results(self, run_number: int):
        """Save results after each run"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'run_{run_number}_{timestamp}.json'
        filepath = os.path.join(self.results_dir, filename)
        
        run_results = {
            'batch': {
                'memory': self.results['batch']['memory'][-2:],
                'time': self.results['batch']['time'][-2:]
            },
            'incremental': {
                'memory': self.results['incremental']['memory'][-2:],
                'time': self.results['incremental']['time'][-2:]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(run_results, f, indent=4)
        print(f"Results for run {run_number} saved to {filepath}")
    
    def get_memory_usage(self) -> float:
        """Return current memory usage in GB"""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        return memory_gb
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all required datasets"""
        # Explicitly specify dtype for 'text' column as string
        data_2018 = pd.read_csv(
            os.path.join(self.base_path, "nltk_stopwords_preprocessed_england_speeches_2018.csv"),
            dtype={'text': str}  # Force text column to be string
        )
        data_jan_2019 = pd.read_csv(
            os.path.join(self.base_path, "nltk_stopwords_preprocessed_england_speeches_Jan_2019.csv"),
            dtype={'text': str}
        )
        data_feb_2019 = pd.read_csv(
            os.path.join(self.base_path, "nltk_stopwords_preprocessed_england_speeches_Feb_2019.csv"),
            dtype={'text': str}
        )
        
        # Additional safety check to convert any non-string text to string
        data_2018['text'] = data_2018['text'].astype(str)
        data_jan_2019['text'] = data_jan_2019['text'].astype(str)
        data_feb_2019['text'] = data_feb_2019['text'].astype(str)
        
        return data_2018, data_jan_2019, data_feb_2019
    
    def run_batch_experiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """Run batch processing and measure resources"""
        try:
            gc.collect()
            initial_memory = self.get_memory_usage()
            start_time = time.time()

            model = SentenceTransformer("all-miniLM-L6-v2")
            umap_model = UMAP(
                n_components=8,
                n_neighbors=15,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

            topic_model = BERTopic(embedding_model=model,
                                umap_model=umap_model,
                                verbose=True)
            
            topics, probs = topic_model.fit_transform(data['text'].tolist())

            end_time = time.time()
            peak_memory = self.get_memory_usage()

            results_dict = {
                'peak_memory': float(peak_memory),
                'memory_used': float(peak_memory - initial_memory),
                'time_taken': float(end_time - start_time)
            }
            print("Results dictionary:", results_dict) 
            return results_dict

        except Exception as e:
            print(f"Error in run_batch_experiment: {e}")
            traceback.print_exc()  
            raise
        
    def run_first_incremental(self, base_data: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, float]:
        """Special case for first incremental run (2018 + January)"""
        gc.collect()
        initial_memory = self.get_memory_usage()
        start_time = time.time()

        model = SentenceTransformer("all-miniLM-L6-v2")
        umap_model = UMAP(
            n_components=8,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Train base model
        base_model = BERTopic(embedding_model=model,
                            umap_model=umap_model,
                            verbose=True)
        base_model.fit(base_data['text'].tolist())
        
        # Train new model
        new_model = BERTopic(embedding_model=model,
                           umap_model=umap_model,
                           verbose=True)
        new_model.fit(new_data['text'].tolist())
        
        # Merge models
        merged_model = base_model.merge_models([new_model])
        
        end_time = time.time()
        peak_memory = self.get_memory_usage()
        
        return {
            'peak_memory': peak_memory,
            'memory_used': peak_memory - initial_memory,
            'time_taken': end_time - start_time,
            'model': merged_model
        }

    def run_incremental_experiment(self, base_model: BERTopic, new_data: pd.DataFrame) -> Dict[str, float]:
        """Run incremental processing and measure resources"""
        gc.collect()
        initial_memory = self.get_memory_usage()
        start_time = time.time()
        
        model = SentenceTransformer("all-miniLM-L6-v2")
        umap_model = UMAP(
            n_components=8,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Train model on new data
        new_model = BERTopic(embedding_model=model,
                           umap_model=umap_model,
                           verbose=True)
        new_model.fit(new_data['text'].tolist())
        
        # Merge with existing model
        merged_model = base_model.merge_models([new_model])
        
        end_time = time.time()
        peak_memory = self.get_memory_usage()
        
        return {
            'peak_memory': peak_memory,
            'memory_used': peak_memory - initial_memory,
            'time_taken': end_time - start_time,
            'model': merged_model
        }

    def run_experiments(self, n_runs: int=5):
        """Run complete experiment suite"""
        data_2018, data_jan_2019, data_feb_2019 = self.load_data()
        
        for run in range(n_runs):
            try:
                print(f"\nStarting Run {run + 1}/{n_runs}")

                # Scenario 1: 2018 + January 2019
                print("Running Scenario 1: 2018 + January 2019")

                # Batch approach
                combined_data = pd.concat([data_2018, data_jan_2019])
                batch_results = self.run_batch_experiment(combined_data)
                self.results['batch']['memory'].append(batch_results['memory_used'])
                self.results['batch']['time'].append(batch_results['time_taken'])

                # CHANGE THIS PART
                if isinstance(batch_results, dict):
                    self.results['batch']['memory'].append(batch_results['memory_used'])
                    self.results['batch']['time'].append(batch_results['time_taken'])
                else:
                    print(f"Error: batch_results is not a dictionary, it's a {type(batch_results)}")
                    return
                
                # Incremental approach
                incr_results = self.run_first_incremental(data_2018, data_jan_2019)
                self.results['incremental']['memory'].append(incr_results['memory_used'])
                self.results['incremental']['time'].append(incr_results['time_taken'])
                merged_model = incr_results['model']
                
                # Scenario 2: Adding February 2019
                print("Running Scenario 2: Adding February 2019")

                # Batch approach
                combined_data = pd.concat([data_2018, data_jan_2019, data_feb_2019])
                batch_results = self.run_batch_experiment(combined_data)
                self.results['batch']['memory'].append(batch_results['memory_used'])
                self.results['batch']['time'].append(batch_results['time_taken'])
                
                # Incremental approach - use stored merged model
                incr_results = self.run_incremental_experiment(merged_model, data_feb_2019)
                self.results['incremental']['memory'].append(incr_results['memory_used'])
                self.results['incremental']['time'].append(incr_results['time_taken'])
                
                # Save results after each complete run
                self.save_run_results(run + 1)
                
                # Clean up
                del merged_model
                gc.collect()
                
            except Exception as e:
                print(f"Error in run {run + 1}: {str(e)}")
                if any(len(v['memory']) > 0 for v in self.results.values()):
                    self.save_run_results(run + 1)
                raise

    def get_summary_statistics(self) -> Dict:
        return {
            'batch': {
                'memory_mean': np.mean(self.results['batch']['memory']),
                'memory_std': np.std(self.results['batch']['memory']),
                'time_mean': np.mean(self.results['batch']['time']),
                'time_std': np.std(self.results['batch']['time'])
            },
            'incremental': {
                'memory_mean': np.mean(self.results['incremental']['memory']),
                'memory_std': np.std(self.results['incremental']['memory']),
                'time_mean': np.mean(self.results['incremental']['time']),
                'time_std': np.std(self.results['incremental']['time'])
            }
        }

if __name__ == "__main__":
    base_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data"
    experiment = ResourceExperiment(base_path)
    
    try:
        experiment.run_experiments(n_runs=5)
        stats = experiment.get_summary_statistics()
        print("\nSummary Statistics:")
        print(stats)
        
        # Save final summary
        with open(os.path.join(experiment.results_dir, 'final_summary.json'), 'w') as f:
            json.dump(stats, f, indent=4)
            
    except Exception as e:
        print(f"Experiment failed: {str(e)}")