import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
import psutil
import os
import time
from typing import Dict, List, Tuple, Union, Optional
import gc
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import traceback
import torch
import matplotlib.pyplot as plt
from pathlib import Path

torch.cuda.set_device(3)

class BERTopicMemoryComparison:
    def __init__(self, base_path: str):
        """Initialize BERTopic Memory Comparison.
        
        Args:
            base_path: Base directory for output files
        """
        self.base_path = Path(base_path)
        
        # Set up directories
        self.results_dir = self.base_path  # Changed to use base_path directly
        self.visualization_dir = self.results_dir / 'visualizations'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Add model configuration
        self.model_config = {
        'embedding_model': "all-miniLM-L6-v2",
        'umap_components': 8,
        'umap_neighbors': 15,
        'umap_min_dist': 0.0,
        'random_state': 42
        }

        # Initialize results structure
        self.results = {
            'batch': {
                'peak_memory': [],  # Peak memory during batch training
                'gpu_memory': [],
                'training_time': []
            },
            'simultaneous_merge': {
                'training': {
                    'peak_memory': [],  # Peak memory during individual trainings
                    'gpu_memory': [],
                    'time': []
                },
                'merge': {
                    'peak_memory': [],  # Peak memory during merge operation
                    'gpu_memory': [],
                    'time': []
                }
            }
        }

    def _initialize_model(self) -> Tuple[SentenceTransformer, UMAP]:
        model = SentenceTransformer(self.model_config['embedding_model'])
        umap_model = UMAP(
            n_components=self.model_config['umap_components'],
            n_neighbors=self.model_config['umap_neighbors'],
            min_dist=self.model_config['umap_min_dist'],
            metric='cosine',
            random_state=self.model_config['random_state']
        )
        return model, umap_model

    def get_memory_usage(self) -> float:
        """Get current CPU memory usage in GB."""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        return memory_gb

    def get_gpu_memory_usage(self) -> Optional[List[Dict]]:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return [{
                'device': i,
                'allocated': torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024,
                'reserved': torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            } for i in range(torch.cuda.device_count())]
        return None

    def load_yearly_data(self, year: int) -> pd.DataFrame:
        """Load data for a specific year."""
        # Adjust the path to use the correct data directory
        filepath = Path("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/yearly_england") / f"nltk_stopwords_preprocessed_england_speeches_{year}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file for year {year} not found: {filepath}")
        
        data = pd.read_csv(filepath)
        return data[['text']].dropna()

    def run_batch_experiment(self, start_year: int = 2000, end_year: int = 2019) -> Dict:
        """Run and measure batch processing approach."""
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load all data
        print("Loading all data for batch processing...")
        all_data = pd.concat([
            self.load_yearly_data(year) for year in range(start_year, end_year + 1)
        ])
        
        initial_memory = self.get_memory_usage()
        initial_gpu = self.get_gpu_memory_usage()
        start_time = time.time()
        peak_memory = initial_memory
        
        try:
            # Initialize and train model
            model, umap_model = self._initialize_model()
            topic_model = BERTopic(
                embedding_model=model,
                umap_model=umap_model,
                verbose=True
            )
            
            # Track memory during training
            topics, _ = topic_model.fit_transform(all_data['text'].tolist())
            current_memory = self.get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
            
            final_gpu = self.get_gpu_memory_usage()
            total_time = time.time() - start_time
            
            return {
                'peak_memory': peak_memory,
                'memory_increase': peak_memory - initial_memory,
                'gpu_memory': final_gpu,
                'training_time': total_time
            }
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            traceback.print_exc()
            raise

    def run_merge(self, start_year: int = 2000, end_year: int = 2019) -> Dict:
        """Run and measure memory usage with progressive merging approach."""
        gc.collect()
        torch.cuda.empty_cache()
        
        temp_dir = self.results_dir / 'temp_models'
        temp_dir.mkdir(exist_ok=True)
        training_results = []
        current_merged_model = None
        overall_peak_memory = 0
        
        try:
            # Train and merge models progressively
            for year in range(start_year, end_year + 1):
                print(f"\nTraining model for year {year}")
                initial_memory = self.get_memory_usage()
                start_time = time.time()
                
                # Train model for current year
                current_data = self.load_yearly_data(year)
                model, umap_model = self._initialize_model()
                
                year_model = BERTopic(
                    embedding_model=model,
                    umap_model=umap_model,
                    verbose=True
                )
                
                # Train and track memory
                topics, _ = year_model.fit_transform(current_data['text'].tolist())
                current_memory = self.get_memory_usage()
                peak_memory = max(current_memory, initial_memory)
                overall_peak_memory = max(overall_peak_memory, peak_memory)
                
                training_results.append({
                    'year': year,
                    'peak_memory': peak_memory,
                    'memory_increase': peak_memory - initial_memory,
                    'training_time': time.time() - start_time
                })
                
                # Merge with existing model if it exists
                if current_merged_model is None:
                    current_merged_model = year_model
                else:
                    print(f"Merging model for year {year}")
                    merge_start_time = time.time()
                    initial_merge_memory = self.get_memory_usage()
                    
                    # Perform merge
                    current_merged_model = BERTopic.merge_models(
                        [current_merged_model, year_model],
                        min_similarity=0.7
                    )
                    
                    current_memory = self.get_memory_usage()
                    merge_peak_memory = max(current_memory, initial_merge_memory)
                    overall_peak_memory = max(overall_peak_memory, merge_peak_memory)
                    
                    training_results[-1].update({
                        'merge_memory': merge_peak_memory,
                        'merge_time': time.time() - merge_start_time
                    })
                
                # Clear memory
                del year_model
                del model
                del umap_model
                gc.collect()
                torch.cuda.empty_cache()
            
            return {
                'training_results': training_results,
                'overall_peak_memory': overall_peak_memory
            }
            
        except Exception as e:
            print(f"Error in simultaneous merge processing: {e}")
            traceback.print_exc()
            raise

    def _store_results(self, batch_results: Dict, merge_results: Dict):
        """Store results from both approaches."""
        # Store batch results
        self.results['batch']['peak_memory'].append(batch_results['peak_memory'])
        self.results['batch']['gpu_memory'].append(batch_results['gpu_memory'])
        self.results['batch']['training_time'].append(batch_results['training_time'])
        
        # Store merge results
        training_results = merge_results['training_results']
        max_training_memory = max(r['peak_memory'] for r in training_results)
        merge_memory = max(r.get('merge_memory', 0) for r in training_results)  # Use get() with default
        merge_time = sum(r.get('merge_time', 0) for r in training_results)  # Sum up all merge times
        
        self.results['simultaneous_merge']['training']['peak_memory'].append(max_training_memory)
        self.results['simultaneous_merge']['merge']['peak_memory'].append(merge_memory)
        self.results['simultaneous_merge']['merge']['time'].append(merge_time)

    def visualize_memory_comparison(self):
        """Create visualization comparing memory usage between approaches."""
        plt.figure(figsize=(12, 6))
        
        # Calculate averages
        batch_avg = np.mean(self.results['batch']['peak_memory'])
        training_avg = np.mean(self.results['simultaneous_merge']['training']['peak_memory'])
        merge_avg = np.mean(self.results['simultaneous_merge']['merge']['peak_memory'])
        
        # Create bar plot
        bars = plt.bar([0, 1, 2], 
                      [batch_avg, training_avg, merge_avg],
                      color=['blue', 'green', 'red'],
                      alpha=0.6)
        
        # Customize plot
        plt.title('Peak Memory Usage Comparison', fontsize=14)
        plt.ylabel('Memory Usage (GB)', fontsize=12)
        plt.xticks([0, 1, 2], ['Batch Processing', 'Individual Training', 'Merge Operation'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} GB',
                    ha='center', va='bottom')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(self.visualization_dir / 'memory_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()

    def run_experiment(self, n_runs: int = 3):
        """Run complete experiment comparing both approaches."""
        for run in range(n_runs):
            print(f"\nStarting Run {run + 1}/{n_runs}")
            
            print("\nRunning Batch Approach...")
            batch_results = self.run_batch_experiment()
            
            print("\nRunning Simultaneous Merge Approach...")
            merge_results = self.run_merge()
            
            self._store_results(batch_results, merge_results)
            
            # Save results after each run
            self.save_run_results(run + 1)
            
            # Create visualization
            self.visualize_memory_comparison()
            
            # Clear memory between runs
            gc.collect()
            torch.cuda.empty_cache()

    def save_run_results(self, run_number: int):
        """Save detailed results for each run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.results_dir / f'run_{run_number}_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)

def main():
    try:
        print("Starting memory analysis...")
        # Set paths
        DATA_DIR = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/yearly_england"
        OUTPUT_DIR = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_4/Iteration_8_incremental"
        
        print(f"Using output directory: {OUTPUT_DIR}")
        
        # Create evaluator
        comparison = BERTopicMemoryComparison(OUTPUT_DIR)
        comparison.run_experiment(n_runs=3)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()