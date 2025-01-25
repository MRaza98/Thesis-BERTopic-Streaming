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
        self.base_path = Path(base_path)
        # Simplified results structure focused on memory comparison
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
        
        # Set up directories
        self.results_dir = self.base_path / 'memory_comparison_results'
        self.visualization_dir = self.results_dir / 'visualizations'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_config = {
            'embedding_model': "all-miniLM-L6-v2",
            'umap_components': 8,
            'umap_neighbors': 15,
            'umap_min_dist': 0.0,
            'random_state': 42
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
        filepath = self.base_path / f"data/yearly_england/nltk_stopwords_preprocessed_england_speeches_{year}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file for year {year} not found: {filepath}")
        
        data = pd.read_csv(filepath, dtype={'text': str})
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

    def run_simultaneous_merge(self, start_year: int = 2000, end_year: int = 2019) -> Dict:
        """Run and measure simultaneous merge approach with disk storage to minimize memory usage."""
        gc.collect()
        torch.cuda.empty_cache()
        
        temp_dir = self.results_dir / 'temp_models'
        temp_dir.mkdir(exist_ok=True)
        model_paths = []
        training_results = []
        overall_peak_memory = 0
        
        try:
            # Phase 1: Train individual models and save to disk
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
                
                # Save model to disk and clear from memory
                model_path = temp_dir / f'model_{year}'
                year_model.save(model_path)
                model_paths.append(model_path)
                
                training_results.append({
                    'year': year,
                    'peak_memory': peak_memory,
                    'memory_increase': peak_memory - initial_memory,
                    'training_time': time.time() - start_time
                })
                
                # Clear memory after saving
                del year_model
                del model
                del umap_model
                gc.collect()
                torch.cuda.empty_cache()
            
            # Phase 2: Load models and merge
            print("\nLoading models for merge operation...")
            merge_start_time = time.time()
            initial_merge_memory = self.get_memory_usage()
            
            # Load all models
            models_to_merge = []
            for path in model_paths:
                models_to_merge.append(BERTopic.load(path))
                
            # Track memory after loading
            post_load_memory = self.get_memory_usage()
            print(f"Memory after loading models: {post_load_memory:.2f} GB")
            
            # Perform merge
            merged_model = BERTopic.merge_models(models_to_merge)
            current_memory = self.get_memory_usage()
            merge_peak_memory = max(current_memory, post_load_memory)
            overall_peak_memory = max(overall_peak_memory, merge_peak_memory)
            
            merge_results = {
                'peak_memory': merge_peak_memory,
                'memory_increase': merge_peak_memory - initial_merge_memory,
                'merge_time': time.time() - merge_start_time,
                'load_memory': post_load_memory - initial_merge_memory
            }
            
            return {
                'training_results': training_results,
                'merge_results': merge_results,
                'overall_peak_memory': overall_peak_memory
            }
            
        finally:
            # Clean up temporary files
            for path in model_paths:
                try:
                    if path.exists():
                        if path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                except Exception as e:
                    print(f"Error cleaning up {path}: {e}")
            
            try:
                if temp_dir.exists():
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Error removing temp directory: {e}")

    def _store_results(self, batch_results: Dict, merge_results: Dict):
        """Store results from both approaches."""
        # Store batch results
        self.results['batch']['peak_memory'].append(batch_results['peak_memory'])
        self.results['batch']['gpu_memory'].append(batch_results['gpu_memory'])
        self.results['batch']['training_time'].append(batch_results['training_time'])
        
        # Store simultaneous merge results
        max_training_memory = max(r['peak_memory'] for r in merge_results['training_results'])
        self.results['simultaneous_merge']['training']['peak_memory'].append(max_training_memory)
        self.results['simultaneous_merge']['merge']['peak_memory'].append(merge_results['merge_results']['peak_memory'])
        self.results['simultaneous_merge']['merge']['time'].append(merge_results['merge_results']['merge_time'])

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
            merge_results = self.run_simultaneous_merge()
            
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

if __name__ == "__main__":
    base_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments"
    comparison = BERTopicMemoryComparison(base_path)
    comparison.run_experiment(n_runs=3)