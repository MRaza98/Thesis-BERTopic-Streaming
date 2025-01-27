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
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class ResourceExperiment:
    
    def __init__(self, base_path: str):

        self.base_path = Path(base_path)
        self.results = {
            'batch': {
                'cpu_memory': [], 
                'gpu_memory': [],
                'time': []
            },
            'incremental': {
                'cpu_memory': [], 
                'gpu_memory': [],
                'time': []
            }
        }
        
        # Set up directories
        self.results_dir = self.base_path / 'Experiment_5/Iteration_5_Corrected_Indent'
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
        
        # Set up plotting style
        plt.style.use('seaborn')
        
        # Resource warning thresholds
        self.memory_warning_threshold = 0.9  # 90% of available memory
    
    def _initialize_model(self) -> Tuple[SentenceTransformer, UMAP]:
        """Initialize embedding and UMAP models with configured parameters."""
        model = SentenceTransformer(self.model_config['embedding_model'])
        umap_model = UMAP(
            n_components=self.model_config['umap_components'],
            n_neighbors=self.model_config['umap_neighbors'],
            min_dist=self.model_config['umap_min_dist'],
            metric='cosine',
            random_state=self.model_config['random_state']
        )
        return model, umap_model
    
    def load_yearly_data(self, year: int) -> pd.DataFrame:
            """
            Load and validate data for a specific year.
            
            Args:
                year (int): Year to load data for
                
            Returns:
                pd.DataFrame: Loaded and validated data
                
            Raises:
                FileNotFoundError: If data file doesn't exist
                ValueError: If data validation fails
            """
            filepath = self.base_path / f"data/yearly_england/nltk_stopwords_preprocessed_england_speeches_{year}.csv"
            
            if not filepath.exists():
                raise FileNotFoundError(f"Data file for year {year} not found: {filepath}")
            
            data = pd.read_csv(filepath, dtype={'text': str})
            data['text'] = data['text'].astype(str)
            
            # Validate data
            if data.empty:
                raise ValueError(f"Empty dataset for year {year}")
            if data['text'].isnull().any():
                raise ValueError(f"Dataset for year {year} contains null values")
                
            return data

    def create_visualizations(self):
        """Create and save all visualization plots"""
        self._plot_memory_usage_over_time()
        self._plot_batch_vs_incremental()
        self._plot_comparative_memory_usage()
        plt.close('all')
    
    def _plot_comparative_memory_usage(self):
        """Create comparison plot of memory usage between batch and incremental approaches"""
        stats = self.get_summary_statistics()
        
        plt.figure(figsize=(15, 8))
        
        # Prepare data
        batch_memory = [
            stats['batch']['initial']['cpu_memory_mean'],
            stats['batch']['full']['cpu_memory_mean']
        ]
        total_incremental_memory = stats['incremental']['overall']['cpu_memory_mean']
        
        # Create grouped bar plot
        positions = np.arange(3)
        plt.bar(positions[0:2], batch_memory, 
                alpha=0.8, label='Batch Processing',
                color='skyblue')
        plt.bar(positions[2], total_incremental_memory,
                alpha=0.8, label='Incremental Processing',
                color='lightgreen')
        
        # Customize plot
        plt.title('Memory Usage Comparison: Batch vs Incremental', fontsize=14)
        plt.ylabel('Memory Usage (GB)', fontsize=12)
        plt.xticks(positions, ['Batch\n(2000-2018)', 'Batch\n(2000-2019)', 'Incremental\n(2000-2019)'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(batch_memory + [total_incremental_memory]):
            plt.text(positions[i], v, f'{v:.2f} GB',
                    ha='center', va='bottom')
        
        plt.savefig(
            self.visualization_dir / 'memory_usage_comparison.png',
            bbox_inches='tight',
            dpi=300
        )

    def _plot_memory_usage_over_time(self):
        """Create line plot of memory usage over time"""
        years = list(range(2000, 2020))
        stats = self.get_summary_statistics()
        
        # Prepare data
        incremental_memory = [stats['incremental']['by_year'][str(year)]['cpu_memory_mean'] for year in years]
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, incremental_memory, marker='o', label='Incremental Approach')
        
        plt.title('CPU Memory Usage Over Time', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Memory Usage (GB)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.visualization_dir, 'memory_usage_over_time.png'), 
                    bbox_inches='tight', dpi=300)
    
    def print_gpu_memory_stats(self, stage: str):
        """Print GPU memory statistics at a given stage"""
        if torch.cuda.is_available():
            print(f"\nGPU Memory at {stage}:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
                print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def get_memory_usage(self) -> float:
        """Get current CPU memory usage in GB."""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        return memory_gb

    def get_gpu_memory_usage(self) -> Optional[List[Dict]]:
        """Get current GPU memory usage in GB for all GPUs."""
        if torch.cuda.is_available():
            return [{
                'device': i,
                'allocated': torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024,
                'reserved': torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            } for i in range(torch.cuda.device_count())]
        return None

    def run_batch_experiment(self, data: pd.DataFrame, stage: str) -> Dict[str, Union[float, Dict]]:
        """
        Run batch processing and measure resources.
        
        Args:
            data (pd.DataFrame): Data to process
            stage (str): Description of current processing stage
            
        Returns:
            Dict: Results including memory and time measurements
        """
        gc.collect()
        torch.cuda.empty_cache()
        
        initial_cpu_memory = self.get_memory_usage()
        initial_gpu_memory = self.get_gpu_memory_usage()
        start_time = time.time()

        try:
            # Initialize models
            model, umap_model = self._initialize_model()
            topic_model = BERTopic(
                embedding_model=model,
                umap_model=umap_model,
                verbose=True
            )
            
            # Train model
            topics, probs = topic_model.fit_transform(data['text'].tolist())
            
            end_time = time.time()
            peak_cpu_memory = self.get_memory_usage()
            peak_gpu_memory = self.get_gpu_memory_usage()

            results_dict = self._calculate_resource_usage(
                initial_cpu_memory, peak_cpu_memory,
                initial_gpu_memory, peak_gpu_memory,
                start_time, end_time
            )
            
            print(f"{stage} results:", results_dict)
            return results_dict

        except Exception as e:
            error_msg = f"Error in batch experiment ({stage}): {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise

    def run_incremental_model(
        self, 
        data: pd.DataFrame, 
        base_model: Optional[BERTopic] = None
    ) -> Tuple[BERTopic, Dict[str, Union[float, Dict]]]:
        
        gc.collect()
        torch.cuda.empty_cache()
        
        initial_cpu_memory = self.get_memory_usage()
        initial_gpu_memory = self.get_gpu_memory_usage()
        start_time = time.time()

        try:
            model, umap_model = self._initialize_model()

            if base_model is None:
                # First model
                new_model = BERTopic(
                    embedding_model=model,
                    umap_model=umap_model,
                    verbose=True
                )
                new_model.fit(data['text'].tolist())
                result_model = new_model
            else:
                # Create and merge new model
                new_model = BERTopic(
                    embedding_model=model,
                    umap_model=umap_model,
                    verbose=True
                )
                new_model.fit(data['text'].tolist())
                result_model = base_model.merge_models([new_model])

            end_time = time.time()
            peak_cpu_memory = self.get_memory_usage()
            peak_gpu_memory = self.get_gpu_memory_usage()

            results = self._calculate_resource_usage(
                initial_cpu_memory, peak_cpu_memory,
                initial_gpu_memory, peak_gpu_memory,
                start_time, end_time
            )

            return result_model, results

        except Exception as e:
            error_msg = f"Error in incremental model: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise
    
    def _calculate_resource_usage(
        self,
        initial_cpu_memory: float,
        peak_cpu_memory: float,
        initial_gpu_memory: Optional[List[Dict]],
        peak_gpu_memory: Optional[List[Dict]],
        start_time: float,
        end_time: float
    ) -> Dict[str, Union[float, Dict]]:
        """Calculate resource usage from measurements."""
        gpu_memory_used = None
        if initial_gpu_memory and peak_gpu_memory:
            gpu_memory_used = {
                f'gpu_{i}': {
                    'allocated_diff': peak_gpu_memory[i]['allocated'] - initial_gpu_memory[i]['allocated'],
                    'reserved_diff': peak_gpu_memory[i]['reserved'] - initial_gpu_memory[i]['reserved']
                }
                for i in range(len(peak_gpu_memory))
            }

        return {
            'cpu_peak_memory': float(peak_cpu_memory),
            'cpu_memory_used': float(peak_cpu_memory - initial_cpu_memory),
            'gpu_memory': gpu_memory_used,
            'time_taken': float(end_time - start_time)
        }

    def run_experiments(self, n_runs: int = 3):
        """
        Run complete experiment suite comparing batch and incremental approaches.
        
        Args:
            n_runs (int): Number of experiment runs
        """
        if n_runs < 1:
            raise ValueError("Number of runs must be positive")

        for run in range(n_runs):
            try:
                print(f"\nStarting Run {run + 1}/{n_runs}")
                
                # BATCH APPROACH
                print("\nRunning Batch Approach")
                
                # First batch: 2000-2018
                print("Processing 2000-2018 batch")
                historical_data = pd.concat([
                    self.load_yearly_data(year) for year in range(2000, 2019)
                ])
                batch_results = self.run_batch_experiment(
                    historical_data, 
                    "2000-2018 batch"
                )
                self._store_results('batch', batch_results)
                
                # INCREMENTAL APPROACH
                print("\nRunning Incremental Approach")
                merged_model = None
                
                # Process each year incrementally
                for year in range(2000, 2020):
                    print(f"\nProcessing year {year}")
                    current_data = self.load_yearly_data(year)
                    merged_model, incr_results = self.run_incremental_model(
                        current_data, 
                        merged_model
                    )
                    self._store_results('incremental', incr_results)
                
                # Second batch: 2000-2019
                print("\nProcessing 2000-2019 batch")
                full_data = pd.concat([
                    self.load_yearly_data(year) for year in range(2000, 2020)
                ])
                batch_results = self.run_batch_experiment(
                    full_data,
                    "2000-2019 batch"
                )
                self._store_results('batch', batch_results)
                
                # Save results and create visualizations
                self.save_run_results(run + 1)
                self.create_visualizations()
                
                # Cleanup
                del merged_model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                error_msg = f"Error in run {run + 1}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                raise

    def save_run_results(self, run_number: int):
        """Save results after each run"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'run_{run_number}_{timestamp}.json'
        filepath = self.results_dir / filename
        
        # Calculate yearly statistics for incremental approach
        incremental_stats = []
        years = list(range(2000, 2020))
        for i in range(len(years)):
            idx = i + (run_number - 1) * len(years)
            if idx < len(self.results['incremental']['cpu_memory']):
                year_stats = {
                    'year': years[i],
                    'cpu_memory': self.results['incremental']['cpu_memory'][idx],
                    'gpu_memory': self.results['incremental']['gpu_memory'][idx],
                    'time': self.results['incremental']['time'][idx]
                }
                incremental_stats.append(year_stats)
        
        # Get batch results for this run
        batch_stats = {
            'initial': {  # 2000-2018
                'cpu_memory': self.results['batch']['cpu_memory'][-2],
                'gpu_memory': self.results['batch']['gpu_memory'][-2],
                'time': self.results['batch']['time'][-2]
            },
            'full': {     # 2000-2019
                'cpu_memory': self.results['batch']['cpu_memory'][-1],
                'gpu_memory': self.results['batch']['gpu_memory'][-1],
                'time': self.results['batch']['time'][-1]
            }
        }
        
        run_results = {
            'run_number': run_number,
            'timestamp': timestamp,
            'batch_results': batch_stats,
            'incremental_results': incremental_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(run_results, f, indent=4)
        print(f"Results for run {run_number} saved to {filepath}")
    
    def _store_results(self, approach: str, results: Dict):
        """Store results from an experiment run."""
        self.results[approach]['cpu_memory'].append(results['cpu_memory_used'])
        self.results[approach]['gpu_memory'].append(results['gpu_memory'])
        self.results[approach]['time'].append(results['time_taken'])

    def _plot_batch_vs_incremental(self):
        """Create comparison plot between batch and incremental approaches."""
        stats = self.get_summary_statistics()
        
        plt.figure(figsize=(15, 8))
        
        # Plot batch results
        batch_times = [
            stats['batch']['initial']['time_mean'],
            stats['batch']['full']['time_mean']
        ]
        plt.bar(['Batch 2000-2018', 'Batch 2000-2019'], 
                batch_times, 
                alpha=0.8,
                label='Batch Processing')
        
        # Plot incremental results
        total_incremental_time = stats['incremental']['overall']['time_mean']
        plt.bar('Incremental 2000-2019', 
                total_incremental_time,
                alpha=0.8,
                label='Incremental Processing')
        
        plt.title('Processing Time Comparison: Batch vs Incremental', 
                 fontsize=14)
        plt.ylabel('Processing Time (seconds)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(
            self.visualization_dir / 'batch_vs_incremental_comparison.png',
            bbox_inches='tight',
            dpi=300
        )

    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics including GPU memory for both batch and incremental approaches"""
        years = list(range(2000, 2020))
        stats = {
            'batch': {
                'initial': {  # 2000-2018
                    'cpu_memory_mean': np.mean(self.results['batch']['cpu_memory'][::2]),
                    'cpu_memory_std': np.std(self.results['batch']['cpu_memory'][::2]),
                    'time_mean': np.mean(self.results['batch']['time'][::2]),
                    'time_std': np.std(self.results['batch']['time'][::2]),
                    'gpu_memory_stats': {}
                },
                'full': {     # 2000-2019
                    'cpu_memory_mean': np.mean(self.results['batch']['cpu_memory'][1::2]),
                    'cpu_memory_std': np.std(self.results['batch']['cpu_memory'][1::2]),
                    'time_mean': np.mean(self.results['batch']['time'][1::2]),
                    'time_std': np.std(self.results['batch']['time'][1::2]),
                    'gpu_memory_stats': {}
                }
            },
            'incremental': {
                'by_year': {},
                'overall': {
                    'cpu_memory_mean': np.mean(self.results['incremental']['cpu_memory']),
                    'cpu_memory_std': np.std(self.results['incremental']['cpu_memory']),
                    'time_mean': np.mean(self.results['incremental']['time']),
                    'time_std': np.std(self.results['incremental']['time']),
                    'gpu_memory_stats': {}
                }
            }
        }
        
        # Calculate per-year statistics for incremental approach
        n_runs = len(self.results['incremental']['cpu_memory']) // len(years)
        for year_idx, year in enumerate(years):
            year_indices = [year_idx + i * len(years) for i in range(n_runs)]
            
            year_stats = {
                'cpu_memory_mean': np.mean([self.results['incremental']['cpu_memory'][i] for i in year_indices]),
                'cpu_memory_std': np.std([self.results['incremental']['cpu_memory'][i] for i in year_indices]),
                'time_mean': np.mean([self.results['incremental']['time'][i] for i in year_indices]),
                'time_std': np.std([self.results['incremental']['time'][i] for i in year_indices])
            }
            
            stats['incremental']['by_year'][str(year)] = year_stats
        
        # Calculate GPU memory statistics
        if self.results['batch']['gpu_memory']:
            # For batch initial
            gpu_stats_initial = {}
            gpu_stats_full = {}
            for gpu_idx in range(torch.cuda.device_count()):
                gpu_key = f'gpu_{gpu_idx}'
                
                # Initial batch (2000-2018)
                allocated_diffs_initial = [mem[gpu_key]['allocated_diff'] 
                                for mem in self.results['batch']['gpu_memory'][::2] if mem]
                reserved_diffs_initial = [mem[gpu_key]['reserved_diff'] 
                                for mem in self.results['batch']['gpu_memory'][::2] if mem]
                
                gpu_stats_initial[gpu_key] = {
                    'allocated_mean': float(np.mean(allocated_diffs_initial)),
                    'allocated_std': float(np.std(allocated_diffs_initial)),
                    'reserved_mean': float(np.mean(reserved_diffs_initial)),
                    'reserved_std': float(np.std(reserved_diffs_initial))
                }
                
                # Full batch (2000-2019)
                allocated_diffs_full = [mem[gpu_key]['allocated_diff'] 
                                for mem in self.results['batch']['gpu_memory'][1::2] if mem]
                reserved_diffs_full = [mem[gpu_key]['reserved_diff'] 
                                for mem in self.results['batch']['gpu_memory'][1::2] if mem]
                
                gpu_stats_full[gpu_key] = {
                    'allocated_mean': float(np.mean(allocated_diffs_full)),
                    'allocated_std': float(np.std(allocated_diffs_full)),
                    'reserved_mean': float(np.mean(reserved_diffs_full)),
                    'reserved_std': float(np.std(reserved_diffs_full))
                }
            
            stats['batch']['initial']['gpu_memory_stats'] = gpu_stats_initial
            stats['batch']['full']['gpu_memory_stats'] = gpu_stats_full
        
        if self.results['incremental']['gpu_memory']:
            gpu_stats = {}
            for gpu_idx in range(torch.cuda.device_count()):
                gpu_key = f'gpu_{gpu_idx}'
                allocated_diffs = [mem[gpu_key]['allocated_diff'] 
                                for mem in self.results['incremental']['gpu_memory'] if mem]
                reserved_diffs = [mem[gpu_key]['reserved_diff'] 
                                for mem in self.results['incremental']['gpu_memory'] if mem]
                
                gpu_stats[gpu_key] = {
                    'allocated_mean': float(np.mean(allocated_diffs)),
                    'allocated_std': float(np.std(allocated_diffs)),
                    'reserved_mean': float(np.mean(reserved_diffs)),
                    'reserved_std': float(np.std(reserved_diffs))
                }
            stats['incremental']['overall']['gpu_memory_stats'] = gpu_stats
        
        return stats

if __name__ == "__main__":
    base_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments"
    experiment = ResourceExperiment(base_path)
    
    try:
        experiment.run_experiments(n_runs=3)
        stats = experiment.get_summary_statistics()
        print("\nSummary Statistics:")
        print(json.dumps(stats, indent=4))
        
        # Save final summary
        with open(os.path.join(experiment.results_dir, 'final_summary.json'), 'w') as f:
            json.dump(stats, f, indent=4)
            
    except Exception as e:
        print(f"Experiment failed: {str(e)}")