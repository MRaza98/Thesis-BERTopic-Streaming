# experiment_tracker.py
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import socket
from typing import Any, Dict, Optional, List
import pandas as pd
import logging
import torch
import time
import hashlib

@dataclass
class ModelExperiment:
    """Data class to store experiment information"""
    experiment_id: str
    model_name: str
    embedding_model: str
    num_topics: int
    dimension_reduction: str  # 'umap', 'kpca', or 'pca'
    clustering_method: str    # 'hdbscan' or 'kmeans'
    n_components: int
    preprocessing_method: str  # 'full_speeches', 'sentences', or 'no_stopwords'
    batch_size: Optional[int] = None
    num_documents: Optional[int] = None
    avg_document_length: Optional[float] = None  # Average length in words
    duration_seconds: Optional[float] = None
    gpu_device: Optional[str] = None
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hostname: str = socket.gethostname()
    clustering_params: Dict[str, Any] = None  # Specific clustering parameters
    reduction_params: Dict[str, Any] = None   # Specific dimension reduction parameters
    preprocessing_stats: Dict[str, Any] = None  # Additional preprocessing statistics

def generate_experiment_id(embedding_model: str, dimension_reduction: str, 
                         clustering_method: str, num_topics: int, 
                         n_components: int, preprocessing_method: str,
                         timestamp: str) -> str:
    """Generate a unique, readable experiment ID"""
    # Create a base ID from main parameters
    base_id = f"{embedding_model.replace('-', '_')}_{dimension_reduction}_{clustering_method}_{preprocessing_method}_{num_topics}topics_{n_components}dim_{timestamp}"
    
    # Create a short hash for uniqueness
    hash_suffix = hashlib.md5(base_id.encode()).hexdigest()[:6]
    
    return f"{base_id}_{hash_suffix}"

def detect_clustering_method(model):
    """Detect the clustering method and its parameters"""
    if hasattr(model, 'hdbscan_model'):
        if model.hdbscan_model is not None:
            return {
                'method': 'hdbscan',
                'params': {
                    'min_cluster_size': getattr(model.hdbscan_model, 'min_cluster_size', None),
                    'min_samples': getattr(model.hdbscan_model, 'min_samples', None),
                    'cluster_selection_epsilon': getattr(model.hdbscan_model, 'cluster_selection_epsilon', None)
                }
            }
    if hasattr(model, 'kmeans_model'):
        if model.kmeans_model is not None:
            return {
                'method': 'kmeans',
                'params': {
                    'n_clusters': model.nr_topics,
                    'random_state': getattr(model.kmeans_model, 'random_state', None),
                    'n_init': getattr(model.kmeans_model, 'n_init', None)
                }
            }
    return {
        'method': 'unknown',
        'params': {}
    }

def detect_dimension_reduction(model):
    """Detect the dimension reduction method and its parameters"""
    umap_model = model.umap_model
    
    if hasattr(umap_model, 'n_neighbors'):
        return {
            'method': 'umap',
            'params': {
                'n_components': umap_model.n_components,
                'n_neighbors': umap_model.n_neighbors,
                'min_dist': umap_model.min_dist,
                'metric': umap_model.metric,
                'random_state': umap_model.random_state
            }
        }
    elif hasattr(umap_model, 'kernel'):
        return {
            'method': 'kpca',
            'params': {
                'n_components': umap_model.n_components,
                'kernel': umap_model.kernel,
                'random_state': getattr(umap_model, 'random_state', None)
            }
        }
    elif hasattr(umap_model, 'explained_variance_ratio_'):
        return {
            'method': 'pca',
            'params': {
                'n_components': umap_model.n_components_,
                'explained_variance_ratio': umap_model.explained_variance_ratio_.sum(),
                'random_state': getattr(umap_model, 'random_state', None)
            }
        }
    return {
        'method': 'unknown',
        'params': {
            'n_components': getattr(umap_model, 'n_components', None)
        }
    }

def calculate_document_stats(documents: List[str]) -> Dict[str, Any]:
    """Calculate statistics about the document collection"""
    doc_lengths = [len(doc.split()) for doc in documents]
    return {
        "total_documents": len(documents),
        "avg_document_length": sum(doc_lengths) / len(documents),
        "min_document_length": min(doc_lengths),
        "max_document_length": max(doc_lengths),
        "total_words": sum(doc_lengths)
    }

def get_device_info(model):
    """Get information about the device being used"""
    try:
        if hasattr(model, 'device'):
            return str(model.device)
        elif hasattr(model, 'parameters'):
            return str(next(model.parameters()).device)
        return str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    except Exception as e:
        return "unknown"

class ExperimentTracker:
    def __init__(self, output_dir: str = "experiment_tracking"):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "model_experiments.csv")
        self.json_path = os.path.join(output_dir, "model_experiments.json")
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.experiments = self._load_existing_experiments()
    
    def _load_existing_experiments(self) -> list:
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return []
    
    def log_experiment(self, experiment: ModelExperiment):
        exp_dict = asdict(experiment)
        self.experiments.append(exp_dict)
        
        # Save to JSON
        with open(self.json_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        # Save to CSV
        df = pd.DataFrame(self.experiments)
        df.to_csv(self.csv_path, index=False)
        
        self.logger.info(f"Logged experiment: {experiment.experiment_id}")
        # Print the mapping info for easy reference
        print(f"\nExperiment ID Mapping:")
        print(f"ID: {experiment.experiment_id}")
        print(f"Model: {experiment.embedding_model}")
        print(f"Preprocessing: {experiment.preprocessing_method}")
        print(f"Dimension Reduction: {experiment.dimension_reduction}")
        print(f"Topics: {experiment.num_topics}")
        print(f"Components: {experiment.n_components}")
        print(f"Timestamp: {experiment.timestamp}")
    
    def get_experiments_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.experiments)
    
    def summarize_experiments(self):
        df = self.get_experiments_df()
        print("\nExperiment Summary:")
        print(f"Total experiments: {len(df)}")
        
        print("\nExperiments by preprocessing method:")
        preproc_summary = df.groupby('preprocessing_method').size()
        print(preproc_summary)
        
        print("\nExperiments by dimension reduction and preprocessing method:")
        summary = df.groupby(['dimension_reduction', 'preprocessing_method', 'embedding_model'])[['duration_seconds']].agg({
            'duration_seconds': ['count', 'mean', 'std']
        }).round(2)
        print(summary)
        
        # Print detailed stats for each preprocessing method
        print("\nDocument Statistics by Preprocessing Method:")
        for method in df['preprocessing_method'].unique():
            method_df = df[df['preprocessing_method'] == method]
            print(f"\n{method.upper()}:")
            print(f"Average document length: {method_df['avg_document_length'].mean():.1f} words")
            print(f"Average number of documents: {method_df['num_documents'].mean():.0f}")

def track_model_training(tracker: ExperimentTracker, model_name: str, embedding_model: str, 
                        topic_model, documents: list, preprocessing_method: str,
                        start_time: float):
    """Helper function to track a single model training run"""
    # Detect methods and parameters
    dim_reduction_info = detect_dimension_reduction(topic_model)
    clustering_info = detect_clustering_method(topic_model)
    doc_stats = calculate_document_stats(documents)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate experiment ID
    experiment_id = generate_experiment_id(
        embedding_model=embedding_model,
        dimension_reduction=dim_reduction_info['method'],
        clustering_method=clustering_info['method'],
        num_topics=topic_model.nr_topics,
        n_components=dim_reduction_info['params']['n_components'],
        preprocessing_method=preprocessing_method,
        timestamp=timestamp
    )
    
    experiment = ModelExperiment(
        experiment_id=experiment_id,
        model_name=model_name,
        embedding_model=embedding_model,
        num_topics=topic_model.nr_topics,
        dimension_reduction=dim_reduction_info['method'],
        clustering_method=clustering_info['method'],
        n_components=dim_reduction_info['params']['n_components'],
        preprocessing_method=preprocessing_method,
        num_documents=len(documents),
        avg_document_length=doc_stats["avg_document_length"],
        duration_seconds=time.time() - start_time,
        gpu_device=get_device_info(topic_model.embedding_model),
        timestamp=timestamp,
        clustering_params=clustering_info['params'],
        reduction_params=dim_reduction_info['params'],
        preprocessing_stats=doc_stats
    )
    
    tracker.log_experiment(experiment)
    return experiment

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    tracker = ExperimentTracker()
    
    # You can now use the tracker with different preprocessing methods:
    # tracker.track_model_training(..., preprocessing_method="full_speeches", ...)
    # tracker.track_model_training(..., preprocessing_method="sentences", ...)
    # tracker.track_model_training(..., preprocessing_method="no_stopwords", ...)