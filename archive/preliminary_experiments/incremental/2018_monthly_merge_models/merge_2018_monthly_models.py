from bertopic import BERTopic
import os

# Base path for models
base_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/merged_models/2018_monthly"

# List of model paths from your screenshot
model_paths = [
    "model_2018_2018_01_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-27-34_7520d5",
    "model_2018_2018_02_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-30-50_a7635f",
    "model_2018_2018_03_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-35-05_8ebe48",
    "model_2018_2018_04_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-37-41_4ac8e9",
    "model_2018_2018_05_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-40-46_e41ffd",
    "model_2018_2018_06_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-43-19_54e8e9",
    "model_2018_2018_07_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-46-04_47ca0a",
    "model_2018_2018_09_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-47-32_667500",
    "model_2018_2018_10_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-50-01_95c5c7",
    "model_2018_2018_11_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-53-05_64b7e8",
    "model_2018_2018_12_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-28_15-56-15_9c78d1"
]

def merge_monthly_models(model_paths, base_path, min_similarity=0.7):
    """
    Load and merge monthly BERTopic models
    
    Args:
        model_paths: List of model directory names
        base_path: Base directory where models are stored
        min_similarity: Minimum similarity threshold for merging topics
    
    Returns:
        merged_model: Combined BERTopic model
    """
    # Load all models
    print("Loading models...")
    models = []
    for path in model_paths:
        full_path = os.path.join(base_path, path)
        print(f"Loading model from: {full_path}")
        try:
            model = BERTopic.load(full_path)
            models.append(model)
            print(f"Successfully loaded model: {path}")
        except Exception as e:
            print(f"Error loading model {path}: {str(e)}")
    
    if not models:
        raise ValueError("No models were successfully loaded")
    
    # Merge models
    print(f"\nMerging {len(models)} models with minimum similarity {min_similarity}...")
    merged_model = BERTopic.merge_models(models, min_similarity=min_similarity)
    
    # Create output filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(
        base_path, 
        f"merged_model_2018_monthly_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_{timestamp}"
    )
    
    # Save merged model
    print(f"\nSaving merged model to: {output_path}")
    merged_model.save(output_path)
    
    return merged_model, output_path

if __name__ == "__main__":
    print("Starting model merging process...")
    
    try:
        merged_model, output_path = merge_monthly_models(model_paths, base_path)
        print("\nModel merging completed successfully!")
        print(f"Merged model saved to: {output_path}")
        
        # Print some basic information about the merged model
        topic_info = merged_model.get_topic_info()
        print(f"\nMerged model statistics:")
        print(f"Total number of topics: {len(topic_info)}")
        print(f"Number of non-outlier topics: {len(topic_info[topic_info['Topic'] != -1])}")
        
    except Exception as e:
        print(f"\nError during model merging: {str(e)}")