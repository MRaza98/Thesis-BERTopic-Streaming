import numpy as np
import pandas as pd
from bertopic import BERTopic
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_topic_evolution(model_paths, merged_model_path):
    """Analyze how topics evolve as models are merged."""
    # Load models
    individual_models = [BERTopic.load(path) for path in model_paths]
    merged_model = BERTopic.load(merged_model_path)
    
    # Analysis dictionaries
    topic_counts = {}
    topic_similarities = defaultdict(list)
    
    # Analyze individual models
    years = list(range(2016, 2019))
    for i, model in enumerate(individual_models):
        topics = model.get_topic_info()
        topic_counts[years[i]] = len(topics[topics['Topic'] != -1])
        
        # Calculate topic similarities
        topic_reps = model.topic_embeddings_
        if topic_reps is not None:
            similarities = np.inner(topic_reps, topic_reps)
            avg_sim = (similarities.sum() - similarities.trace()) / (similarities.shape[0] * (similarities.shape[0] - 1))
            topic_similarities[years[i]].append(avg_sim)
    
    # Analyze merged model
    merged_topics = merged_model.get_topic_info()
    topic_counts[2019] = len(merged_topics[merged_topics['Topic'] != -1])  # Using 2019 as merged model year
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot topic counts
    plt.subplot(1, 2, 1)
    x_pos = np.arange(len(topic_counts))
    plt.bar(x_pos, list(topic_counts.values()))
    plt.xticks(x_pos, [str(year) if year != 2019 else 'Merged' for year in topic_counts.keys()], rotation=45)
    plt.title('Number of Topics by Model')
    plt.ylabel('Number of Topics')
    
    # Plot topic similarities
    plt.subplot(1, 2, 2)
    similarity_data = [topic_similarities[year] for year in years]
    plt.boxplot(similarity_data, labels=[str(year) for year in years])
    plt.title('Topic Similarities Distribution')
    plt.ylabel('Similarity Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('topic_evolution_analysis.png')
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': [f'Year {year}' for year in years] + ['Merged'],
        'Num_Topics': [topic_counts[year] for year in years] + [topic_counts[2019]],
        'Avg_Similarity': [np.mean(topic_similarities[year]) if year in topic_similarities else None 
                          for year in years] + [None]
    })
    
    # Analyze top topics
    top_topics_df = merged_topics.nlargest(10, 'Count')[['Topic', 'Name', 'Count']]
    
    return comparison_df, top_topics_df

def print_analysis(comparison_df, top_topics_df):
    """Print the analysis results"""
    print("\n=== Topic Evolution Analysis ===")
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    print("\nTop 10 Topics in Merged Model:")
    print(top_topics_df.to_string(index=False))

# Example usage
if __name__ == "__main__":
    model_paths = [
        "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2016_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-40-19_efdd56",
        "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2017_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-46-39_fac649",
        "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2018_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-55-50_2faf61"
    ]
    merged_model_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/merged_model_2016_to_2018_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-15"
    
    comparison_df, top_topics_df = analyze_topic_evolution(model_paths, merged_model_path)
    print_analysis(comparison_df, top_topics_df)