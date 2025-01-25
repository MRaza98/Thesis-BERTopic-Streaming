import pandas as pd
import numpy as np
from typing import List
import time
import os
from datetime import datetime
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Getting ride of the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_topic_coherence(topic_words: List[str], tokenized_docs: List[List[str]], dictionary: Dictionary) -> tuple:
    """
    Calculate coherence score for a single topic and return score and calculation time
    """
    start_time = time.time()
    try:
        coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        calculation_time = time.time() - start_time
        return coherence_score, calculation_time
    except:
        calculation_time = time.time() - start_time
        return 0.0, calculation_time

def save_individual_model(model_results: dict, output_dir: str, model: BERTopic = None, tokenized_docs: List[List[str]] = None, dictionary: Dictionary = None):
    """
    Save results for a single model iteration using BERTopic's native functions
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    num_topics = model_results['num_topics']
    
    # Save model metrics
    metrics_df = pd.DataFrame({
        'num_topics': [num_topics],
        'coherence_score': [model_results['coherence_score']],
        'training_time': [model_results['training_time']],
        'coherence_calculation_time': [model_results['coherence_calculation_time']]  # Added this line
    })
    metrics_df.to_csv(f'{output_dir}/model_{num_topics}_topics_metrics.csv', index=False)
    
    if model is not None and tokenized_docs is not None and dictionary is not None:
        # Get topic information using BERTopic's native function
        topic_info = model.get_topic_info()
        
        # Get representation for each topic (including outliers)
        topics_data = []
        for topic in topic_info['Topic']:
            # Always include topic info even for outlier topic
            words, weights = zip(*model.get_topic(topic))
            
            # Calculate coherence score only for non-outlier topics
            topic_coherence = 0.0
            coherence_time = 0.0
            if topic != -1:
                topic_coherence, coherence_time = calculate_topic_coherence(
                    list(words),
                    tokenized_docs,
                    dictionary
                )
            
            topics_data.append({
                'topic_id': topic,
                'cv_score': topic_coherence,
                'coherence_calculation_time': coherence_time,
                'size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].values[0],
                'top_words': ", ".join(words),
                'word_weights': ", ".join([f"{w:.3f}" for w in weights])
            })
        
        # Create and save topics DataFrame
        topics_df = pd.DataFrame(topics_data)
        topics_df.to_csv(f'{output_dir}/model_{num_topics}_topics_details.csv', index=False)

def update_progress_plot(results: list, output_dir: str):
    """
    Update the coherence score plot with current progress using dual y-axes
    """
    df = pd.DataFrame(results)
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot coherence scores on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Componenets')
    ax1.set_ylabel('Coherence Score (c_v)', color=color1)
    line1 = ax1.plot(df['n_components'], df['coherence_score'], 'o-', color=color1, label='Coherence Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot number of topics on secondary y-axis
    color2 = 'tab:red'
    ax2.set_ylabel('Number of Topics', color=color2)
    line2 = ax2.plot(df['n_components'], df['num_topics'], 'o-', color=color2, label='Number of Topics')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add annotations for coherence scores
    for x, y in zip(df['n_components'], df['coherence_score']):
        ax1.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    color=color1)
    
    # Add annotations for number of topics
    for x, y in zip(df['n_components'], df['num_topics']):
        ax2.annotate(f'{y}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    color=color2)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Topic Coherence Scores and Number of Topics vs Number of Components')
    plt.grid(True, alpha=0.3)
    ax1.set_xticks(df['n_components'])
    ax1.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_scores_progress.png')
    plt.close()

def train_and_evaluate_models(documents: List[str], 
                            output_dir: str,
                            start: int = 10, 
                            end: int = 50, 
                            step: int = 5) -> pd.DataFrame:
    results = []
    # Generate sequence of n_components values
    n_components_range = list(range(start, end + 1, step))
    
    # Prepare documents for coherence calculation
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)

    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/pca_svd_full_bertopic_run_{timestamp}_20k"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved in: {output_dir}")
    print(f"Will train models with n_components values: {n_components_range}")
    
    for n_component in n_components_range:
        print(f"\nTraining model with n_component={n_component}...")
        start_time = time.time()
        
        try:
            # Initialize UMAP with current n_components value
            umap_model = PCA(
                n_components=n_component,
                svd_solver='full',
                random_state=42
            )
            
            # Train model
            topic_model = BERTopic(embedding_model=model,
                                 umap_model=umap_model,
                                 verbose=True)
            topics, _ = topic_model.fit_transform(documents)
            
            training_time = time.time() - start_time
            
            # Get topic info directly from BERTopic
            topic_info = topic_model.get_topic_info()
            
            # Calculate number of non-outlier topics
            num_topics = len([topic for topic in topic_info['Topic'] if topic != -1])
            
            # Calculate overall coherence using all topics
            topic_words = []
            for topic in topic_info['Topic']:
                if topic != -1:  # Skip outlier topic
                    words, _ = zip(*topic_model.get_topic(topic))
                    topic_words.append(list(words))
            
            # Measure coherence calculation time separately
            coherence_start_time = time.time()
            if topic_words:
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                c_v_score = coherence_model.get_coherence()
            else:
                c_v_score = np.nan
            coherence_calculation_time = time.time() - coherence_start_time
            
            # Store results
            model_results = {
                'n_components': n_component,
                'num_topics': num_topics,
                'coherence_score': c_v_score,
                'training_time': training_time,
                'coherence_calculation_time': coherence_calculation_time
            }
            
            # Save individual model results and the model itself
            save_individual_model(model_results, output_dir, topic_model, tokenized_docs, dictionary)
            
            # Add to overall results
            results.append(model_results)
            
            # Update progress plot
            update_progress_plot(results, output_dir)
            
            print(f"Completed n_components={n_component}:")
            print(f"Number of topics: {num_topics}")
            print(f"Coherence score: {c_v_score:.4f}")
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Coherence calculation time: {coherence_calculation_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing model with n_components={n_component}: {str(e)}")
            print("Saving progress and continuing with next iteration...")
            continue
    
    # Save final combined results
    final_df = pd.DataFrame(results)
    final_df.to_csv(f'{output_dir}/final_combined_results.csv', index=False)
    
    return final_df, output_dir

if __name__ == "__main__":
    
    print("Starting BERTopic analysis...")

    df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_sample_2015_20k.csv')
    documents = df['text'].tolist()
    
    # Train models and get results
    results_df, output_dir = train_and_evaluate_models(documents, "bertopic_results")
    
    print("\nAnalysis complete! Results saved in:", output_dir)
    print("Files saved include:")
    print("- Individual model metrics and topic details")
    print("- Progress plot")
    print("- Final combined results")