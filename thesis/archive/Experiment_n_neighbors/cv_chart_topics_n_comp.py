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
from umap import UMAP
from sentence_transformers import SentenceTransformer

# Getting rid of the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_topic_coherence(topic_words: List[str], tokenized_docs: List[List[str]], dictionary: Dictionary) -> tuple:
    """
    Calculate coherence score for a single topic and return score and calculation time
    """
    start_time = time.time()
    try:
        # Ensure topic words are in the dictionary
        valid_words = [word for word in topic_words if word in dictionary.token2id]
        if len(valid_words) < 3:  # Need at least 3 words for coherence calculation
            return 0.0, time.time() - start_time
            
        coherence_model = CoherenceModel(
            topics=[valid_words],
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
    
    n_neighbours = model_results['n_neighbours']
    
    # Save model metrics
    metrics_df = pd.DataFrame({
        'num_topics': [model_results['num_topics']],
        'coherence_score': [model_results['coherence_score']],
        'training_time': [model_results['training_time']],
        'coherence_calculation_time': [model_results['coherence_calculation_time']]
    })
    metrics_df.to_csv(f'{output_dir}/model_{n_neighbours}_topics_metrics.csv', index=False)
    
    if model is not None and tokenized_docs is not None and dictionary is not None:
        # Get topic information using BERTopic's native function
        topic_info = model.get_topic_info()
        
        # Get top 20 topics by size (excluding outlier topic -1)
        top_20_topics = topic_info[topic_info['Topic'] != -1].nlargest(20, 'Count')['Topic'].tolist()
        
        # Always include outlier topic at the start if it exists
        if -1 in topic_info['Topic'].values:
            top_20_topics = [-1] + top_20_topics
        
        # Get representation for selected topics
        topics_data = []
        for topic in top_20_topics:
            words, weights = zip(*model.get_topic(topic))
            
            # Calculate coherence score only for non-outlier topics
            topic_coherence = 0.0
            coherence_time = 0.0
            if topic != -1:
                valid_words = [word.lower() for word in words if word.lower() in dictionary.token2id]
                if len(valid_words) >= 3:
                    topic_coherence, coherence_time = calculate_topic_coherence(
                        valid_words,
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
        topics_df.to_csv(f'{output_dir}/model_{n_neighbours}_topics_details.csv', index=False)

def update_progress_plot(results: list, output_dir: str):
    """
    Update the coherence score plot with current progress using dual y-axes
    """
    df = pd.DataFrame(results)
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot coherence scores on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Neighbours')
    ax1.set_ylabel('Coherence Score (c_v)', color=color1)
    line1 = ax1.plot(df['n_neighbours'], df['coherence_score'], 'o-', color=color1, label='Coherence Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot number of topics on secondary y-axis
    color2 = 'tab:red'
    ax2.set_ylabel('Number of Topics', color=color2)
    line2 = ax2.plot(df['n_neighbours'], df['num_topics'], 'o-', color=color2, label='Number of Topics')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add annotations for coherence scores
    for x, y in zip(df['n_neighbours'], df['coherence_score']):
        ax1.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    color=color1)
    
    # Add annotations for number of topics
    for x, y in zip(df['n_neighbours'], df['num_topics']):
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
    
    plt.title('Topic Coherence Scores and Number of Topics vs Number of Neighbours')
    plt.grid(True, alpha=0.3)
    ax1.set_xticks(df['n_neighbours'])
    ax1.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_scores_progress.png')
    plt.close()

def train_and_evaluate_models(train_documents: List[str], 
                            coherence_documents: List[str],
                            output_dir: str,
                            start: int = 50, 
                            end: int = 100, 
                            step: int = 50) -> pd.DataFrame:
    results = []
    # Generate sequence of n_neighbours values
    n_neighbours_range = list(range(start, end + 1, step))
    
    # Prepare coherence documents for coherence calculation
    print("Preparing documents for coherence calculation...")
    tokenized_coherence_docs = [doc.lower().split() for doc in coherence_documents]
    coherence_dictionary = Dictionary(tokenized_coherence_docs)
    print(f"Dictionary size: {len(coherence_dictionary)}")

    model = SentenceTransformer("all-miniLM-L6-v2")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/bertopic_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved in: {output_dir}")
    print(f"Will train models with n_neighbours values: {n_neighbours_range}")
    print(f"Number of documents for training: {len(train_documents)}")
    print(f"Number of documents for coherence calculation: {len(coherence_documents)}")
    
    for n_neighbor in n_neighbours_range:
        print(f"\nTraining model with n_neighbor={n_neighbor}...")
        start_time = time.time()
        
        try:
            # Initialize UMAP with current n_neighbors value
            umap_model = UMAP(
                n_components=11,
                n_neighbors=n_neighbor,
                random_state=42
            )
            
            # Train model on 2018 data
            topic_model = BERTopic(embedding_model=model,
                                 umap_model=umap_model,
                                 verbose=True)
            topics, _ = topic_model.fit_transform(train_documents)
            
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
                    # Convert words to lowercase for matching with dictionary
                    words = [word.lower() for word in words]
                    valid_words = [word for word in words if word in coherence_dictionary.token2id]
                    if len(valid_words) >= 3:  # Only include topics with at least 3 valid words
                        topic_words.append(valid_words)
            
            # Measure coherence calculation time separately
            coherence_start_time = time.time()
            if topic_words:
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized_coherence_docs,
                    dictionary=coherence_dictionary,
                    coherence='c_v'
                )
                c_v_score = coherence_model.get_coherence()
            else:
                c_v_score = np.nan
            coherence_calculation_time = time.time() - coherence_start_time
            
            # Store results
            model_results = {
                'n_neighbours': n_neighbor,
                'num_topics': num_topics,
                'coherence_score': c_v_score,
                'training_time': training_time,
                'coherence_calculation_time': coherence_calculation_time
            }
            
            # Save individual model results and the model itself
            save_individual_model(model_results, output_dir, topic_model, tokenized_coherence_docs, coherence_dictionary)
            
            # Add to overall results
            results.append(model_results)
            
            # Update progress plot
            update_progress_plot(results, output_dir)
            
            print(f"Completed n_neighbours={n_neighbor}:")
            print(f"Number of topics: {num_topics}")
            print(f"Number of topics used for coherence: {len(topic_words)}")
            print(f"Coherence score (on 2015-2019 data): {c_v_score:.4f}")
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Coherence calculation time: {coherence_calculation_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing model with n_neighbours={n_neighbor}: {str(e)}")
            print("Saving progress and continuing with next iteration...")
            continue
    
    # Save final combined results
    final_df = pd.DataFrame(results)
    final_df.to_csv(f'{output_dir}/final_combined_results.csv', index=False)
    
    return final_df, output_dir

if __name__ == "__main__":
    print("Starting BERTopic analysis...")

    # Load and clean training data (2018)
    train_df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2018.csv')
    print(f"Original number of training documents: {len(train_df)}")
    train_df['text'] = train_df['text'].fillna('')
    train_df['text'] = train_df['text'].astype(str)
    train_documents = [text.strip() for text in train_df['text'] if text.strip()]
    print(f"Final number of training documents after cleaning: {len(train_documents)}")

    # Load and clean coherence evaluation data (2015-2019)
    coherence_df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2015_2019.csv')
    print(f"Original number of coherence evaluation documents: {len(coherence_df)}")
    coherence_df['text'] = coherence_df['text'].fillna('')
    coherence_df['text'] = coherence_df['text'].astype(str)
    coherence_documents = [text.strip() for text in coherence_df['text'] if text.strip()]
    print(f"Final number of coherence evaluation documents after cleaning: {len(coherence_documents)}")
    
    # Train models and get results
    output_dir = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_3"
    results_df, final_output_dir = train_and_evaluate_models(
        train_documents,
        coherence_documents,
        output_dir,
        start=50,    # Start from 50 neighbors
        end=1000,     # Go until 1000 neighbors
        step=50      # Steps of 50
    )
    
    print("\nAnalysis complete! Results saved in:", final_output_dir)
    print("Files saved include:")
    print("- Individual model metrics and topic details")
    print("- Progress plot")
    print("- Final combined results")