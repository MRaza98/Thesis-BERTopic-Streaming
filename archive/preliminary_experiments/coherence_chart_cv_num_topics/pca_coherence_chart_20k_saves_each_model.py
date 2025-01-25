from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List
import time
import os
from datetime import datetime

# Getting ride of the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_topic_coherence(topic_words: List[str], tokenized_docs: List[List[str]], dictionary: Dictionary) -> float:
    """
    Calculate coherence score for a single topic
    """
    try:
        coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    except:
        return 0.0

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
        'training_time': [model_results['training_time']]
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
            if topic != -1:
                topic_coherence = calculate_topic_coherence(
                    list(words),
                    tokenized_docs,
                    dictionary
                )
            
            topics_data.append({
                'topic_id': topic,
                'cv_score': topic_coherence,
                'size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].values[0],
                'top_words': ", ".join(words),
                'word_weights': ", ".join([f"{w:.3f}" for w in weights])
            })
        
        # Create and save topics DataFrame
        topics_df = pd.DataFrame(topics_data)
        topics_df.to_csv(f'{output_dir}/model_{num_topics}_topics_details.csv', index=False)

def update_progress_plot(results: list, output_dir: str):
    """
    Update the coherence score plot with current progress
    """
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 8))
    plt.plot(df['num_topics'], df['coherence_score'], 'b-', marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score (c_v)')
    plt.title('Topic Coherence Scores vs Number of Topics (In Progress)')
    plt.grid(True)
    plt.xticks(df['num_topics'], rotation=45)
    
    for x, y in zip(df['num_topics'], df['coherence_score']):
        plt.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_scores_progress.png')
    plt.close()

def train_and_evaluate_models(documents: List[str], 
                            output_dir: str,
                            start: int = 10, 
                            end: int = 300, 
                            step: int = 10) -> pd.DataFrame:
    """
    Train BERTopic models and evaluate their coherence scores with incremental saving
    """
    results = []
    topic_numbers = range(start, end + step, step)
    
    # Prepare documents for coherence calculation
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)

    model = SentenceTransformer("all-mpnet-base-v2")

    # Initialize UMAP
    umap_model = PCA(
        n_components=10,
        random_state=42
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/pca_bertopic_run_{timestamp}_20k"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved in: {output_dir}")
    
    for num_topics in topic_numbers:
        print(f"\nTraining model with {num_topics} topics...")
        start_time = time.time()
        
        try:
            # Train model
            topic_model = BERTopic(nr_topics=num_topics,
                                 embedding_model=model,
                                 umap_model=umap_model,
                                 verbose=True)
            topics, _ = topic_model.fit_transform(documents)
            
            # Get topic info directly from BERTopic
            topic_info = topic_model.get_topic_info()
            
            # Calculate overall coherence using all topics
            topic_words = []
            for topic in topic_info['Topic']:
                if topic != -1:  # Skip outlier topic
                    words, _ = zip(*topic_model.get_topic(topic))
                    topic_words.append(list(words))
            
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
            
            # Store results
            model_results = {
                'num_topics': num_topics,
                'coherence_score': c_v_score,
                'training_time': time.time() - start_time
            }
            
            # Save individual model results and the model itself
            save_individual_model(model_results, output_dir, topic_model, tokenized_docs, dictionary)
            
            # Add to overall results
            results.append(model_results)
            
            # Update progress plot
            update_progress_plot(results, output_dir)
            
            print(f"Completed {num_topics} topics. Coherence score: {c_v_score:.4f}")
            print(f"Results saved in {output_dir}")
            
        except Exception as e:
            print(f"Error processing model with {num_topics} topics: {str(e)}")
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