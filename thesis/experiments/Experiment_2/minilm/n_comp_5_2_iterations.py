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

def save_individual_model(model_results: dict, 
                         output_dir: str, 
                         topic_model: BERTopic,
                         tokenized_docs: List[List[str]],
                         dictionary: Dictionary) -> None:
    """
    Save individual model results and topic information.
    """
    run_dir = f"{output_dir}/n_components_{model_results['n_components']}_run_{model_results['run']}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model metrics
    pd.DataFrame([model_results]).to_csv(f'{run_dir}/metrics.csv', index=False)
    
    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(f'{run_dir}/topic_info.csv', index=False)
    
    # Save detailed topic words and their weights
    topic_details = {}
    for topic in topic_info['Topic']:
        if topic != -1:  # Skip outlier topic
            words, weights = zip(*topic_model.get_topic(topic))
            topic_details[f'Topic_{topic}'] = {
                'words': words,
                'weights': weights
            }
    
    with open(f'{run_dir}/topic_details.txt', 'w') as f:
        for topic, details in topic_details.items():
            f.write(f"\n{topic}:\n")
            for word, weight in zip(details['words'], details['weights']):
                f.write(f"{word}: {weight:.4f}\n")

def train_and_evaluate_models(train_documents: List[str], 
                            coherence_documents: List[str],
                            output_dir: str,
                            n_runs: int = 3) -> pd.DataFrame:
    results = []
    n_components_values = [11, 20]  # Only use 11 and 20
    
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
    
    for n_component in n_components_values:
        for run in range(n_runs):
            print(f"\nTraining model with n_component={n_component}, run={run+1}...")
            start_time = time.time()
            
            try:
                # Initialize UMAP with current n_components value
                umap_model = UMAP(
                    n_components=n_component,
                    random_state=42 + run  # Different seed for each run
                )
                
                # Train model
                topic_model = BERTopic(embedding_model=model,
                                     umap_model=umap_model,
                                     verbose=True)
                topics, _ = topic_model.fit_transform(train_documents)
                
                training_time = time.time() - start_time
                
                # Get topic info
                topic_info = topic_model.get_topic_info()
                num_topics = len([topic for topic in topic_info['Topic'] if topic != -1])
                
                # Calculate coherence
                topic_words = []
                for topic in topic_info['Topic']:
                    if topic != -1:
                        words, _ = zip(*topic_model.get_topic(topic))
                        words = [word.lower() for word in words]
                        valid_words = [word for word in words if word in coherence_dictionary.token2id]
                        if len(valid_words) >= 3:
                            topic_words.append(valid_words)
                
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
                    'n_components': n_component,
                    'run': run + 1,
                    'num_topics': num_topics,
                    'coherence_score': c_v_score,
                    'training_time': training_time,
                    'coherence_calculation_time': coherence_calculation_time
                }
                
                # Save individual model results
                save_individual_model(model_results, output_dir, topic_model, 
                                   tokenized_coherence_docs, coherence_dictionary)
                
                results.append(model_results)
                
                print(f"Completed n_components={n_component}, run={run+1}:")
                print(f"Number of topics: {num_topics}")
                print(f"Coherence score: {c_v_score:.4f}")
                print(f"Training time: {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error in run {run+1} with n_components={n_component}: {str(e)}")
                continue
    
    # Save final combined results
    final_df = pd.DataFrame(results)
    final_df.to_csv(f'{output_dir}/final_combined_results.csv', index=False)
    
    return final_df, output_dir

if __name__ == "__main__":
    print("Starting BERTopic analysis...")

    # Load and clean training data (2018)
    train_df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2018.csv')
    train_df['text'] = train_df['text'].fillna('')
    train_df['text'] = train_df['text'].astype(str)
    train_documents = [text.strip() for text in train_df['text'] if text.strip()]

    # Load and clean coherence evaluation data (2015-2019)
    coherence_df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2015_2019.csv')
    coherence_df['text'] = coherence_df['text'].fillna('')
    coherence_df['text'] = coherence_df['text'].astype(str)
    coherence_documents = [text.strip() for text in coherence_df['text'] if text.strip()]
    
    # Run analysis
    output_dir = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_2"
    results_df, final_output_dir = train_and_evaluate_models(
        train_documents,
        coherence_documents,
        output_dir,
        n_runs=3
    )
    
    print("\nAnalysis complete! Results saved in:", final_output_dir)