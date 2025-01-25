from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple
import time
import os
from datetime import datetime

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_and_preprocess_data() -> Dict[str, Tuple[List[str], List[List[str]], Dictionary]]:
    periods = {
        '2016': '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2016.csv',
        '2017': '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2017.csv',
        '2018': '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2018.csv',
        '2016-2018': '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2016_2018.csv'
    }
    
    processed_data = {}
    
    # Process all years
    for period, file_path in periods.items():
        print(f"\nLoading and preprocessing data for {period}...")
        start_time = time.time()
        
        df = pd.read_csv(file_path)
        documents = df['text'].tolist()
        # Ensure proper tokenization by splitting on whitespace and filtering empty tokens
        tokenized_docs = [[token for token in doc.split() if token] for doc in documents]
        dictionary = Dictionary(tokenized_docs)
        
        processed_data[period] = (documents, tokenized_docs, dictionary)
        print(f"Loaded {len(documents)} documents from {period}")
        print(f"Processing took {time.time() - start_time:.2f} seconds")
    
    return processed_data

def calculate_topic_coherence(topic_words: List[str], tokenized_docs: List[List[str]], dictionary: Dictionary) -> float:
    """
    Calculate coherence score for a single topic
    """
    try:
        # Ensure topic words are in the dictionary
        valid_topic_words = [word for word in topic_words if word in dictionary.token2id]
        if not valid_topic_words:
            return 0.0
            
        coherence_model = CoherenceModel(
            topics=[valid_topic_words],
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    except:
        return 0.0

def calculate_model_coherence(model: BERTopic, tokenized_docs: List[List[str]], dictionary: Dictionary) -> Tuple[float, List[Dict]]:
    """
    Calculate overall coherence score and per-topic coherence for a model on given data
    """
    print("\nCalculating coherence scores...")
    start_time = time.time()
    
    topic_info = model.get_topic_info()
    topics_data = []
    topic_words_list = []
    
    # Process each topic
    for topic in topic_info['Topic']:
        topic_data = model.get_topic(topic)
        if not topic_data:  # Skip empty topics
            continue
            
        words, weights = zip(*topic_data)
        
        # Calculate individual topic coherence
        topic_coherence = 0.0
        if topic != -1:  # Skip outlier topic for overall coherence
            # Filter words that exist in the dictionary
            valid_words = [word for word in words if word in dictionary.token2id]
            if valid_words:
                topic_coherence = calculate_topic_coherence(
                    valid_words,
                    tokenized_docs,
                    dictionary
                )
                topic_words_list.append(valid_words)
        
        topics_data.append({
            'topic_id': topic,
            'cv_score': topic_coherence,
            'size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].values[0],
            'top_words': ", ".join(words),
            'word_weights': ", ".join([f"{w:.3f}" for w in weights])
        })
    
    # Calculate overall coherence
    overall_coherence = 0.0
    if topic_words_list:
        try:
            coherence_model = CoherenceModel(
                topics=topic_words_list,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            overall_coherence = coherence_model.get_coherence()
        except ValueError as e:
            print(f"Warning: Could not calculate overall coherence: {e}")
            overall_coherence = sum(d['cv_score'] for d in topics_data) / len(topics_data)
    
    print(f"Coherence calculation took {time.time() - start_time:.2f} seconds")
    return overall_coherence, topics_data

def save_temporal_results(results: List[Dict], output_dir: str, model_name: str):
    """
    Save temporal evaluation results and create visualization
    """
    # Create output directory if it doesn't exist
    base_dir = f"{output_dir}/out_of_sample_evaluation"
    os.makedirs(base_dir, exist_ok=True)
    
    # Extract year from model name (assuming it's in the model name)
    model_year = "2016_to_2018"  # Or extract from model_name if needed
    
    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{base_dir}/{model_year}_cross_year_coherence.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Ensure consistent order of periods
    periods = ['2016', '2017', '2018', '2016-2018']
    df_results['period'] = pd.Categorical(df_results['period'], categories=periods, ordered=True)
    df_results = df_results.sort_values('period')
    
    plt.plot(df_results['period'], df_results['overall_coherence'], 'b-', marker='o')
    plt.xlabel('Time Period')
    plt.ylabel('Coherence Score (c_v)')
    plt.title('Topic Model Coherence Across Time Periods')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Add value labels
    for x, y in zip(df_results['period'], df_results['overall_coherence']):
        plt.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/{model_year}_cross_year_coherence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual period details
    for period_data in results:
        period = period_data['period']
        df_topics = pd.DataFrame(period_data.get('topics_data', []))
        if not df_topics.empty:
            df_topics.to_csv(f'{base_dir}/{model_year}_topics_details_{period}.csv', index=False)
    
    print(f"\nResults saved in: {base_dir}")
    return base_dir

def evaluate_model_temporally(model_path: str, output_dir: str) -> pd.DataFrame:
    """
    Evaluate a BERTopic model's coherence across different time periods
    """
    print("\nLoading BERTopic model...")
    start_time = time.time()
    topic_model = BERTopic.load(model_path)
    model_name = os.path.basename(model_path)
    print(f"Model loading took {time.time() - start_time:.2f} seconds")
    
    # Load all datasets
    processed_data = load_and_preprocess_data()
    results = []
    
    # Evaluate each period
    for period, (documents, tokenized_docs, dictionary) in processed_data.items():
        print(f"\nEvaluating period: {period}")
        
        # Calculate coherence
        overall_coherence, topics_data = calculate_model_coherence(
            topic_model,
            tokenized_docs,
            dictionary
        )
        
        # Store results
        period_results = {
            'period': period,
            'overall_coherence': overall_coherence,
            'num_documents': len(documents),
            'processing_time': time.time() - start_time,
            'topics_data': topics_data  # Store topics data for saving later
        }
        results.append(period_results)
        
        print(f"Period {period} - Coherence: {overall_coherence:.4f}")
    
    # Save and visualize results
    final_output_dir = save_temporal_results(results, output_dir, model_name)
    
    return pd.DataFrame(results), final_output_dir

if __name__ == "__main__":
    print("Starting temporal evaluation of BERTopic model...")
    
    # Model path
    model = 'model_2016_to_2018_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_02-13-43_5a411a'
    model_path = f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/{model}"
    
    # Evaluate model
    results_df, output_dir = evaluate_model_temporally(
        model_path,
        "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/analysis/incremental"
    )
    
    print("\nEvaluation complete! Results saved in:", output_dir)