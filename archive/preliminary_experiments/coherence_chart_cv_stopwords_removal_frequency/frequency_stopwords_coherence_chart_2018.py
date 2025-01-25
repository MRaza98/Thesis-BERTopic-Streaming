from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from umap import UMAP
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List
import time
import os
from datetime import datetime
from collections import Counter
from tqdm import tqdm

def get_frequent_words(documents: List[str], n_words: int) -> set:
    """
    Get the n most frequent words across all documents.
    Count frequency by document occurrence, not total word count.
    """
    # Count in how many documents each word appears
    doc_frequency = Counter()
    for doc in documents:
        # Count each word only once per document
        unique_words = set(doc.split())
        doc_frequency.update(unique_words)
    
    # Get the n most frequent words
    frequent_words = set(word for word, count in doc_frequency.most_common(n_words))
    return frequent_words

def remove_frequent_words(documents: List[str], stopwords: set) -> List[str]:
    """Remove specified stopwords from documents"""
    processed_docs = []
    for doc in documents:
        words = doc.split()
        filtered_words = [word for word in words if word not in stopwords]
        processed_docs.append(' '.join(filtered_words))
    return processed_docs

def analyze_with_different_stopwords(documents: List[str], 
                                   output_dir: str,
                                   max_words: int = 500,
                                   step: int = 50) -> pd.DataFrame:
    """
    Analyze coherence scores and number of topics with different numbers of frequent words removed
    """
    results = []
    n_words_list = range(0, max_words + step, step)
    
    # Initialize models
    model = SentenceTransformer("all-mpnet-base-v2")
    umap_model = UMAP(n_components=10, n_neighbors=165, random_state=42)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/2018_freq_stopwords_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved in: {output_dir}")
    
    # For each number of frequent words to remove
    for n_words in tqdm(n_words_list):
        print(f"\nProcessing with {n_words} most frequent words removed...")
        start_time = time.time()
        
        try:
            # Get and remove frequent words
            frequent_words = get_frequent_words(documents, n_words)
            processed_docs = remove_frequent_words(documents, frequent_words)
            
            # Prepare for coherence calculation
            tokenized_docs = [doc.split() for doc in processed_docs]
            dictionary = Dictionary(tokenized_docs)
            
            # Train BERTopic
            topic_model = BERTopic(
                embedding_model=model,
                umap_model=umap_model,
                verbose=True
            )
            
            topics, _ = topic_model.fit_transform(processed_docs)
            
            # Calculate coherence score
            topic_info = topic_model.get_topic_info()
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
            result = {
                'n_words_removed': n_words,
                'coherence_score': c_v_score,
                'num_topics': len(topic_words),
                'training_time': time.time() - start_time,
                'frequent_words': ', '.join(sorted(list(frequent_words)))
            }
            results.append(result)
            
            # Save individual results
            pd.DataFrame([result]).to_csv(
                f'{output_dir}/model_freq_{n_words}_words_metrics.csv',
                index=False
            )
            
            # Save topic details
            topic_details = topic_model.get_topic_info()
            topic_details.to_csv(
                f'{output_dir}/model_freq_{n_words}_words_topics.csv',
                index=False
            )
            
            # Create dual-axis plot
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot coherence scores on primary y-axis
            results_df = pd.DataFrame(results)
            color1 = 'tab:blue'
            ax1.set_xlabel('Number of Most Frequent Words Removed')
            ax1.set_ylabel('Coherence Score (c_v)', color=color1)
            line1 = ax1.plot(results_df['n_words_removed'], 
                           results_df['coherence_score'], 
                           color=color1, marker='o',
                           label='Coherence Score')
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # Add coherence score annotations
            for x, y in zip(results_df['n_words_removed'], 
                          results_df['coherence_score']):
                ax1.annotate(f'{y:.3f}', (x, y), 
                           textcoords="offset points", 
                           xytext=(0,10), ha='center',
                           color=color1)
            
            # Create secondary y-axis for number of topics
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('Number of Topics', color=color2)
            line2 = ax2.plot(results_df['n_words_removed'], 
                           results_df['num_topics'], 
                           color=color2, marker='s',
                           label='Number of Topics')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Add number of topics annotations
            for x, y in zip(results_df['n_words_removed'], 
                          results_df['num_topics']):
                ax2.annotate(f'{y}', (x, y), 
                           textcoords="offset points", 
                           xytext=(0,-15), ha='center',
                           color=color2)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            plt.title('Topic Coherence and Number of Topics vs Frequent Words Removed')
            plt.grid(True)
            plt.xticks(results_df['n_words_removed'], rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/coherence_scores_progress.png')
            plt.close()
            
            print(f"Completed {n_words} words. Coherence: {c_v_score:.4f}, Topics: {len(topic_words)}")
            
        except Exception as e:
            print(f"Error processing {n_words} words: {str(e)}")
            continue
    
    # Save final combined results
    final_df = pd.DataFrame(results)
    final_df.to_csv(f'{output_dir}/final_combined_results.csv', index=False)
    
    return final_df, output_dir

if __name__ == "__main__":
    print("Starting frequency-based stopwords analysis...")
    
    # Load data
    df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2018.csv')
    documents = df['text'].tolist()
    
    # Run analysis
    results_df, output_dir = analyze_with_different_stopwords(
        documents,
        "bertopic_results",
        max_words=500,
        step=50
    )
    
    print("\nAnalysis complete! Results saved in:", output_dir)
    print("\nFiles saved include:")
    print("- Individual model metrics and topic details")
    print("- Lists of removed words for each iteration")
    print("- Progress plot")
    print("- Final combined results")