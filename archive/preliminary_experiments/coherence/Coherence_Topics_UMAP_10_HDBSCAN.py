from bertopic import BERTopic
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/sentences_england_speeches_sample_2015_20k.csv')
documents = df['text'].tolist()

# Load model
model = 'model_all_MiniLM_L6_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-10-25_00-02-03_e41641'
topic_model = BERTopic.load(f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/{model}")

def calculate_coherence_scores(topic_model, docs):
    # Get topic words for all topics except -1
    topic_words = []
    topic_ids = []
    topic_info = []
    
    # Get topics and their sizes
    topic_sizes = topic_model.get_topic_freq()
    # Sort topics by size (frequency) and get top 10 (excluding -1)
    top_topics = topic_sizes[topic_sizes['Topic'] != -1]['Topic'].iloc[:10].tolist()
    
    # Process top 10 topics
    for topic_num in top_topics:
        topic_terms = topic_model.get_topic(topic_num)
        words = [word for word, _ in topic_terms][:10]
        if words:
            topic_words.append(words)
            topic_ids.append(topic_num)
            # Store topic words and their weights
            topic_info.append({
                'topic_id': topic_num,
                'words': words,
                'weights': [weight for _, weight in topic_terms[:10]]
            })
    
    # Create dictionary for coherence calculation
    texts = [doc.split() for doc in docs]
    dictionary = corpora.Dictionary(texts)
    
    # Calculate overall c_v coherence score for top 10 topics
    cv_coherence_model = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    cv_overall_score = cv_coherence_model.get_coherence()

    # Calculate overall c_npmi coherence score for top 10 topics
    cnpmi_coherence_model = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    cnpmi_overall_score = cnpmi_coherence_model.get_coherence()
    
    # Calculate individual coherence scores
    cv_coherence = {}
    for idx, topic_id in enumerate(topic_ids):
        cv_topic_coherence_model = CoherenceModel(
            topics=[topic_words[idx]],
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        cv_coherence[topic_id] = cv_topic_coherence_model.get_coherence()

    # Calculate individual coherence scores
    cnpmi_coherence = {}
    for idx, topic_id in enumerate(topic_ids):
        cnpmi_topic_coherence_model = CoherenceModel(
            topics=[topic_words[idx]],
            texts=texts,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        cnpmi_coherence[topic_id] = cnpmi_topic_coherence_model.get_coherence()
    
    # Create results DataFrame
    results = []
    for info in topic_info:
        topic_id = info['topic_id']
        result = {
            'topic_id': topic_id,
            'cv_score': cv_coherence[topic_id],
            'cnpmi_score': cnpmi_coherence[topic_id],
            'size': topic_sizes[topic_sizes['Topic'] == topic_id]['Count'].iloc[0],
            'top_words': ', '.join(info['words']),
            'word_weights': ', '.join([f'{w:.4f}' for w in info['weights']])
        }
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Add overall score
    overall_result = pd.DataFrame([{
        'topic_id': 'Overall',
        'cv_score': cv_overall_score,
        'cnpmi_score': cnpmi_overall_score,
        'size': None,
        'top_words': None,
        'word_weights': None
    }])
    
    df_results = pd.concat([df_results, overall_result], ignore_index=True)
    
    return df_results

# Calculate scores and save to CSV
print("Calculating coherence scores...")
results_df = calculate_coherence_scores(topic_model, documents)

# Create output file
output_path = f'/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/analysis/coherence/coherence_scores_{model}.csv'

# Save results
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")