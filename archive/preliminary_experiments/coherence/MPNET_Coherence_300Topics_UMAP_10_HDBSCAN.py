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
model = 'model_all_mpnet_base_v2_umap_hdbscan_speeches_300topics_10dim_2024-11-02_11-04-20_802644'
topic_model = BERTopic.load(f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/{model}")

def calculate_coherence_scores(topic_model, docs):
    # Get topic words for all topics except -1
    topic_words = []
    topic_ids = []
    topic_info = []
    
    # Get all topics from the model
    topics = topic_model.get_topics()
    
    # Get topics and their sizes
    topic_sizes = topic_model.get_topic_freq()
    
    # Process all topics except -1 (noise)
    for topic_num in topics.keys():
        if topic_num != -1:  # Skip the noise topic
            # Get topic terms with their weights
            topic_terms = [(word, weight) for word, weight in topics[topic_num]]
            words = [word for word, _ in topic_terms][:10]  # Get top 10 words
            
            if words:  # Only process if we have words
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
    
    # Calculate overall c_v coherence score for all topics
    cv_coherence_model = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    cv_overall_score = cv_coherence_model.get_coherence()

    # Calculate overall c_npmi coherence score for all topics
    cnpmi_coherence_model = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    cnpmi_overall_score = cnpmi_coherence_model.get_coherence()
    
    # Calculate individual coherence scores
    cv_coherence = {}
    cnpmi_coherence = {}
    
    for idx, topic_id in enumerate(topic_ids):
        # Calculate C_V coherence
        cv_topic_coherence_model = CoherenceModel(
            topics=[topic_words[idx]],
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        cv_coherence[topic_id] = cv_topic_coherence_model.get_coherence()
        
        # Calculate C_NPMI coherence
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
        'size': topic_sizes['Count'].sum(),  # Total size across all topics
        'top_words': None,
        'word_weights': None
    }])
    
    # Sort by topic_id numerically (except 'Overall' which will be at the end)
    df_results = df_results.sort_values(by='topic_id', key=lambda x: pd.to_numeric(x, errors='coerce'))
    df_results = pd.concat([df_results, overall_result], ignore_index=True)
    
    return df_results

# Calculate scores and save to CSV
print("Calculating coherence scores...")
results_df = calculate_coherence_scores(topic_model, documents)

# Create output file
output_path = f'/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/analysis/coherence/coherence_scores_all_{model}.csv'

# Save results
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")