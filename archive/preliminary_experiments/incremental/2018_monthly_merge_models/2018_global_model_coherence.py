from bertopic import BERTopic
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import numpy as np

# Load data
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/nltk_stopwords_removed_england_speeches_sample_2018.csv')
documents = df['text'].tolist()

# Load model
model = 'model_2018_all_mpnet_base_v2_umap_hdbscan_nltk_stopwords_Nonetopics_10dim_2024-11-30_14-14-04_f66a63'
topic_model = BERTopic.load(f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/{model}")

def calculate_coherence_scores(topic_model, docs):
    # Get topic words for all topics except -1
    topic_words = []
    topic_ids = []
    topic_info = []
    
    # Get topics and their sizes
    topic_sizes = topic_model.get_topic_freq()
    # Get all topics except -1 (noise topic)
    all_topics = topic_sizes[topic_sizes['Topic'] != -1]['Topic'].tolist()
    
    # Create dictionary for coherence calculation first
    texts = [doc.split() for doc in docs]
    dictionary = corpora.Dictionary(texts)
    
    # Process all topics
    for topic_num in all_topics:
        topic_terms = topic_model.get_topic(topic_num)
        words = [word for word, _ in topic_terms][:10]
        
        # Only include topics with enough words that exist in the dictionary
        valid_words = [word for word in words if word in dictionary.token2id]
        if len(valid_words) >= 3:  # Ensure at least 3 valid words
            topic_words.append(valid_words)
            topic_ids.append(topic_num)
            topic_info.append({
                'topic_id': topic_num,
                'words': valid_words,
                'weights': [weight for word, weight in topic_terms[:10] if word in dictionary.token2id]
            })
    
    if not topic_words:
        return pd.DataFrame()  # Return empty DataFrame if no valid topics
    
    # Calculate overall c_v coherence score for all topics
    try:
        cv_coherence_model = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        cv_overall_score = cv_coherence_model.get_coherence()
    except:
        cv_overall_score = np.nan
    
    # Calculate individual coherence scores
    cv_coherence = {}
    for idx, topic_id in enumerate(topic_ids):
        try:
            cv_topic_coherence_model = CoherenceModel(
                topics=[topic_words[idx]],
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            cv_coherence[topic_id] = cv_topic_coherence_model.get_coherence()
        except:
            cv_coherence[topic_id] = np.nan
    
    # Create results DataFrame
    results = []
    for info in topic_info:
        topic_id = info['topic_id']
        if topic_id in cv_coherence:  # Only include topics with valid coherence scores
            result = {
                'topic_id': topic_id,
                'cv_score': cv_coherence[topic_id],
                'size': topic_sizes[topic_sizes['Topic'] == topic_id]['Count'].iloc[0],
                'top_words': ', '.join(info['words']),
                'word_weights': ', '.join([f'{w:.4f}' for w in info['weights']])
            }
            results.append(result)
    
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Add overall score
        overall_result = pd.DataFrame([{
            'topic_id': 'Overall',
            'cv_score': cv_overall_score,
            'size': topic_sizes['Count'].sum(),
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
output_path = f'/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/analysis/incremental/2018_monthly_merge_models/coherence_{model}.csv'

# Save results
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")