from bertopic import BERTopic
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time

def calculate_single_topic_coherence(topic_words, texts, dictionary):
    print(f"\nStarting coherence calculation for topic words: {', '.join(topic_words)}")
    start_time = time.time()
    
    try:
        # Calculate C_V coherence
        print(f"  Creating C_V coherence model...")
        cv_start = time.time()
        cv_coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        print(f"  C_V model creation took {time.time() - cv_start:.2f} seconds")
        
        print(f"  Calculating C_V coherence score...")
        cv_calc_start = time.time()
        cv_score = cv_coherence_model.get_coherence()
        print(f"  C_V score calculation took {time.time() - cv_calc_start:.2f} seconds")
        print(f"  C_V score: {cv_score}")
        
        # Calculate C_NPMI coherence
        print(f"  Creating C_NPMI coherence model...")
        cnpmi_start = time.time()
        cnpmi_coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=texts,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        print(f"  C_NPMI model creation took {time.time() - cnpmi_start:.2f} seconds")
        
        print(f"  Calculating C_NPMI coherence score...")
        cnpmi_calc_start = time.time()
        cnpmi_score = cnpmi_coherence_model.get_coherence()
        print(f"  C_NPMI score calculation took {time.time() - cnpmi_calc_start:.2f} seconds")
        print(f"  C_NPMI score: {cnpmi_score}")
        
        total_time = time.time() - start_time
        print(f"Total coherence calculation time for this topic: {total_time:.2f} seconds")
        
        return cv_score, cnpmi_score
    except Exception as e:
        print(f"Error calculating coherence for words {topic_words}: {str(e)}")
        return None, None

def calculate_coherence_scores(topic_model, docs):
    overall_start_time = time.time()
    print("\n1. Starting document preprocessing...")
    preprocess_start = time.time()
    
    print("  1.1 Splitting documents into words...")
    texts = [doc.split() for doc in docs]
    print(f"  Number of documents: {len(texts)}")
    print(f"  Average document length: {np.mean([len(text) for text in texts]):.1f} words")
    
    print("  1.2 Creating dictionary...")
    dict_start = time.time()
    dictionary = corpora.Dictionary(texts)
    print(f"  Dictionary creation took {time.time() - dict_start:.2f} seconds")
    print(f"  Dictionary size: {len(dictionary)} unique tokens")
    
    print(f"Total preprocessing took {time.time() - preprocess_start:.2f} seconds")
    
    print("\n2. Getting topic information...")
    # Get topics and their sizes
    print("  2.1 Extracting topics from model...")
    topics = topic_model.get_topics()
    topic_sizes = topic_model.get_topic_freq()
    
    # Get outlier information
    outlier_size = topic_sizes[topic_sizes['Topic'] == -1]['Count'].iloc[0] if -1 in topic_sizes['Topic'].values else 0
    total_docs = topic_sizes['Count'].sum()
    outlier_percentage = (outlier_size / total_docs) * 100
    
    print(f"  Total number of topics (including outliers): {len(topics)}")
    print(f"  Number of outlier documents (Topic -1): {outlier_size}")
    print(f"  Percentage of outliers: {outlier_percentage:.2f}%")
    
    print("  2.2 Identifying top 20 topics by size...")
    top_20_topics = topic_sizes[topic_sizes['Topic'] != -1].nlargest(20, 'Count')['Topic'].tolist()
    print(f"  Top 20 topics: {top_20_topics}")
    
    # Process topics
    print("\n3. Processing topic data...")
    all_topic_words = []  # For coherence calculation (excluding -1)
    topic_data = []      # For final results (including -1)
    top_20_data = []
    top_20_words_list = []
    top_20_ids = []
    
    # First process outlier topic
    if -1 in topics:
        outlier_words = [word for word, _ in topics[-1]][:10]
        outlier_weights = [weight for _, weight in topics[-1]][:10]
        topic_data.append({
            'topic_id': -1,
            'words': outlier_words,
            'weights': outlier_weights,
            'size': outlier_size
        })
        print(f"  Processed outlier topic (-1)")
        print(f"    Words: {', '.join(outlier_words)}")
        print(f"    Weights: {', '.join([f'{w:.4f}' for w in outlier_weights])}")
    
    # Process regular topics
    for topic_num, topic_terms in topics.items():
        if topic_num != -1:
            words = [word for word, _ in topic_terms][:10]
            weights = [weight for _, weight in topic_terms][:10]
            
            if words:
                all_topic_words.append(words)
                topic_data.append({
                    'topic_id': topic_num,
                    'words': words,
                    'weights': weights,
                    'size': topic_sizes[topic_sizes['Topic'] == topic_num]['Count'].iloc[0]
                })
                
                if topic_num in top_20_topics:
                    print(f"  Processing top-20 topic {topic_num}")
                    print(f"    Words: {', '.join(words)}")
                    print(f"    Weights: {', '.join([f'{w:.4f}' for w in weights])}")
                    top_20_words_list.append(words)
                    top_20_ids.append(topic_num)
                    top_20_data.append({
                        'topic_id': topic_num,
                        'words': words,
                        'weights': weights
                    })

    # Calculate coherence scores for all topics
    print("\n4. Calculating coherence scores...")
    coherence_results = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Calculate coherence for individual topics
        future_to_topic = {
            executor.submit(calculate_single_topic_coherence, words, texts, dictionary): topic_id
            for topic_id, words in zip(top_20_ids, top_20_words_list)
        }
        
        for future in as_completed(future_to_topic):
            topic_id = future_to_topic[future]
            try:
                cv_score, cnpmi_score = future.result()
                coherence_results[topic_id] = (cv_score, cnpmi_score)
                print(f"Completed coherence calculation for topic {topic_id}")
            except Exception as e:
                print(f"Error processing topic {topic_id}: {str(e)}")
    
    # Calculate overall coherence
    print("\n5. Calculating overall coherence...")
    try:
        # Calculate overall C_V coherence
        cv_coherence_model = CoherenceModel(
            topics=all_topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        cv_overall_score = cv_coherence_model.get_coherence()
        print(f"Overall C_V coherence score: {cv_overall_score}")
        
        # Calculate overall C_NPMI coherence
        cnpmi_coherence_model = CoherenceModel(
            topics=all_topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        cnpmi_overall_score = cnpmi_coherence_model.get_coherence()
        print(f"Overall C_NPMI coherence score: {cnpmi_overall_score}")
    except Exception as e:
        print(f"Error calculating overall coherence: {str(e)}")
        cv_overall_score = None
        cnpmi_overall_score = None
    
    print("\n6. Creating results DataFrame...")
    results = []
    
    # Add all topics (including outliers) to results
    for info in topic_data:
        topic_id = info['topic_id']
        result = {
            'topic_id': topic_id,
            'cv_score': None if topic_id == -1 else coherence_results.get(topic_id, (None, None))[0],
            'cnpmi_score': None if topic_id == -1 else coherence_results.get(topic_id, (None, None))[1],
            'size': info['size'],
            'top_words': ', '.join(info['words']),
            'word_weights': ', '.join([f'{w:.4f}' for w in info['weights']])
        }
        results.append(result)
        print(f"  Added results for topic {topic_id}")
    
    df_results = pd.DataFrame(results)
    
    print("  Adding overall score...")
    overall_result = pd.DataFrame([{
        'topic_id': 'Overall',
        'cv_score': cv_overall_score,
        'cnpmi_score': cnpmi_overall_score,
        'size': topic_sizes['Count'].sum(),
        'top_words': None,
        'word_weights': None
    }])
    
    # Sort by topic_id numerically (except 'Overall' which will be at the end)
    # Make sure -1 appears first
    df_results = df_results.sort_values(by='topic_id', key=lambda x: pd.to_numeric(x, errors='coerce'))
    df_results = pd.concat([df_results, overall_result], ignore_index=True)
    
    total_time = time.time() - overall_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return df_results

# Load data
print("\nLoading data...")
load_start = time.time()
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2017.csv')
documents = df['text'].tolist()
print(f"Data loading took {time.time() - load_start:.2f} seconds")

# Load model
print("\nLoading BERTopic model...")
model_start = time.time()
model = 'model_2017_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-46-39_fac649'
topic_model = BERTopic.load(f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/{model}")
print(f"Model loading took {time.time() - model_start:.2f} seconds")

# Calculate scores and save to CSV
results_df = calculate_coherence_scores(topic_model, documents)

# Create output file
output_path = f'/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/analysis/incremental/{model}.csv'
print(f"\nSaving results to {output_path}")
results_df.to_csv(output_path, index=False)
print("Done!")