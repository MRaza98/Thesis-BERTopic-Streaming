# Major parts of the code were written with AI.
# AI was especially used in debugging, improving logging, and giving feedback on my initial drafts.

from sentence_transformers import SentenceTransformer
from experiment_tracker import BERTopicExperimentTracker
import pandas as pd
import torch
import os
import time
from bertopic import BERTopic
from umap import UMAP
import numpy as np

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_coherence_docs(file_path: str):
    """Load documents for coherence evaluation."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text'])
    coherence_docs = [str(text) for text in df['text'].tolist() if str(text).strip()]
    print(f"Number of documents for coherence evaluation: {len(coherence_docs)}")
    return coherence_docs

def save_top_topics(topic_model, output_dir, run_number):
    """Save top 10 topics with their words and weights."""
    topic_info = topic_model.get_topic_info()
    # Filter out -1 (outlier topic) and get top 10 topics by size
    valid_topics = topic_info[topic_info['Topic'] != -1].nlargest(10, 'Count')
    
    # Create a list to store topic details
    topic_details = []
    
    for _, row in valid_topics.iterrows():
        topic_id = row['Topic']
        topic_words = topic_model.get_topic(topic_id)
        # Format words and weights
        words = [word for word, _ in topic_words[:10]]  # Get top 10 words
        weights = [weight for _, weight in topic_words[:10]]
        
        topic_details.append({
            'run_number': run_number,
            'topic_id': topic_id,
            'size': row['Count'],
            'words': ', '.join(words),
            'weights': ', '.join([f'{w:.4f}' for w in weights])
        })
    
    # Convert to DataFrame and save
    topic_df = pd.DataFrame(topic_details)
    
    # If file exists, append; otherwise create new
    topic_file = os.path.join(output_dir, 'topic_details.csv')
    if os.path.exists(topic_file):
        existing_topics = pd.read_csv(topic_file)
        topic_df = pd.concat([existing_topics, topic_df], ignore_index=True)
    
    topic_df.to_csv(topic_file, index=False)
    return topic_df

def run_single_experiment(train_docs, coherence_docs, run_number, tracker):
    """Run a single experiment iteration."""
    # Explicitly set CUDA device to GPU 3
    torch.cuda.set_device(3)
    
    # Configure model parameters
    model_params = {
        'embedding_model': "all-mpnet-base-v2",
        'random_state': 42 + run_number  # Different seed for each run
    }
    
    umap_model = UMAP(
                    n_components=5,
                    min_dist=0.0,
                    n_neighbors=15,
                    random_state=42 + run_number)
    
    # Initialize BERTopic with random_state and embedding model
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda:3')
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        verbose=True
    )
    
    # Track total training time
    start_time = time.time()
    topics, probs = topic_model.fit_transform(train_docs)
    total_time = time.time() - start_time
    
    # Add training time to parameters
    model_params['total_training_time'] = total_time
    
    # Log the experiment using coherence_docs for coherence calculation
    tracker.log_experiment(
        model_params=model_params,
        topic_model=topic_model,
        docs=coherence_docs,
        model_id=f"mpnetlm_model_run_{run_number}"
    )
    
    # Save top 10 topics
    save_top_topics(topic_model, tracker.output_dir, run_number)
    
    # Get the latest experiment results
    experiments_df = tracker.get_experiments()
    
    # Handle case when DataFrame might be empty
    try:
        if not experiments_df.empty:
            last_experiment = experiments_df.iloc[-1]
            c_v_score = last_experiment['c_v_score']
        else:
            c_v_score = np.nan
    except Exception as e:
        print(f"Warning: Could not retrieve last experiment: {e}")
        c_v_score = np.nan
    
    return {
        'training_time': total_time,
        'c_v_score': c_v_score,
        'num_topics': len(topic_model.get_topics())
    }

def run_multiple_experiments(train_docs, coherence_docs, n_runs=5):
    """Run multiple experiments and store results in a single CSV."""
    output_dir = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_1/Iteration_3_consistent_with_exp2/mpnet"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a single tracker for all runs
    tracker = BERTopicExperimentTracker(output_dir=output_dir)
    
    all_times = []
    all_scores = []
    all_topics = []
    
    print(f"\nRunning {n_runs} experiments...")
    
    for run in range(n_runs):
        print(f"\nStarting Run {run + 1}/{n_runs}")
        try:
            result = run_single_experiment(train_docs, coherence_docs, run + 1, tracker)
            
            all_times.append(result['training_time'])
            all_scores.append(result['c_v_score'])
            all_topics.append(result['num_topics'])
            
            # Print individual run results
            print(f"\nRun {run + 1} Results:")
            print(f"Training time: {result['training_time']:.2f} seconds")
            print(f"Coherence score: {result['c_v_score']:.4f}")
            print(f"Number of topics: {result['num_topics']}")
            
        except Exception as e:
            print(f"Error in run {run + 1}: {str(e)}")
            continue
    
    # Calculate averages and standard deviations (excluding NaN values)
    summary = {
        'avg_training_time': np.nanmean(all_times),
        'std_training_time': np.nanstd(all_times),
        'avg_coherence_score': np.nanmean(all_scores),
        'std_coherence_score': np.nanstd(all_scores),
        'avg_num_topics': np.nanmean(all_topics),
        'std_num_topics': np.nanstd(all_topics)
    }
    
    # Save summary to a separate CSV
    summary_df = pd.DataFrame({
        'metric': ['Training Time', 'Coherence Score', 'Number of Topics'],
        'average': [summary['avg_training_time'], 
                   summary['avg_coherence_score'], 
                   summary['avg_num_topics']],
        'std_dev': [summary['std_training_time'], 
                   summary['std_coherence_score'], 
                   summary['std_num_topics']]
    })
    summary_df.to_csv(os.path.join(output_dir, 'average_results.csv'), index=False)
    
    return summary

if __name__ == "__main__":
    # Load the training data (2018)
    train_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2018.csv"
    df = pd.read_csv(train_path)
    
    # Clean the training data
    print(f"Original number of training documents: {len(df)}")
    df = df.dropna(subset=['text'])
    
    # Convert all texts to strings and remove empty strings
    train_docs = [str(text) for text in df['text'].tolist() if str(text).strip()]
    print(f"Final number of training documents after cleaning: {len(train_docs)}")
    
    # Load the coherence evaluation data (2015-2019)
    coherence_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/preprocessed_england_speeches_2015_2019.csv"
    coherence_docs = load_coherence_docs(coherence_path)
    
    # Run experiments
    summary = run_multiple_experiments(train_docs, coherence_docs)
    
    # Print final summary
    print("\nFinal Results Summary:")
    print(f"Average training time: {summary['avg_training_time']:.2f} ± {summary['std_training_time']:.2f} seconds")
    print(f"Average coherence score: {summary['avg_coherence_score']:.4f} ± {summary['std_coherence_score']:.4f}")
    print(f"Average number of topics: {summary['avg_num_topics']:.1f} ± {summary['std_num_topics']:.1f}")
    print("\nDetailed results for each run can be found in 'bertopic_experiments.csv'")
    print("Summary statistics have been saved to 'average_results.csv'")
    print("Top 10 topics for each run have been saved to 'topic_details.csv'")
