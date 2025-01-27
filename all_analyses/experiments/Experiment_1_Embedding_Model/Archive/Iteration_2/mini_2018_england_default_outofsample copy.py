from sentence_transformers import SentenceTransformer
from experiment_tracker import BERTopicExperimentTracker
from bertopic import BERTopic
import pandas as pd
import torch
import os
import time
from umap import UMAP

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_coherence_docs(file_path: str):
    """Load documents for coherence evaluation."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text'])
    coherence_docs = [str(text) for text in df['text'].tolist() if str(text).strip()]
    print(f"Number of documents for coherence evaluation: {len(coherence_docs)}")
    return coherence_docs

def run_minilm_experiment(train_docs, coherence_docs):
    # Explicitly set CUDA device to GPU 3
    torch.cuda.set_device(3)
    
    # Initialize the tracker with specific output directory
    tracker = BERTopicExperimentTracker(output_dir="/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_1/Iteration_2")
    
    # Configure model parameters
    model_params = {
        'embedding_model': "all-MiniLM-L6-v2",
        'random_state': 42
    }
    
    umap_model = UMAP(random_state=42)
    
    # Initialize BERTopic with random_state and embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda:3')
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
        docs=coherence_docs,  # Use 2015-2019 docs for coherence calculation
        model_id="minilm_model"
    )
    
    # Read the latest experiments data
    latest_experiments = pd.read_csv(tracker.experiments_file)
    
    return topic_model, latest_experiments, tracker.get_topics()

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
    
    # Run experiment
    model, experiments_df, topics_df = run_minilm_experiment(train_docs, coherence_docs)
    
    # Print summary
    print("\nExperiment Results:")
    print(f"Number of training documents (2018): {len(train_docs)}")
    print(f"Number of coherence evaluation documents (2015-2019): {len(coherence_docs)}")
    print(f"Number of topics: {len(model.get_topics())}")
    
    # Get the last experiment results from the updated DataFrame
    if not experiments_df.empty:
        last_experiment = experiments_df.iloc[-1]
        print("\nTiming Results:")
        print(f"Total training time: {last_experiment['total_training_time']:.2f} seconds")
        print(f"Overall coherence score (calculated on 2015-2019 data): {last_experiment['c_v_score']:.4f}")
    else:
        print("\nNo experiment results found in the CSV file.")