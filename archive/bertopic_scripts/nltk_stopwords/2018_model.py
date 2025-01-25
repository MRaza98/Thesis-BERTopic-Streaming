# main_topic_modeling.py
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
import logging
import torch
import time
from experiment_tracker import ExperimentTracker, track_model_training

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize experiment tracker
tracker = ExperimentTracker(output_dir="/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/experiment_tracker.py")

# Load data
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/nltk_stopwords_removed_england_speeches_sample_2018.csv')
documents = df['text'].tolist()

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load documents
logger.info("Loading documents...")
logger.info(f"Loaded {len(documents)} documents")

# Initialize UMAP
umap_model = UMAP(
    n_components=10,
    n_neighbors=765,
    min_dist=0.1,
    random_state=42
)

# Model 1
logger.info("Initializing Model (all-mpnet-base-v2)")
model = SentenceTransformer("all-mpnet-base-v2").to(device)
topic_model1 = BERTopic(embedding_model=model,
                        umap_model=umap_model,
                        verbose=True)
logger.info("Model initialization complete")

# Fit and transform Model 1
logger.info("Starting fit_transform for Model (all-mpnet-base-v2)")
start_time = time.time()
topics1, _ = topic_model1.fit_transform(documents)

# Track the experiment
experiment = track_model_training(
    tracker=tracker,
    model_name="Model",
    embedding_model="all-mpnet-base-v2",
    topic_model=topic_model1,
    documents=documents,
    start_time=start_time,
    preprocessing_method="nltk_stopwords"
)

# Log the completion and duration
end_time = time.time()
duration = end_time - start_time
logger.info(f"Model 1 processing time: {duration:.2f} seconds")

# Save Model 1
logger.info("Saving Model")
save_path = f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2018_{experiment.experiment_id}"
topic_model1.save(save_path)
logger.info("Model saved successfully")

# Print experiment summary
tracker.summarize_experiments()