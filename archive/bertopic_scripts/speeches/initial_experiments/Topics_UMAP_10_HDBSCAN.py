# main_topic_modeling.py
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.decomposition import KernelPCA
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
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_sample_2015_20k.csv')
documents = df['text'].tolist()

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load documents
logger.info("Loading documents...")
logger.info(f"Loaded {len(documents)} documents")

# Initialize kPCA
umap_model = UMAP(
    n_components=10,
    random_state=42
)

# Model 1
logger.info("Initializing Model 1 (all-MiniLM-L6-v2)")
model1 = SentenceTransformer('all-MiniLM-L6-v2').to(device)
topic_model1 = BERTopic(embedding_model=model1,
                        umap_model=umap_model,
                        verbose=True)
logger.info("Model 1 initialization complete")

# Fit and transform Model 1
logger.info("Starting fit_transform for Model 1 (all-MiniLM-L6-v2)")
start_time = time.time()
topics1, _ = topic_model1.fit_transform(documents)

# Track the experiment
experiment = track_model_training(
    tracker=tracker,
    model_name="Model1",
    embedding_model="all-MiniLM-L6-v2",
    topic_model=topic_model1,
    documents=documents,
    start_time=start_time,
    preprocessing_method="speeches"
)

# Log the completion and duration
end_time = time.time()
duration = end_time - start_time
logger.info(f"Model 1 (mini) processing time: {duration:.2f} seconds")

# Save Model 1
logger.info("Saving Model 1")
save_path = f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_{experiment.experiment_id}"
topic_model1.save(save_path)
logger.info("Model 1 saved successfully")

# Print experiment summary
tracker.summarize_experiments()