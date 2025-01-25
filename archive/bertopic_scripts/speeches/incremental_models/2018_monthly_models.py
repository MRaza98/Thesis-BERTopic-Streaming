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
speech_month = pd.to_datetime(df['date']).dt.strftime('%Y_%m')
unique_months = speech_month.unique()
documents = df['text'].tolist()

# Load documents
logger.info("Loading documents...")
logger.info(f"Loaded {len(documents)} documents")

# Initialize UMAP
umap_model = UMAP(
    n_components=10,
    n_neighbors=165,
    min_dist=0.1,
    random_state=42
)

logger.info("Initializing Model (all-mpnet-base-v2)")
model = SentenceTransformer("all-mpnet-base-v2")
topic_model1 = BERTopic(embedding_model=model,
                        umap_model=umap_model,
                        verbose=True)
logger.info("Model initialization complete")

for month in unique_months:
    logger.info(f"Processing month: {month}")
    
    # Filter documents for current month
    monthly_mask = speech_month == month
    monthly_documents = df[monthly_mask]['text'].tolist()

    logger.info(f"Processing {len(monthly_documents)} documents for {month}")

    # Fit and transform each model
    logger.info("Starting fit_transform for Model (all-mpnet-base-v2)")
    start_time = time.time()
    topics1, _ = topic_model1.fit_transform(monthly_documents)

    # Track the experiment
    experiment = track_model_training(
        tracker=tracker,
        model_name="Model",
        embedding_model="all-mpnet-base-v2",
        topic_model=topic_model1,
        documents=documents,
        start_time=start_time,
        preprocessing_method="speeches"
    )

    # Log the completion and duration
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Model processing time: {duration:.2f} seconds")

    # Save each model
    logger.info("Saving Model")
    save_path = f"/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/merged_models/2018_monthly/model_2018_{month}_{experiment.experiment_id}"
    topic_model1.save(save_path)
    logger.info("Model saved successfully")

    # Print experiment summary
    tracker.summarize_experiments()