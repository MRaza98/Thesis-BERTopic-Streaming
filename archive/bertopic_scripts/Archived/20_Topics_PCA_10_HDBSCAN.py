# %%
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import time
import psutil
import GPUtil
import threading
import queue

# %%
keep_logging = True

# %%
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_sample_2015.csv')
documents = df['text'].tolist()

# %%
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(documents)

# %%
pca = PCA()
pca.fit(embeddings)

# %%
# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# %%
# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


# %%
# Find optimal number of components (elbow method)
diff = np.diff(cumulative_variance_ratio)
elbow = np.argmax(diff < 0.01) + 1
print(f"Optimal number of components (elbow method): {elbow}")

# %%
# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components for 95% variance: {n_components_95}")

# %%
# Update your models with the optimal number of components
pca_model = PCA(n_components=10)

# %%
def log_gpu_usage():
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        logger.info(f"GPU {i}: {gpu.name}")
        logger.info(f"  Memory Use: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        logger.info(f"  GPU Utilization: {gpu.load*100:.2f}%")

def continuous_gpu_logging(interval=5):
    while keep_logging:
        log_gpu_usage()
        time.sleep(interval)

# %%
def log_system_info():
    logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
    logger.info(f"RAM Usage: {psutil.virtual_memory().percent}%")

# %%
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load documents
logger.info("Loading documents...")
# Assuming 'documents' is defined somewhere in your code
logger.info(f"Loaded {len(documents)} documents")

# %%
# Model 1
logger.info("Initializing Model 1 (all-MiniLM-L6-v2)")
model1 = SentenceTransformer('all-MiniLM-L6-v2').to(device)
topic_model1 = BERTopic(embedding_model=model1, nr_topics=20, umap_model=pca_model)
logger.info("Model 1 initialization complete")
log_gpu_usage()

# %%
# Model 2
logger.info("Initializing Model 2 (all-mpnet-base-v2)")
model2 = SentenceTransformer('all-mpnet-base-v2').to(device)
topic_model2 = BERTopic(embedding_model=model2, nr_topics=20, umap_model=pca_model)
logger.info("Model 2 initialization complete")
log_gpu_usage()

# %%
# Start background GPU logging
gpu_logging_thread = threading.Thread(target=continuous_gpu_logging)
gpu_logging_thread.start()

# %%
# Fit and transform Model 1
logger.info("Starting fit_transform for Model 1 (all-MiniLM-L6-v2)")
start_time = time.time()
topics1, _ = topic_model1.fit_transform(documents)
end_time = time.time()
duration = end_time - start_time
logger.info(f"Model 1 (mini) processing time: {duration:.2f} seconds")

# %%
# Save Model 1
logger.info("Saving Model 1")
topic_model1.save("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_all_MiniLM_L6_v2_20_topics_pca_10_hdbscan")
logger.info("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_all_mpnet_base_v2_20_topics_pca_10_hdbscan")

# %%
# Fit and transform Model 2
logger.info("Starting fit_transform for Model 2 (all-mpnet-base-v2)")
start_time = time.time()
topics2, _ = topic_model2.fit_transform(documents)
end_time = time.time()
duration = end_time - start_time
logger.info(f"Model 2 (mpnet) processing time: {duration:.2f} seconds")

# %%
# Save Model 2
logger.info("Saving Model 2")
topic_model2.save("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_all_mpnet_base_v2_20_topics_pca_10_hdbscan")
logger.info("Model 2 saved successfully")

# %%
# Stop background GPU logging
keep_logging = False
gpu_logging_thread.join()

logger.info("All processing complete")

# %%
# Load model 1 results

topic_model_mini = BERTopic.load("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_all_MiniLM_L6_v2_20_topics_pca_10_hdbscan")

# %%
def compare_topic_distributions(model1):
    info1 = topic_model_mini.get_topic_info()

    print("mini:")
    print(f"Number of topics: {len(info1) - 1}")  # Subtract 1 to exclude the -1 topic (outliers)
    print(f"Average topic size: {info1[info1['Topic'] != -1]['Count'].mean():.2f}")
    print(f"Largest topic size: {info1['Count'].max()}")
    print(f"Smallest topic size: {info1[info1['Topic'] != -1]['Count'].min()}")

compare_topic_distributions(topic_model_mini)

# %%
def compare_top_topics(model1, model2, n_topics=10):
    info1 = model1.get_topic_info()
    info2 = model2.get_topic_info()

    print("Top 5 topics for each model:")
    for i in range(n_topics):
        print(f"\nTopic {i}:")
        print("mini:", info1.iloc[i]['Name'], "-", info1.iloc[i]['Representation'])
        print("mpnet:", info2.iloc[i]['Name'], "-", info2.iloc[i]['Representation'])

compare_top_topics(topic_model_mini, topic_model_mpnet)

# %%
# Get topics and counts for both models
topics_mini = topic_model_mini.get_topic_info()
topics_mpnet = topic_model_mpnet.get_topic_info()

# Create DataFrames for each model
df_mini = topics_mini[['Topic', 'Count', 'Name']].rename(columns={
    'Topic': 'Topic_MiniLM',
    'Count': 'Count_MiniLM',
    'Name': 'Name_MiniLM'
})

df_mpnet = topics_mpnet[['Topic', 'Count', 'Name']].rename(columns={
    'Topic': 'Topic_MPNet',
    'Count': 'Count_MPNet',
    'Name': 'Name_MPNet'
})

# Combine the DataFrames side by side
df_combined = pd.concat([df_mini, df_mpnet], axis=1)

# Reorder columns for better readability
column_order = ['Topic_MiniLM', 'Count_MiniLM', 'Name_MiniLM', 
                'Topic_MPNet', 'Count_MPNet', 'Name_MPNet']
df_combined = df_combined[column_order]

# Sort by Count_MiniLM in descending order (you can change this if needed)
df_combined = df_combined.sort_values('Count_MiniLM', ascending=False)

# Display the first few rows of the combined DataFrame
print(df_combined)

df_combined.to_csv('bertopic_comparison_side_by_side.csv', index=False)


# %%
def compare_topic_diversity(model1, model2):
    topics1 = model1.get_topics()
    topics2 = model2.get_topics()

    unique_words1 = set(word for topic in topics1.values() for word, _ in topic)
    unique_words2 = set(word for topic in topics2.values() for word, _ in topic)

    print("Topic diversity:")
    print(f"mini unique words: {len(unique_words1)}")
    print(f"mpnet unique words: {len(unique_words2)}")

compare_topic_diversity(topic_model_mini, topic_model_mpnet)

# %%
new_topics_mini = topic_model1.reduce_outliers(documents, topics_mini)
new_topics_mpnet = topic_model2.reduce_outliers(documents, topics_mpnet)

# %%
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess

documents = [simple_preprocess(doc) for doc in documents]
dictionary = Dictionary(documents)

# %%
def get_topic_words(topic_model, num_words=10):
    topic_words = []
    for topic in range(len(topic_model.get_topic_info())-1):  # -1 to exclude the -1 topic
        words = [word for word, _ in topic_model.get_topic(topic)][:num_words]
        topic_words.append(words)
    return topic_words

# %%
topics_mini = get_topic_words(topic_model_mini)
topics_mpnet = get_topic_words(topic_model_mpnet)

def try_different_coherence(topics, texts, dictionary, model_name):
    for coherence_type in ['c_v', 'u_mass', 'c_uci']:
        try:
            coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence_type)
            coherence_score = coherence_model.get_coherence()
            print(f"{model_name} {coherence_type} Coherence: {coherence_score}")
        except Exception as e:
            print(f"Error calculating {coherence_type} coherence for {model_name}: {e}")

try_different_coherence(topic_model_mini, documents, dictionary, "mini")
try_different_coherence(topic_model_mpnet, documents, dictionary, "mpnet")

# %%



