import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_topic_details(model: BERTopic, evaluation_docs: list) -> pd.DataFrame:
    """Calculate detailed metrics for top 20 topics."""
    # Process documents
    tokenized_docs = [doc.lower().split() for doc in evaluation_docs if doc.strip()]
    dictionary = Dictionary(tokenized_docs)
    
    # Get topic information
    topic_info = model.get_topic_info()
    
    # Get top 20 topics by size (excluding outlier topic -1)
    top_20_topics = topic_info[topic_info['Topic'] != -1].nlargest(20, 'Count')['Topic'].tolist()
    
    # Always include outlier topic at the start if it exists
    if -1 in topic_info['Topic'].values:
        top_20_topics = [-1] + top_20_topics
    
    topics_data = []
    for topic in top_20_topics:
        topic_terms = model.get_topic(topic)
        words, weights = zip(*topic_terms)
        topic_name = topic_info.loc[topic_info['Topic'] == topic, 'Name'].iloc[0]
        
        # Calculate individual topic coherence
        topic_coherence = 0.0
        if topic != -1:
            valid_words = [word.lower() for word in words if word.lower() in dictionary.token2id]
            if len(valid_words) >= 3:
                try:
                    coherence_model = CoherenceModel(
                        topics=[valid_words],
                        texts=tokenized_docs,
                        dictionary=dictionary,
                        coherence='c_v'
                    )
                    topic_coherence = coherence_model.get_coherence()
                except Exception as e:
                    logger.error(f"Error calculating topic coherence: {e}")
        
        topics_data.append({
            'topic_id': topic,
            'topic_name': topic_name,
            'coherence_score': topic_coherence,
            'size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].values[0],
            'top_words': ", ".join(words),
            'word_weights': ", ".join([f"{w:.3f}" for w in weights])
        })
    
    return pd.DataFrame(topics_data)

def main():
    # Paths
    DATA_DIR = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/yearly_england"
    MODEL_PATH = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/evaluation_20241225_191414/incremental_model"  # Update this path
    OUTPUT_PATH = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/evaluation_20241225_191414"  # Update this path
    
    # Load documents (2010-2019)
    all_docs = []
    for year in range(2010, 2020):
        df = pd.read_csv(Path(DATA_DIR) / f"nltk_stopwords_preprocessed_england_speeches_{year}.csv")
        docs = [text.strip() for text in df['text'].astype(str) if text.strip()]
        all_docs.extend(docs)
    
    # Load the incremental model
    incremental_model = BERTopic.load(MODEL_PATH)
    
    # Calculate topic details
    topic_details = calculate_topic_details(incremental_model, all_docs)
    
    # Save results
    output_file = Path(OUTPUT_PATH) / 'top_20_topics_with_coherence.csv'
    topic_details.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()