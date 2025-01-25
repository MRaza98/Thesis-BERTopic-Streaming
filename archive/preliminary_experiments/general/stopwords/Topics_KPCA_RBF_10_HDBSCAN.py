from bertopic import BERTopic
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model
logger.info("Loading the model...")
topic_model_mini = BERTopic.load("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_all_MiniLM_L6_v2_kpca_rbf_hdbscan_stopwords_Nonetopics_10dim_2024-10-25_19-05-16_7a0041")

def topic_distributions(model):
    """Analyze topic distribution statistics including outliers"""
    info = model.get_topic_info()
    
    print("\nTopic Distribution Statistics:")
    print("-" * 30)
    print(f"Total number of topics (including outliers): {len(info)}")
    print(f"Average topic size: {info['Count'].mean():.2f}")
    print(f"Median topic size: {info['Count'].median():.2f}")
    print(f"Largest topic size: {info['Count'].max()}")
    print(f"Smallest topic size: {info['Count'].min()}")
    
    # Statistics about outlier topic if it exists
    outlier_info = info[info['Topic'] == -1]
    if not outlier_info.empty:
        print(f"Outlier topic size: {outlier_info['Count'].iloc[0]}")
        print(f"Percentage of documents in outliers: {(outlier_info['Count'].iloc[0] / info['Count'].sum() * 100):.2f}%")
    
    return info

def analyze_top_topics(model, n_topics=10):
    """Analyze top N topics (including outliers) with their keywords and sizes"""
    info = model.get_topic_info()
    
    print(f"\nTop {n_topics} Topics Analysis (including outliers):")
    print("-" * 30)
    
    for i in range(min(n_topics, len(info))):
        topic_id = info.iloc[i]['Topic']
        topic_size = info.iloc[i]['Count']
        topic_name = info.iloc[i]['Name']
        
        print(f"\nTopic {topic_id}:")
        print(f"Size: {topic_size} documents")
        print(f"Name: {topic_name}")
        
        if topic_id != -1:
            topic_words = model.get_topic(topic_id)
            words = ", ".join([word for word, _ in topic_words[:5]])
            print(f"Top words: {words}")
        else:
            print("This is the outlier topic")

def create_detailed_csv(model, output_path):
    """Create a detailed CSV with topic information including outliers"""
    info = model.get_topic_info()
    
    detailed_topics = []
    
    for idx, row in info.iterrows():
        topic_dict = {
            'Topic_ID': row['Topic'],
            'Count': row['Count'],
            'Name': row['Name'],
            'Representation': row['Representation']
        }
        
        # Add top words for non-outlier topics
        if row['Topic'] != -1:
            topic_words = model.get_topic(row['Topic'])
            topic_dict['Top_Words'] = ', '.join([word for word, _ in topic_words[:5]])
        else:
            topic_dict['Top_Words'] = 'Outlier topic'
            
        detailed_topics.append(topic_dict)
    
    # Convert to DataFrame and save
    df_detailed = pd.DataFrame(detailed_topics)
    df_detailed.to_csv(output_path, index=False)
    logger.info(f"Saved detailed topic analysis to {output_path}")
    
    return df_detailed

# Run the analysis
if __name__ == "__main__":
    # Get topic distribution statistics
    topic_info = topic_distributions(topic_model_mini)
    
    # Analyze top topics
    analyze_top_topics(topic_model_mini, n_topics=10)
    
    # Create and save detailed CSV
    output_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/Analysis/stopwords/model_all_MiniLM_L6_v2_kpca_rbf_hdbscan_stopwords_Nonetopics_10dim_2024-10-25_19-05-16_7a0041.csv"
    detailed_df = create_detailed_csv(topic_model_mini, output_path)
    
    # Print full dataframe
    print("\nComplete Topic Analysis:")
    print("-" * 30)
    print(detailed_df)
    
    # Print total number of documents
    total_docs = detailed_df['Count'].sum()
    print(f"\nTotal number of documents: {total_docs}")