import pandas as pd
import numpy as np
from typing import List, Dict
import time
from datetime import datetime
import os
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from umap import UMAP
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TimingTracker:
    def __init__(self):
        self.timings = {}
        self.current_operations = {}
        
    def start(self, operation_name: str):
        self.current_operations[operation_name] = time.time()
        
    def stop(self, operation_name: str):
        if operation_name in self.current_operations:
            elapsed = time.time() - self.current_operations[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.current_operations[operation_name]
            return elapsed
        return 0
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for operation, times in self.timings.items():
            summary[operation] = {
                'total': sum(times),
                'mean': np.mean(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        return summary
    
    def save_summary(self, output_dir: str):
        summary = self.get_summary()
        df_data = []
        for operation, metrics in summary.items():
            metrics['operation'] = operation
            df_data.append(metrics)
        
        df = pd.DataFrame(df_data)
        df = df[['operation', 'total', 'mean', 'min', 'max', 'count']]
        df.to_csv(f'{output_dir}/timing_summary.csv', index=False)
        
        # Also save detailed timings
        detailed_data = []
        for operation, times in self.timings.items():
            for i, t in enumerate(times):
                detailed_data.append({
                    'operation': operation,
                    'iteration': i + 1,
                    'time': t
                })
        pd.DataFrame(detailed_data).to_csv(f'{output_dir}/detailed_timings.csv', index=False)

# Create global timing tracker
timer = TimingTracker()

def calculate_topic_coherence(topic_words: List[str], tokenized_docs: List[List[str]], 
                            dictionary: Dictionary) -> tuple:
    """Calculate coherence score for a single topic"""
    timer.start('individual_topic_coherence')
    try:
        valid_words = [word for word in topic_words if word in dictionary.token2id]
        if len(valid_words) < 3:
            timer.stop('individual_topic_coherence')
            return 0.0, 0.0
            
        coherence_model = CoherenceModel(
            topics=[valid_words],
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        elapsed = timer.stop('individual_topic_coherence')
        return coherence_score, elapsed
    except:
        elapsed = timer.stop('individual_topic_coherence')
        return 0.0, elapsed

def save_model_results(model_results: dict, output_dir: str, model: BERTopic = None, 
                      tokenized_docs: List[List[str]] = None, dictionary: Dictionary = None):
    """Save model results and calculate per-topic metrics"""
    timer.start('save_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model metrics
    metrics_df = pd.DataFrame({
        'num_topics': [model_results['num_topics']],
        'coherence_score': [model_results['coherence_score']],
        'training_time': [model_results['training_time']],
        'coherence_calculation_time': [model_results['coherence_calculation_time']]
    })
    metrics_df.to_csv(f'{output_dir}/model_metrics.csv', index=False)
    
    if model is not None and tokenized_docs is not None and dictionary is not None:
        timer.start('topic_analysis')
        topic_info = model.get_topic_info()
        top_20_topics = topic_info[topic_info['Topic'] != -1].nlargest(20, 'Count')['Topic'].tolist()
        
        if -1 in topic_info['Topic'].values:
            top_20_topics = [-1] + top_20_topics
        
        topics_data = []
        for topic in top_20_topics:
            timer.start(f'process_topic_{topic}')
            words, weights = zip(*model.get_topic(topic))
            topic_name = topic_info.loc[topic_info['Topic'] == topic, 'Name'].iloc[0]
            
            topic_coherence = 0.0
            coherence_time = 0.0
            if topic != -1:
                valid_words = [word.lower() for word in words if word.lower() in dictionary.token2id]
                if len(valid_words) >= 3:
                    topic_coherence, coherence_time = calculate_topic_coherence(
                        valid_words,
                        tokenized_docs,
                        dictionary
                    )
            
            topics_data.append({
                'topic_id': topic,
                'topic_name': topic_name,
                'cv_score': topic_coherence,
                'coherence_calculation_time': coherence_time,
                'size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].values[0],
                'top_words': ", ".join(words),
                'word_weights': ", ".join([f"{w:.3f}" for w in weights])
            })
            timer.stop(f'process_topic_{topic}')
        
        timer.stop('topic_analysis')
        
        topics_df = pd.DataFrame(topics_data)
        topics_df.to_csv(f'{output_dir}/topic_details.csv', index=False)
    
    timer.stop('save_results')

def train_and_evaluate_model(train_documents: List[str], 
                           coherence_documents: List[str],
                           output_dir: str) -> tuple:
    """Main function to train and evaluate the model"""
    timer.start('total_execution')
    
    print("Preparing documents for coherence calculation...")
    timer.start('document_preparation')
    tokenized_coherence_docs = [doc.lower().split() for doc in coherence_documents]
    coherence_dictionary = Dictionary(tokenized_coherence_docs)
    timer.stop('document_preparation')
    print(f"Dictionary size: {len(coherence_dictionary)}")

    timer.start('model_initialization')
    model = SentenceTransformer("all-miniLM-L6-v2")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/bertopic_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    timer.stop('model_initialization')
    
    print(f"Results will be saved in: {output_dir}")
    print(f"Number of documents for training: {len(train_documents)}")
    print(f"Number of documents for coherence calculation: {len(coherence_documents)}")
    
    try:
        # UMAP initialization
        timer.start('umap_init')
        umap_model = UMAP(
            n_components=8,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        timer.stop('umap_init')
        
        # BERTopic initialization and training
        timer.start('bertopic_init')
        topic_model = BERTopic(embedding_model=model,
                             umap_model=umap_model,
                             verbose=True)
        timer.stop('bertopic_init')
        
        # Model fitting
        timer.start('model_fitting')
        topics, _ = topic_model.fit_transform(train_documents)
        model_fitting_time = timer.stop('model_fitting')
        
        # Topic analysis
        timer.start('topic_extraction')
        topic_info = topic_model.get_topic_info()
        num_topics = len([topic for topic in topic_info['Topic'] if topic != -1])
        
        topic_words = []
        for topic in topic_info['Topic']:
            if topic != -1:
                words, _ = zip(*topic_model.get_topic(topic))
                words = [word.lower() for word in words]
                valid_words = [word for word in words if word in coherence_dictionary.token2id]
                if len(valid_words) >= 3:
                    topic_words.append(valid_words)
        timer.stop('topic_extraction')
        
        # Coherence calculation
        timer.start('coherence_calculation')
        if topic_words:
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=tokenized_coherence_docs,
                dictionary=coherence_dictionary,
                coherence='c_v'
            )
            c_v_score = coherence_model.get_coherence()
        else:
            c_v_score = np.nan
        coherence_time = timer.stop('coherence_calculation')
        
        model_results = {
            'num_topics': num_topics,
            'coherence_score': c_v_score,
            'training_time': model_fitting_time,
            'coherence_calculation_time': coherence_time
        }
        
        save_model_results(model_results, output_dir, topic_model, tokenized_coherence_docs, coherence_dictionary)
        
        # Print timing summary
        print("\nTiming Summary:")
        summary = timer.get_summary()
        for operation, metrics in summary.items():
            print(f"\n{operation}:")
            print(f"  Total time: {metrics['total']:.2f}s")
            print(f"  Mean time: {metrics['mean']:.2f}s")
            print(f"  Min time: {metrics['min']:.2f}s")
            print(f"  Max time: {metrics['max']:.2f}s")
            print(f"  Iterations: {metrics['count']}")
        
        # Save timing data
        timer.save_summary(output_dir)
        
        timer.stop('total_execution')
        return model_results, output_dir
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        timer.stop('total_execution')
        return None, output_dir

if __name__ == "__main__":
    print("Starting BERTopic analysis...")
    timer.start('data_loading')
    
    # Load training data (2018 + Jan-Feb 2019)
    train_df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/nltk_stopwords_preprocessed_england_speeches_2018_2019_JF.csv')
    print("\nTraining data loaded from: nltk_stopwords_preprocessed_england_speeches_2018_2019_JF.csv")
    print(f"Number of training documents: {len(train_df)}")
    
    # Load coherence evaluation data (2015-2019)
    coherence_df = pd.read_csv('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/nltk_stopwords_preprocessed_england_speeches_2015_2019.csv')
    print("\nCoherence data loaded from: nltk_stopwords_preprocessed_england_speeches_2015_2019.csv")
    print(f"Number of coherence evaluation documents: {len(coherence_df)}")
    
    timer.stop('data_loading')
    
    # Prepare training documents
    timer.start('document_preprocessing')
    train_df['text'] = train_df['text'].fillna('')
    train_df['text'] = train_df['text'].astype(str)
    train_documents = [text.strip() for text in train_df['text'] if text.strip()]
    print(f"Final number of training documents after cleaning: {len(train_documents)}")
    
    # Prepare coherence documents
    coherence_df['text'] = coherence_df['text'].fillna('')
    coherence_df['text'] = coherence_df['text'].astype(str)
    coherence_documents = [text.strip() for text in coherence_df['text'] if text.strip()]
    print(f"Final number of coherence documents after cleaning: {len(coherence_documents)}")
    timer.stop('document_preprocessing')
    
    output_dir = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_4"
    results, final_output_dir = train_and_evaluate_model(
        train_documents,
        coherence_documents,
        output_dir
    )
    
    if results:
        print("\nAnalysis complete! Results saved in:", final_output_dir)
        print("\nTiming files saved:")
        print("- timing_summary.csv: Aggregated timing statistics")
        print("- detailed_timings.csv: Individual timing measurements")