import pandas as pd
import time
import hashlib
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from bertopic import BERTopic
import os

class BERTopicExperimentTracker:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use a single experiments file for all models
        self.experiments_file = os.path.join(output_dir, "bertopic_experiments.csv")
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.current_experiment = {}
        self.timer_start = {}
        
        try:
            self.experiments_df = pd.read_csv(self.experiments_file)
            # Ensure 'evaluation_sample' column exists in loaded DataFrame
            if 'evaluation_sample' not in self.experiments_df.columns:
                self.experiments_df['evaluation_sample'] = pd.NA
        except FileNotFoundError:
            self.experiments_df = pd.DataFrame(columns=[
                'timestamp', 'model_id', 'model_hashkey', 'embedding_model',
                'c_v_score', 'total_training_time', 'coherence_calc_time',
                'num_topics', 'evaluation_sample'
            ])
    
    def start_timer(self, stage: str):
        """Start timing a specific stage of the process."""
        self.timer_start[stage] = time.time()
    
    def stop_timer(self, stage: str) -> float:
        """Stop timing a specific stage and return elapsed time."""
        if stage in self.timer_start:
            elapsed = time.time() - self.timer_start[stage]
            self.current_experiment[f'{stage}_time'] = elapsed
            return elapsed
        return 0
    
    def generate_model_hash(self, model_params: Dict[str, Any]) -> str:
        """Generate a unique hash for the model based on its parameters."""
        param_str = json.dumps(model_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def calculate_coherence_scores(self, topic_model, docs):
        """Calculate coherence scores for all non-outlier topics."""
        self.start_timer('coherence_calc')
        
        topic_info = topic_model.get_topic_info()
        texts = [doc.split() for doc in docs]
        dictionary = corpora.Dictionary(texts)
        
        topic_words = []
        
        for topic in topic_info['Topic']:
            if topic != -1:  # Skip outlier topic
                words, _ = zip(*topic_model.get_topic(topic))
                valid_words = [word for word in words if word in dictionary.token2id]
                if len(valid_words) >= 3:
                    topic_words.append(valid_words)
        
        if not topic_words:
            self.stop_timer('coherence_calc')
            return pd.DataFrame(), np.nan
            
        try:
            cv_coherence_model = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            cv_overall_score = cv_coherence_model.get_coherence()
        except:
            cv_overall_score = np.nan
        
        self.stop_timer('coherence_calc')
        return pd.DataFrame(), cv_overall_score
    
    def log_experiment(self, 
                  model_params: Dict[str, Any],
                  topic_model,
                  docs,
                  model_id: str = None):
        """Log complete experiment with all metrics and topic details."""
        model_id = model_id or f"model_{len(self.experiments_df) + 1}"
        
        # Create unique topics file for this experiment
        topics_file = os.path.join(self.output_dir, f"topic_details_{model_id}_{self.timestamp}.csv")
        
        # Create or load topics DataFrame for this experiment
        try:
            topics_df = pd.read_csv(topics_file)
        except FileNotFoundError:
            topics_df = pd.DataFrame(columns=[
                'model_id', 'topic_id', 'cv_score', 'size', 
                'top_words', 'word_weights'
            ])
        
        self.current_experiment = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_id': model_id,
            'model_hashkey': self.generate_model_hash(model_params),
            'embedding_model': model_params.get('embedding_model', 'unknown'),
            'num_topics': len(topic_model.get_topics()),
            'total_training_time': model_params.get('total_training_time', np.nan),
            'evaluation_sample': pd.NA  # Initialize as NA, can be filled manually later
        }
        
        # Calculate coherence scores
        topic_coherence_df, overall_coherence = self.calculate_coherence_scores(topic_model, docs)
        self.current_experiment['c_v_score'] = overall_coherence
        self.current_experiment['coherence_calc_time'] = self.current_experiment.get('coherence_calc_time', np.nan)

        if not topic_coherence_df.empty:
            topic_coherence_df['model_id'] = model_id
            topics_df = pd.concat([topics_df, topic_coherence_df], ignore_index=True)
            topics_df.to_csv(topics_file, index=False)
        
        # Read the existing experiments file again to ensure we have the latest data
        try:
            existing_experiments = pd.read_csv(self.experiments_file)
            # Ensure evaluation_sample column exists
            if 'evaluation_sample' not in existing_experiments.columns:
                existing_experiments['evaluation_sample'] = pd.NA
        except FileNotFoundError:
            existing_experiments = pd.DataFrame(columns=self.experiments_df.columns)
        
        # Append the new experiment
        updated_experiments = pd.concat([
            existing_experiments,
            pd.DataFrame([self.current_experiment])
        ], ignore_index=True)
        
        # Save experiments to CSV
        updated_experiments.to_csv(self.experiments_file, index=False)
        
        # Store current topics_df for get_topics() method
        self.current_topics_df = topics_df
    
    def get_experiments(self) -> pd.DataFrame:
        """Return the DataFrame of all experiments."""
        return self.experiments_df
    
    def get_topics(self) -> pd.DataFrame:
        """Return the DataFrame of topics for the most recent experiment."""
        return self.current_topics_df