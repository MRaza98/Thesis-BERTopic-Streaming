import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
import os
from datetime import datetime
from pathlib import Path
import logging
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from umap import UMAP
from sentence_transformers import SentenceTransformer
import torch
import copy

# Getting rid of the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'bertopic_evaluation_{timestamp}.log'
    
    # Configure logging
    logger = logging.getLogger('BERTopicEvaluator')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

class BERTopicEvaluator:
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the BERTopic evaluator.
        
        Args:
            data_dir: Directory containing yearly CSV files (2010-2019)
            output_dir: Directory to save evaluation results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(self.output_dir)
        
        # Store all documents by year for coherence calculations
        self.docs_by_year = {}
        for year in range(2010, 2020):
            df = pd.read_csv(self.data_dir / f"nltk_stopwords_preprocessed_england_speeches_{year}.csv")
            self.docs_by_year[year] = [text.strip() for text in df['text'].astype(str) if text.strip()]
            self.logger.info(f"Loaded {len(self.docs_by_year[year])} documents for year {year}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-miniLM-L6-v2")
        
        # Initialize UMAP with reference configuration
        self.umap_model = UMAP(
            n_components=8,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
    
    def get_evaluation_docs(self, up_to_year: int = None) -> List[str]:
        """Get evaluation documents up to specified year, or all if not specified."""
        if up_to_year is None:
            # For batch model, use all documents
            all_docs = []
            for year_docs in self.docs_by_year.values():
                all_docs.extend(year_docs)
            return all_docs
        else:
            # For incremental model, use documents up to specified year
            docs_subset = []
            for year in range(2010, up_to_year + 1):
                docs_subset.extend(self.docs_by_year[year])
            return docs_subset

    def calculate_model_coherence(self, model: BERTopic, evaluation_docs: List[str]) -> tuple:
        """Calculate model coherence using all valid topics."""
        start_time = time.time()

        # Process evaluation documents
        tokenized_docs = [doc.lower().split() for doc in evaluation_docs if doc.strip()]
        dictionary = Dictionary(tokenized_docs)
        
        # Get all non-outlier topics
        topic_info = model.get_topic_info()
        topic_words = []
        
        # Collect words from all topics
        for topic in topic_info['Topic']:
            if topic != -1:  # Skip outlier topic
                topic_terms = model.get_topic(topic)
                if topic_terms:  # Check if we got any terms
                    words = [word.lower() for word, _ in topic_terms]
                    valid_words = [word for word in words if word in dictionary.token2id]
                    if len(valid_words) >= 3:  # Only include topics with at least 3 valid words
                        topic_words.append(valid_words)
        
        # Calculate single coherence score for all topics
        if topic_words:
            try:
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()
            except Exception as e:
                self.logger.error(f"Error calculating coherence: {e}")
                coherence = 0.0
        else:
            coherence = 0.0
        
        calculation_time = time.time() - start_time
        self.logger.info(f"Coherence calculation used {len(topic_words)} valid topics")
        return coherence, calculation_time

    def calculate_topic_stability(self, model1: BERTopic, model2: BERTopic) -> float:
        """Calculate topic stability between two consecutive models."""
        topics1 = model1.get_topics()
        topics2 = model2.get_topics()
        
        # Compare topic similarities
        similarities = []
        for topic1_id, topic1_words in topics1.items():
            if topic1_id != -1:  # Skip outlier topic
                topic1_words = [word for word, _ in topic1_words]
                max_similarity = 0
                for topic2_id, topic2_words in topics2.items():
                    if topic2_id != -1:
                        topic2_words = [word for word, _ in topic2_words]
                        similarity = len(set(topic1_words) & set(topic2_words)) / len(set(topic1_words) | set(topic2_words))
                        max_similarity = max(max_similarity, similarity)
                similarities.append(max_similarity)
        
        return np.mean(similarities) if similarities else 0.0

    def train_baseline_model(self, docs: List[str]) -> Tuple[BERTopic, float]:
        """Train baseline model on all data."""
        self.logger.info("Training baseline model...")
        start_time = time.time()
        
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            verbose=True
        )
        
        topics, _ = topic_model.fit_transform(docs)
        training_time = time.time() - start_time
        
        # Get evaluation documents for batch model (all documents)
        eval_docs = self.get_evaluation_docs()
        coherence, coherence_time = self.calculate_model_coherence(topic_model, eval_docs)
        
        self.logger.info(f"Baseline model training completed in {training_time:.2f} seconds")
        self.logger.info(f"Baseline coherence: {coherence:.4f}")
        
        return topic_model, coherence

    def evaluate_incremental_model(self, 
                             baseline_model: BERTopic,
                             baseline_coherence: float,
                             start_year: int = 2010,
                             end_year: int = 2019) -> Tuple[List[Dict], BERTopic]:
        """Train and merge models incrementally year by year."""
        results = []
        current_merged_model = None
        
        for year in range(start_year, end_year + 1):
            self.logger.info(f"Training model for year {year}...")
            year_docs = self.docs_by_year[year]
            
            # Train model for current year
            year_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                verbose=True
            )
            
            start_time = time.time()
            topics, _ = year_model.fit_transform(year_docs)
            training_time = time.time() - start_time
            
            # Save year model details
            topic_info = year_model.get_topic_info()
            model_details = pd.DataFrame([{
                'topic_id': row['Topic'],
                'topic_name': row['Name'],
                'size': row['Count']
            } for _, row in topic_info.iterrows()])
            
            model_details.to_csv(self.output_dir / f'topics_{year}_details.csv', index=False)
            
            # Merge with previous years if exists
            if current_merged_model is None:
                current_merged_model = year_model
            else:
                self.logger.info(f"Merging model for year {year} with previous years...")
                
                # Store previous state for stability comparison
                previous_model = copy.deepcopy(current_merged_model)
                
                merge_start_time = time.time()
                current_merged_model = BERTopic.merge_models(
                    [current_merged_model, year_model]
                )
                merge_time = time.time() - merge_start_time
                
                # Calculate stability
                stability = self.calculate_topic_stability(previous_model, current_merged_model)
                self.logger.info(f"Topic stability after year {year}: {stability:.4f}")
                
                # Calculate intermediate coherence
                eval_docs = self.get_evaluation_docs(year)
                intermediate_coherence, coherence_time = self.calculate_model_coherence(
                    current_merged_model, eval_docs
                )
                
                self.logger.info(f"Intermediate coherence after year {year}: {intermediate_coherence:.4f}")
            
            # Record results for this year
            results.append({
                'year': year,
                'num_topics': len(year_model.get_topics()),
                'training_time': training_time
            })
            
            # Clear memory
            del year_model
            
        # Calculate final metrics
        final_eval_docs = self.get_evaluation_docs(end_year)
        final_coherence, coherence_time = self.calculate_model_coherence(
            current_merged_model, final_eval_docs
        )
        
        results.append({
            'year': f"{start_year}-{end_year}_merged",
            'num_topics': len(current_merged_model.get_topics()),
            'coherence': final_coherence,
            'baseline_coherence': baseline_coherence,
            'coherence_delta': final_coherence - baseline_coherence,
            'coherence_calc_time': coherence_time
        })
        
        return results, current_merged_model

    def save_results(self, results: List[Dict], 
                    baseline_model: BERTopic,
                    incremental_model: BERTopic):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_dir / f"evaluation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
        
        # Save models using BERTopic's native save method
        baseline_model.save(output_dir / 'baseline_model')
        incremental_model.save(output_dir / 'incremental_model')
        
        # Save detailed topic information for both models
        baseline_details = self.calculate_topic_details(
            baseline_model, 
            self.get_evaluation_docs()
        )
        baseline_details.to_csv(output_dir / 'baseline_topics_details.csv', index=False)
        
        incremental_details = self.calculate_topic_details(
            incremental_model, 
            self.get_evaluation_docs(2019)
        )
        incremental_details.to_csv(output_dir / 'incremental_topics_details.csv', index=False)
        
        self.logger.info(f"Results saved to {output_dir}")

    def calculate_topic_details(self, model: BERTopic, evaluation_docs: List[str]) -> pd.DataFrame:
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
                        self.logger.error(f"Error calculating topic coherence: {e}")
            
            topics_data.append({
                'topic_id': topic,
                'topic_name': topic_name,
                'coherence_score': topic_coherence,
                'size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].values[0],
                'top_words': ", ".join(words),
                'word_weights': ", ".join([f"{w:.3f}" for w in weights])
            })
        
        return pd.DataFrame(topics_data)

def train_baseline_model(self, docs: List[str]) -> Tuple[BERTopic, float]:
    """Train baseline model on all data."""
    self.logger.info("Training baseline model...")
    start_time = time.time()
    
    topic_model = BERTopic(
        embedding_model=self.embedding_model,
        umap_model=self.umap_model,
        verbose=True
    )
    
    topics, _ = topic_model.fit_transform(docs)
    training_time = time.time() - start_time
    
    # Get evaluation documents for batch model (all documents)
    eval_docs = self.get_evaluation_docs()
    coherence, coherence_time = self.calculate_model_coherence(topic_model, eval_docs)
    
    self.logger.info(f"Baseline model training completed in {training_time:.2f} seconds")
    self.logger.info(f"Baseline coherence: {coherence:.4f}")
    
    return topic_model, coherence

def evaluate_incremental_model(self, 
                         baseline_model: BERTopic,
                         baseline_coherence: float,
                         start_year: int = 2010,
                         end_year: int = 2019) -> Tuple[List[Dict], BERTopic]:
    """Train and merge models incrementally year by year."""
    results = []
    current_merged_model = None
    
    for year in range(start_year, end_year + 1):
        self.logger.info(f"Training model for year {year}...")
        year_docs = self.docs_by_year[year]
        
        # Train model for current year
        year_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            verbose=True
        )
        
        start_time = time.time()
        topics, _ = year_model.fit_transform(year_docs)
        training_time = time.time() - start_time
        
        # Save year model details
        topic_info = year_model.get_topic_info()
        model_details = pd.DataFrame([{
            'topic_id': row['Topic'],
            'topic_name': row['Name'],
            'size': row['Count']
        } for _, row in topic_info.iterrows()])
        
        model_details.to_csv(self.output_dir / f'topics_{year}_details.csv', index=False)
        
        # Merge with previous years if exists
        if current_merged_model is None:
            current_merged_model = year_model
        else:
            self.logger.info(f"Merging model for year {year} with previous years...")
            
            # Store previous state for stability comparison
            previous_model = copy.deepcopy(current_merged_model)
            
            merge_start_time = time.time()
            current_merged_model = BERTopic.merge_models(
                [current_merged_model, year_model]
            )
            merge_time = time.time() - merge_start_time
            
            # Calculate stability
            stability = self.calculate_topic_stability(previous_model, current_merged_model)
            self.logger.info(f"Topic stability after year {year}: {stability:.4f}")
            
            # Calculate intermediate coherence
            eval_docs = self.get_evaluation_docs(year)
            intermediate_coherence, coherence_time = self.calculate_model_coherence(
                current_merged_model, eval_docs
            )
            
            self.logger.info(f"Intermediate coherence after year {year}: {intermediate_coherence:.4f}")
            self.logger.info(f"Merge completed in {merge_time:.2f} seconds")
        
        # Record results for this year
        results.append({
            'year': year,
            'num_topics': len(year_model.get_topics()),
            'training_time': training_time,
            'model_size': len(year_model.get_topics())
        })
        
        # Clear memory
        del year_model
        
    # Calculate final metrics
    final_eval_docs = self.get_evaluation_docs(end_year)
    final_coherence, coherence_time = self.calculate_model_coherence(
        current_merged_model, final_eval_docs
    )
    
    results.append({
        'year': f"{start_year}-{end_year}_merged",
        'num_topics': len(current_merged_model.get_topics()),
        'coherence': final_coherence,
        'baseline_coherence': baseline_coherence,
        'coherence_delta': final_coherence - baseline_coherence,
        'coherence_calc_time': coherence_time
    })
    
    return results, current_merged_model

def save_results(self, results: List[Dict], 
                baseline_model: BERTopic,
                incremental_model: BERTopic):
    """Save evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = self.output_dir / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
    
    # Save models using BERTopic's native save method
    baseline_path = output_dir / 'baseline_model'
    incremental_path = output_dir / 'incremental_model'
    
    self.logger.info(f"Saving baseline model to {baseline_path}")
    baseline_model.save(baseline_path)
    
    self.logger.info(f"Saving incremental model to {incremental_path}")
    incremental_model.save(incremental_path)
    
    # Save topic information
    try:
        # Get and save baseline model topics
        baseline_info = baseline_model.get_topic_info()
        baseline_info.to_csv(output_dir / 'baseline_topic_info.csv', index=False)
        
        # Get and save incremental model topics
        incremental_info = incremental_model.get_topic_info()
        incremental_info.to_csv(output_dir / 'incremental_topic_info.csv', index=False)
        
        self.logger.info("Topic information saved successfully")
    except Exception as e:
        self.logger.error(f"Error saving topic information: {e}")
    
    self.logger.info(f"All results saved to {output_dir}")
    return output_dir

def main():
    # Set paths
    DATA_DIR = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/yearly_england"
    OUTPUT_DIR = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/Iteration_incremental_approach"
    
    try:
        # Create evaluator
        evaluator = BERTopicEvaluator(DATA_DIR, OUTPUT_DIR)
        evaluator.logger.info("Starting BERTopic evaluation pipeline...")
        
        # Get all documents for baseline
        all_docs = evaluator.get_evaluation_docs()
        evaluator.logger.info(f"Total documents for baseline: {len(all_docs)}")
        
        # Train baseline model
        baseline_model, baseline_coherence = evaluator.train_baseline_model(all_docs)
        evaluator.logger.info(f"Baseline model trained with coherence: {baseline_coherence:.4f}")
        
        # Run incremental evaluation
        evaluator.logger.info("Starting incremental evaluation...")
        results, incremental_model = evaluator.evaluate_incremental_model(
            baseline_model,
            baseline_coherence,
            start_year=2010,
            end_year=2019
        )
        
        # Save results
        output_dir = evaluator.save_results(results, baseline_model, incremental_model)
        
        evaluator.logger.info("\nEvaluation pipeline completed successfully!")
        evaluator.logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        evaluator.logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()