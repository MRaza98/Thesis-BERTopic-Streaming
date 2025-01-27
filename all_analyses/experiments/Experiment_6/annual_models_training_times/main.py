import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
import time
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
import gc

torch.cuda.set_device(3)

class TrainingTimeAnalysis:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.model_config = {
            'embedding_model': "all-miniLM-L6-v2",
            'umap_components': 8,
            'umap_neighbors': 15,
            'umap_min_dist': 0.0,
            'random_state': 42
        }
        
    def _initialize_model(self):
        model = SentenceTransformer(self.model_config['embedding_model'])
        umap_model = UMAP(
            n_components=self.model_config['umap_components'],
            n_neighbors=self.model_config['umap_neighbors'],
            min_dist=self.model_config['umap_min_dist'],
            metric='cosine',
            random_state=self.model_config['random_state']
        )
        return model, umap_model
    
    def load_yearly_data(self, year: int) -> pd.DataFrame:
        filepath = Path("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/data/yearly_england") / f"nltk_stopwords_preprocessed_england_speeches_{year}.csv"
        data = pd.read_csv(filepath)
        return data[['text']].dropna()
    
    def train_and_measure(self, start_year: int = 2000, end_year: int = 2019):
        training_times = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nTraining model for year {year}")
            gc.collect()
            torch.cuda.empty_cache()
            
            # Load data and initialize model
            current_data = self.load_yearly_data(year)
            model, umap_model = self._initialize_model()
            topic_model = BERTopic(
                embedding_model=model,
                umap_model=umap_model,
                verbose=True
            )
            
            # Train and measure time
            start_time = time.time()
            topics, _ = topic_model.fit_transform(current_data['text'].tolist())
            training_time = time.time() - start_time
            
            training_times.append({
                'year': year,
                'training_time': training_time
            })
            
            # Clean up
            del topic_model, model, umap_model
            
        # Calculate average and save results
        df = pd.DataFrame(training_times)
        df['training_time'] = df['training_time'] / 60  # Convert to minutes
        average_time = df['training_time'].mean()
        
        # Add average row
        df.loc[len(df)] = ['Average', average_time]
        
        # Save results
        output_file = self.base_path / 'training_times.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Average training time: {average_time:.2f} minutes")

def main():
    output_dir = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_4/Iteration_9_Average_Training_Time_Ind_Models"
    analyzer = TrainingTimeAnalysis(output_dir)
    analyzer.train_and_measure()

if __name__ == "__main__":
    main()