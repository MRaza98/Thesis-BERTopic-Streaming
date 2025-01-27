import pandas as pd
from bertopic import BERTopic
import numpy as np
from datetime import datetime
from pathlib import Path

def compare_information_coverage(baseline_model: BERTopic, incremental_model: BERTopic) -> dict:
    """
    Analyze if incremental model captures the important topics from the baseline model
    """
    # Get topic info for both models
    baseline_topics = baseline_model.get_topic_info()
    incremental_topics = incremental_model.get_topic_info()
    
    # Get top 20 most significant topics from baseline model (excluding -1)
    important_baseline_topics = baseline_topics[baseline_topics['Topic'] != -1].nlargest(20, 'Count')
    
    coverage_analysis = []
    
    for _, baseline_topic in important_baseline_topics.iterrows():
        baseline_words = set(baseline_topic['Name'].split('_'))
        
        # Look for similar topics in incremental model
        matches = []
        for _, incr_topic in incremental_topics.iterrows():
            incr_words = set(incr_topic['Name'].split('_'))
            word_overlap = len(baseline_words.intersection(incr_words))
            if word_overlap >= 2:  # if at least 2 keywords match
                matches.append({
                    'incremental_topic': incr_topic['Name'],
                    'incremental_count': incr_topic['Count'],
                    'word_overlap': word_overlap
                })
        
        coverage_analysis.append({
            'baseline_topic': baseline_topic['Name'],
            'baseline_count': baseline_topic['Count'],
            'found_in_incremental': len(matches) > 0,
            'matched_topics': matches
        })
    
    topics_covered = sum(1 for x in coverage_analysis if x['found_in_incremental'])
    coverage_rate = topics_covered / len(important_baseline_topics)
    
    return {
        'detailed_analysis': coverage_analysis,
        'coverage_rate': coverage_rate,
        'topics_covered': topics_covered,
        'total_important_topics': len(important_baseline_topics)
    }

def analyze_topic_distributions(baseline_model: BERTopic, incremental_model: BERTopic) -> dict:
    """
    Compare the document distribution across topics between models
    """
    baseline_topics = baseline_model.get_topic_info()
    incremental_topics = incremental_model.get_topic_info()
    
    # Calculate statistics excluding outlier topic (-1)
    baseline_no_outlier = baseline_topics[baseline_topics['Topic'] != -1]
    incremental_no_outlier = incremental_topics[incremental_topics['Topic'] != -1]
    
    stats = {
        'baseline': {
            'total_documents': baseline_topics['Count'].sum(),
            'docs_in_outlier': baseline_topics[baseline_topics['Topic'] == -1]['Count'].sum(),
            'docs_in_topics': baseline_no_outlier['Count'].sum(),
            'avg_docs_per_topic': baseline_no_outlier['Count'].mean(),
            'std_docs_per_topic': baseline_no_outlier['Count'].std(),
            'num_topics': len(baseline_no_outlier),
            'outlier_percentage': (baseline_topics[baseline_topics['Topic'] == -1]['Count'].sum() / 
                                 baseline_topics['Count'].sum() * 100)
        },
        'incremental': {
            'total_documents': incremental_topics['Count'].sum(),
            'docs_in_outlier': incremental_topics[incremental_topics['Topic'] == -1]['Count'].sum(),
            'docs_in_topics': incremental_no_outlier['Count'].sum(),
            'avg_docs_per_topic': incremental_no_outlier['Count'].mean(),
            'std_docs_per_topic': incremental_no_outlier['Count'].std(),
            'num_topics': len(incremental_no_outlier),
            'outlier_percentage': (incremental_topics[incremental_topics['Topic'] == -1]['Count'].sum() / 
                                 incremental_topics['Count'].sum() * 100)
        }
    }
    
    return stats

def analyze_topic_overlap(baseline_model: BERTopic, incremental_model: BERTopic, min_overlap: int = 2) -> pd.DataFrame:
    """
    Find matching topics between models based on word overlap
    """
    baseline_topics = baseline_model.get_topic_info()
    incremental_topics = incremental_model.get_topic_info()
    
    # Exclude outlier topics
    baseline_topics = baseline_topics[baseline_topics['Topic'] != -1]
    incremental_topics = incremental_topics[incremental_topics['Topic'] != -1]
    
    matching_topics = []
    
    for _, baseline_topic in baseline_topics.iterrows():
        baseline_words = set(baseline_topic['Name'].split('_'))
        
        for _, incr_topic in incremental_topics.iterrows():
            incr_words = set(incr_topic['Name'].split('_'))
            word_overlap = len(baseline_words.intersection(incr_words))
            
            if word_overlap >= min_overlap:
                matching_topics.append({
                    'baseline_topic': baseline_topic['Name'],
                    'baseline_count': baseline_topic['Count'],
                    'incremental_topic': incr_topic['Name'],
                    'incremental_count': incr_topic['Count'],
                    'word_overlap': word_overlap,
                    'count_ratio': incr_topic['Count'] / baseline_topic['Count']
                })
    
    return pd.DataFrame(matching_topics)

def save_analysis_results(output_file: str, 
                        coverage: dict, 
                        dist_stats: dict, 
                        matching_topics: pd.DataFrame):
    """Save all analysis results to a file"""
    with open(output_file, 'w') as f:
        f.write("Baseline vs Incremental Model Comparison\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        # Topic Coverage Analysis
        f.write("1. Topic Coverage Analysis\n")
        f.write("-"*30 + "\n")
        f.write(f"Coverage Rate: {coverage['coverage_rate']:.2%}\n")
        f.write(f"Topics Covered: {coverage['topics_covered']} out of {coverage['total_important_topics']}\n\n")
        
        # Topic Distribution Analysis
        f.write("2. Topic Distribution Analysis\n")
        f.write("-"*30 + "\n")
        f.write("Baseline Model:\n")
        for key, value in dist_stats['baseline'].items():
            f.write(f"{key}: {value:,.2f}\n")
        f.write("\nIncremental Model:\n")
        for key, value in dist_stats['incremental'].items():
            f.write(f"{key}: {value:,.2f}\n")
        f.write("\n")
        
        # Matching Topics Analysis
        f.write("3. Topic Overlap Analysis\n")
        f.write("-"*30 + "\n")
        f.write(f"Found {len(matching_topics)} topic pairs with at least 2 word overlap\n\n")
        
        # Top matches by overlap
        f.write("Top 10 matches by word overlap:\n")
        top_matches = matching_topics.nlargest(10, 'word_overlap')
        for _, match in top_matches.iterrows():
            f.write(f"\nBaseline: {match['baseline_topic']} ({match['baseline_count']} docs)\n")
            f.write(f"Incremental: {match['incremental_topic']} ({match['incremental_count']} docs)\n")
            f.write(f"Word overlap: {match['word_overlap']}, Count ratio: {match['count_ratio']:.2f}\n")
        f.write("\n")
        
        # Detailed Coverage Analysis
        f.write("4. Detailed Coverage Analysis\n")
        f.write("-"*30 + "\n")
        for analysis in coverage['detailed_analysis']:
            f.write(f"\nBaseline Topic: {analysis['baseline_topic']} (Count: {analysis['baseline_count']})\n")
            f.write(f"Found in Incremental: {analysis['found_in_incremental']}\n")
            if analysis['matched_topics']:
                f.write("Matching topics in incremental model:\n")
                for match in analysis['matched_topics']:
                    f.write(f"- {match['incremental_topic']} (Count: {match['incremental_count']}, "
                           f"Word Overlap: {match['word_overlap']})\n")

def main():
    # Load models
    baseline_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/evaluation_20241225_191414/baseline_model"
    incremental_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/evaluation_20241225_191414/incremental_model"
    
    baseline_model = BERTopic.load(baseline_path)
    incremental_model = BERTopic.load(incremental_path)
    
    # Run analyses
    coverage = compare_information_coverage(baseline_model, incremental_model)
    dist_stats = analyze_topic_distributions(baseline_model, incremental_model)
    matching_topics = analyze_topic_overlap(baseline_model, incremental_model)
    
    # Save results
    output_file = "baseline_vs_incremental_analysis.txt"
    save_analysis_results(output_file, coverage, dist_stats, matching_topics)
    print(f"Analysis results have been saved to {output_file}")

if __name__ == "__main__":
    main()