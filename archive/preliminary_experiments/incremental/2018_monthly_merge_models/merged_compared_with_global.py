import pandas as pd
from bertopic import BERTopic
import numpy as np
from collections import defaultdict
from datetime import datetime

def compare_information_coverage(global_model, merged_model):
    """
    Analyze if merged model captures the important topics/information 
    from the global model
    """
    # Get topic info for both models
    global_topics = global_model.get_topic_info()
    merged_topics = merged_model.get_topic_info()
    
    # Get the top N most significant topics from global model
    N = 20
    important_global_topics = global_topics.nlargest(N, 'Count')
    
    # For each important global topic, find if similar topic exists in merged model
    coverage_analysis = []
    
    for _, global_topic in important_global_topics.iterrows():
        topic_words = set(global_topic['Name'].split('_'))
        
        # Look for similar topics in merged model
        matches = []
        for _, merged_topic in merged_topics.iterrows():
            merged_words = set(merged_topic['Name'].split('_'))
            word_overlap = len(topic_words.intersection(merged_words))
            if word_overlap >= 2:  # if at least 2 keywords match
                matches.append({
                    'merged_topic': merged_topic['Name'],
                    'merged_count': merged_topic['Count'],
                    'word_overlap': word_overlap
                })
        
        coverage_analysis.append({
            'global_topic': global_topic['Name'],
            'global_count': global_topic['Count'],
            'found_in_merged': len(matches) > 0,
            'matched_topics': matches
        })
    
    # Calculate coverage metrics
    topics_covered = sum(1 for x in coverage_analysis if x['found_in_merged'])
    coverage_rate = topics_covered / len(important_global_topics)
    
    return {
        'detailed_analysis': coverage_analysis,
        'coverage_rate': coverage_rate,
        'topics_covered': topics_covered,
        'total_important_topics': len(important_global_topics)
    }

def analyze_topic_frequencies(global_model, merged_model):
    """
    Compare the frequency distributions of topics between models
    """
    global_topics = global_model.get_topic_info()
    merged_topics = merged_model.get_topic_info()
    
    # Calculate basic statistics
    stats = {
        'global': {
            'total_documents': global_topics['Count'].sum(),
            'avg_docs_per_topic': global_topics['Count'].mean(),
            'std_docs_per_topic': global_topics['Count'].std(),
            'num_topics': len(global_topics)
        },
        'merged': {
            'total_documents': merged_topics['Count'].sum(),
            'avg_docs_per_topic': merged_topics['Count'].mean(),
            'std_docs_per_topic': merged_topics['Count'].std(),
            'num_topics': len(merged_topics)
        }
    }
    
    return stats

def find_matching_topics(global_model, merged_model, min_word_overlap=2):
    """
    Find matching topics between models based on word overlap
    """
    global_topics = global_model.get_topic_info()
    merged_topics = merged_model.get_topic_info()
    
    matching_topics = []
    
    for _, global_topic in global_topics.iterrows():
        global_words = set(global_topic['Name'].split('_'))
        
        for _, merged_topic in merged_topics.iterrows():
            merged_words = set(merged_topic['Name'].split('_'))
            word_overlap = len(global_words.intersection(merged_words))
            
            if word_overlap >= min_word_overlap:
                matching_topics.append({
                    'global_topic': global_topic['Name'],
                    'global_count': global_topic['Count'],
                    'merged_topic': merged_topic['Name'],
                    'merged_count': merged_topic['Count'],
                    'word_overlap': word_overlap,
                    'count_difference': abs(global_topic['Count'] - merged_topic['Count'])
                })
    
    return pd.DataFrame(matching_topics)

def save_analysis_to_file(output_file, coverage, freq_stats, matching_topics):
    """Save all analysis results to a file"""
    with open(output_file, 'w') as f:
        f.write("Topic Model Comparison Analysis\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        # Information Coverage Analysis
        f.write("1. Information Coverage Analysis:\n")
        f.write("-"*30 + "\n")
        f.write(f"Coverage Rate: {coverage['coverage_rate']:.2%}\n")
        f.write(f"Topics Covered: {coverage['topics_covered']} out of {coverage['total_important_topics']}\n\n")
        
        # Topic Frequency Analysis
        f.write("2. Topic Frequency Analysis:\n")
        f.write("-"*30 + "\n")
        f.write("Global Model Statistics:\n")
        for key, value in freq_stats['global'].items():
            f.write(f"{key}: {value:,.2f}\n")
        f.write("\nMerged Model Statistics:\n")
        for key, value in freq_stats['merged'].items():
            f.write(f"{key}: {value:,.2f}\n")
        f.write("\n")
        
        # Matching Topics Analysis
        f.write("3. Matching Topics Analysis:\n")
        f.write("-"*30 + "\n")
        f.write(f"Found {len(matching_topics)} topic matches with at least 2 word overlap\n\n")
        f.write("Top 10 matching topics by word overlap:\n")
        f.write(matching_topics.nlargest(10, 'word_overlap')[
            ['global_topic', 'global_count', 'merged_topic', 'merged_count', 'word_overlap']
        ].to_string())
        f.write("\n\n")
        
        # Detailed Coverage Analysis
        f.write("4. Detailed Coverage Analysis:\n")
        f.write("-"*30 + "\n")
        for analysis in coverage['detailed_analysis']:
            f.write(f"\nGlobal Topic: {analysis['global_topic']} (Count: {analysis['global_count']})\n")
            f.write(f"Found in Merged: {analysis['found_in_merged']}\n")
            if analysis['matched_topics']:
                f.write("Matching topics in merged model:\n")
                for match in analysis['matched_topics']:
                    f.write(f"- {match['merged_topic']} (Count: {match['merged_count']}, "
                           f"Word Overlap: {match['word_overlap']})\n")

def main():
    # Load models
    merged_model_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/merged_models/2018_monthly/merged_model_2018_monthly_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-30_01-36-44"
    global_model_path = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2018_all_mpnet_base_v2_umap_hdbscan_nltk_stopwords_Nonetopics_10dim_2024-11-30_14-14-04_f66a63"
    
    merged_model = BERTopic.load(merged_model_path)
    global_model = BERTopic.load(global_model_path)
    
    # Run analyses
    coverage = compare_information_coverage(global_model, merged_model)
    freq_stats = analyze_topic_frequencies(global_model, merged_model)
    matching_topics = find_matching_topics(global_model, merged_model)
    
    # Save results to file
    output_file = "model_comparison_analysis.txt"
    save_analysis_to_file(output_file, coverage, freq_stats, matching_topics)
    print(f"Analysis results have been saved to {output_file}")

if __name__ == "__main__":
    main()