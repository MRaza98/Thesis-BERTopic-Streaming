import json
import csv
import ast

def parse_topics_string(topics_str):
    # Convert string representation of list to actual list
    topics = ast.literal_eval(topics_str)
    return topics

def write_topics_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        data = csv.DictReader(f)
        rows = list(data)

    # Create output file with all topics from each period
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['period', 'topic_id', 'cv_score', 'size', 'top_words', 'word_weights'])
        
        # Process each period
        for row in rows:
            period = row['period']
            topics = parse_topics_string(row['topics_data'])
            
            # Write each topic for this period
            for topic in topics:
                writer.writerow([
                    period,
                    topic['topic_id'],
                    topic['cv_score'],
                    topic['size'],
                    topic['top_words'],
                    topic['word_weights']
                ])

# Use the function
input_file = '/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/analysis/incremental/out_of_sample_evaluation/2016_cross_year_coherence.csv'  # Your input file name
output_file = '2016_topics_by_period.csv'  # Your desired output file name
write_topics_to_csv(input_file, output_file)

print("Processing complete! Check topics_by_period.csv for the results.")