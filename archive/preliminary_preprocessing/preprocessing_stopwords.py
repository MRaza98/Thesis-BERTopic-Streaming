import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

def initialize_nltk():
    """Download required NLTK data."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def count_words(text):
    """
    Count the number of words in the given text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        int: Number of words
    """
    if not isinstance(text, str):
        return 0
    
    # Split on whitespace and count non-empty words
    words = [word for word in text.split() if word.strip()]
    return len(words)

def remove_stopwords(text):
    """
    Remove stopwords and punctuation from the given text.
    Returns tuple of (processed_text, num_stopwords_removed)
    
    Args:
        text (str): Input text to process
        
    Returns:
        tuple: (processed_text, number_of_stopwords_removed)
    """
    if not isinstance(text, str):
        return "", 0
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Split into words
    words = text.split()
    
    # Count stopwords removed
    num_stopwords = sum(1 for word in words if word in stop_words)
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join words back together
    return " ".join(filtered_words), num_stopwords

def process_file(input_file, output_file):
    """
    Process the input CSV file and write results to output file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    try:
        # Initialize NLTK
        initialize_nltk()
        
        # Read the input file
        df = pd.read_csv(input_file)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            raise ValueError("Input file must contain a 'text' column")
        
        initial_rows = len(df)
        
        # First remove stopwords and track counts
        processed_results = [remove_stopwords(text) for text in df['text']]
        df['processed_text'] = [result[0] for result in processed_results]
        stopwords_removed = [result[1] for result in processed_results]
        
        # Calculate stopword statistics
        total_stopwords_removed = sum(stopwords_removed)
        avg_stopwords_per_doc = total_stopwords_removed / len(df)
        
        # Count words in processed text and filter
        df['word_count'] = df['processed_text'].apply(count_words)
        df = df[df['word_count'] >= 10]
        rows_removed = initial_rows - len(df)
        
        # Remove the temporary word_count column
        df = df.drop('word_count', axis=1)
        
        # Write to output file
        df.to_csv(output_file, index=False)
        
        # Print detailed statistics
        print("\nProcessing Statistics:")
        print(f"Total documents processed: {initial_rows}")
        print(f"Total stopwords removed: {total_stopwords_removed:,}")
        print(f"Average stopwords per document: {avg_stopwords_per_doc:.2f}")
        print(f"Rows removed (< 10 words after processing): {rows_removed}")
        print(f"Final document count: {len(df)}")
        print(f"\nOutput written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found")
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    input_file = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/preprocessed_england_speeches_2018.csv"
    output_file = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/data/stopwords_removed_england_speeches_sample_2018.csv"
    
    process_file(input_file, output_file)