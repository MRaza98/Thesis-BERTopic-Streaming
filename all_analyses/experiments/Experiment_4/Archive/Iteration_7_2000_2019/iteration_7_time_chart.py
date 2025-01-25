import matplotlib.pyplot as plt
import numpy as np
import json

def create_time_comparison_plot():
    # Load and process the timing data
    data = {
        "batch": {
            "training_time": [
                9947.459701776505,
                8374.72041964531,
                8306.029932975769
            ]
        },
        "simultaneous_merge": {
            "merge": {
                "time": [
                    65.97392106056213,
                    68.56961226463318,
                    67.56850290298462
                ]
            }
        }
    }
    
    # Extract timing data
    batch_times = data["batch"]["training_time"]  # in seconds
    merge_times = data["simultaneous_merge"]["merge"]["time"]  # in seconds
    
    # Convert to minutes for better readability
    batch_times_min = [t/60 for t in batch_times]
    merge_times_min = [t/60 for t in merge_times]
    
    # Calculate averages
    avg_batch = np.mean(batch_times_min)
    avg_merge = np.mean(merge_times_min)
    
    # Create bar plot with improved styling
    plt.figure(figsize=(12, 6))
    bars = plt.bar([0, 1], 
                  [avg_batch, avg_merge],
                  color=['#1f77b4', '#d62728'],
                  alpha=0.8,
                  width=0.6)
    
    # Customize plot appearance
    plt.title('Training and Merge Time Comparison', fontsize=14, pad=20)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.xticks([0, 1], ['Batch Training', 'Merge Operation'], fontsize=10)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} min',
                ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add error bars showing the range of values
    batch_std = np.std(batch_times_min)
    merge_std = np.std(merge_times_min)
    plt.errorbar([0, 1], [avg_batch, avg_merge],
                yerr=[batch_std, merge_std],
                fmt='none', color='black', capsize=5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with high resolution
    plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("\nBatch Training:")
    print(f"Average: {avg_batch:.2f} minutes ({avg_batch*60:.2f} seconds)")
    print(f"Standard Deviation: ±{batch_std:.2f} minutes")
    print(f"Range: {min(batch_times_min):.2f} - {max(batch_times_min):.2f} minutes")
    
    print("\nMerge Operation:")
    print(f"Average: {avg_merge:.2f} minutes ({avg_merge*60:.2f} seconds)")
    print(f"Standard Deviation: ±{merge_std:.2f} minutes")
    print(f"Range: {min(merge_times_min):.2f} - {max(merge_times_min):.2f} minutes")

if __name__ == "__main__":
    create_time_comparison_plot()