import matplotlib.pyplot as plt
import numpy as np

def create_time_comparison_plot():
    # Get time data from the three runs
    batch_times = [2863.80011343956, 2534.064381122589, 2553.7528042793274]  # in seconds
    merge_times = [31.775514364242554, 32.17046093940735, 60.3819842338562]  # in seconds
    
    # Convert to minutes for better readability
    batch_times_min = [t/60 for t in batch_times]
    merge_times_min = [t/60 for t in merge_times]
    
    # Calculate averages
    avg_batch = np.mean(batch_times_min)
    avg_merge = np.mean(merge_times_min)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar([0, 1], 
                  [avg_batch, avg_merge],
                  color=['#1f77b4', '#d62728'],
                  alpha=0.7,
                  width=0.6)
    
    # Customize plot
    plt.title('Processing Time Comparison', fontsize=14, pad=20)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.xticks([0, 1], ['Batch Processing', 'Merge Operation'], fontsize=10)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} min',
                ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print exact values
    print("\nExact averages:")
    print(f"Batch Processing: {avg_batch:.2f} minutes ({avg_batch*60:.2f} seconds)")
    print(f"Merge Operation: {avg_merge:.2f} minutes ({avg_merge*60:.2f} seconds)")

if __name__ == "__main__":
    create_time_comparison_plot()