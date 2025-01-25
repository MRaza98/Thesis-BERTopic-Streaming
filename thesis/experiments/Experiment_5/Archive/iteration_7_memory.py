import matplotlib.pyplot as plt
import numpy as np

def create_memory_comparison_plot():
    # Memory usage data in GB
    operations = ['Batch Processing', 'Merge Operation']
    memory_usage = [39.36, 35.41]  # Average values calculated from the data
    
    # Create figure and axis with specific size
    plt.figure(figsize=(12, 8))
    
    # Create bars with specific colors matching the previous plot
    bars = plt.bar([0, 1], 
                  memory_usage,
                  color=['#1f77b4', '#d62728'],  # Blue and red colors from previous plot
                  alpha=0.8,
                  width=0.6)
    
    # Customize the plot
    plt.title('Peak Memory Usage Comparison', fontsize=14, pad=20)
    plt.ylabel('Memory Usage (GB)', fontsize=12)
    plt.xticks([0, 1], operations, fontsize=10)
    
    # Set y-axis to start at 0 and end at 40 with grid at 5GB intervals
    plt.ylim(0, 40)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB',
                ha='center', va='bottom',
                fontsize=10)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with high resolution
    plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print the exact values
    print("\nPeak Memory Usage:")
    print(f"Batch Processing: {memory_usage[0]:.2f} GB")
    print(f"Merge Operation: {memory_usage[1]:.2f} GB")

if __name__ == "__main__":
    create_memory_comparison_plot()