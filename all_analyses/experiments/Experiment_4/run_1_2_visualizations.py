# For memory_usage_comparison.png
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the save directory
save_dir = "/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_5/Iteration_5_Corrected_Indent/visualizations"

def plot_memory_comparison():
    plt.figure(figsize=(15, 8))
    
    # Prepare data with new statistics
    batch_memory = [8.802236557006836, 9.81488037109375]  # Initial and Full batch
    total_incremental_memory = 0.25057220458984375  # Overall incremental mean
    
    # Create grouped bar plot
    positions = np.arange(3)
    plt.bar(positions[0:2], batch_memory, 
            alpha=0.8, label='Batch Processing',
            color='skyblue')
    plt.bar(positions[2], total_incremental_memory,
            alpha=0.8, label='Incremental Processing',
            color='lightgreen')
    
    # Customize plot
    plt.title('Memory Usage Comparison: Batch vs Incremental', fontsize=14)
    plt.ylabel('Memory Usage (GB)', fontsize=12)
    plt.xticks(positions, ['Batch\n(2000-2018)', 'Batch\n(2000-2019)', 'Incremental\n(2019)'])
    plt.legend()
    plt.grid(False)
    
    # Add value labels
    for i, v in enumerate(batch_memory + [total_incremental_memory]):
        plt.text(positions[i], v, f'{v:.2f} GB',
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(save_dir, 'memory_usage_comparison_run_1_2.png'), bbox_inches='tight', dpi=300)
    plt.close()

# For batch_vs_incremental_comparison.png
def plot_batch_vs_incremental():
    plt.figure(figsize=(15, 8))

    # Prepare data with new statistics
    batch_time = [6190.096589803696, 7293.712253689766]  # Initial and Full batch
    incremental_time = 148.71883916854858  # Incremental mean

    batch_time_mins = [i/60 for i in batch_time]
    incremental_time_mins = incremental_time/60
    
    # Create grouped bar plot
    positions = np.arange(3)
    plt.bar(positions[0:2], batch_time_mins, 
            alpha=0.8, label='Batch Processing',
            color='skyblue')
    plt.bar(positions[2], incremental_time_mins,
            alpha=0.8, label='Incremental Processing',
            color='lightgreen')

    for i, v in enumerate(batch_time_mins + [incremental_time_mins]):
        plt.text(positions[i], v, f'{v:.2f} min.',
                ha='center', va='bottom')
    
    # Customize plot
    plt.title('Processing Time Comparison: Batch vs Incremental', fontsize=14)
    plt.xticks(positions, ['Batch\n(2000-2018)', 'Batch\n(2000-2019)', 'Incremental\n(2019)'])
    plt.ylabel('Processing Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(False)
    
    plt.savefig(os.path.join(save_dir, 'batch_vs_incremental_comparison_run_1_2.png'), bbox_inches='tight', dpi=300)
    plt.close()

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Run both plotting functions
plot_memory_comparison()
plot_batch_vs_incremental()