import matplotlib.pyplot as plt
import numpy as np
import json

# Load and process data
with open('/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/thesis/experiments/Experiment_4/Iteration_8_incremental/run_3_20241230_202234.json', 'r') as f:
    data = json.load(f)

# Calculate averages and convert to minutes
batch_times = np.array(data['batch']['training_time']) / 60  # Convert seconds to minutes
merge_times = np.array(data['simultaneous_merge']['merge']['time']) / 60  # Convert seconds to minutes

batch_avg = np.mean(batch_times)
merge_avg = np.mean(merge_times)
incremental_avg = 4.330607167283693 + merge_avg # Given value in minutes

# Create the visualization
plt.figure(figsize=(12, 6))

# Create bar plot
bars = plt.bar([0, 1], 
              [batch_avg, incremental_avg],
              color=['blue', 'red'],
              alpha=0.6,
              width=0.6)

# Customize plot
plt.title('Training Time Comparison', fontsize=14)
plt.ylabel('Time (minutes)', fontsize=12)
plt.xticks([0, 1], ['Batch Processing', 'Incremental Processing'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f} min',
            ha='center', va='bottom')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limit dynamically
plt.ylim(0, max(batch_avg, incremental_avg) * 1.1)

# Save the plot
plt.tight_layout()
plt.savefig('training_time_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

# Print the values
print(f"Average Batch Training Time: {batch_avg:.2f} minutes")
print(f"Average Merge Time: {merge_avg:.2f} minutes")
print(f"Incremental Training Time: {incremental_avg:.2f} minutes")