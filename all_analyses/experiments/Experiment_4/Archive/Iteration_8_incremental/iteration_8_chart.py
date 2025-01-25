import matplotlib.pyplot as plt
import numpy as np

# Data from your JSON
batch_memory = [36.46211242675781, 36.55507278442383, 36.73215103149414]
merge_memory = [26.902080535888672, 27.314125061035156, 27.389297485351562]

# Calculate averages
batch_avg = np.mean(batch_memory)
merge_avg = np.mean(merge_memory)

# Create the visualization
plt.figure(figsize=(12, 6))

# Create bar plot using the same colors as in the original code
bars = plt.bar([0, 1], 
              [batch_avg, merge_avg],
              color=['blue', 'red'],  # Original colors from the code
              alpha=0.6,
              width=0.6)

# Customize plot
plt.title('Peak Memory Usage Comparison', fontsize=14)
plt.ylabel('Memory Usage (GB)', fontsize=12)
plt.xticks([0, 1], ['Batch Processing', 'Incremental Processing'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f} GB',
            ha='center', va='bottom')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limit
plt.ylim(0, 40)  

# Save the plot
plt.tight_layout()
plt.savefig('memory_comparison.png', bbox_inches='tight', dpi=300)
plt.close()