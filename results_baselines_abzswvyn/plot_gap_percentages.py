import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('maple_comparison.csv')

# Set style
plt.style.use('seaborn')
plt.figure(figsize=(15, 8))

# Create x-axis positions
x = range(len(df))

# Plot static gap percentage
plt.plot(x, df['Static_Gap_%'], 'b-', label='Static', linewidth=2, marker='o')

# Plot dynamic gap percentage with error bars
plt.errorbar(x, df['Dynamic_Mean_Gap_%'], 
            yerr=[df['Dynamic_Mean_Gap_%'] - df['Dynamic_Min'], 
                  df['Dynamic_Max'] - df['Dynamic_Mean_Gap_%']],
            fmt='r-', label='Dynamic', linewidth=2, marker='s',
            capsize=5, capthick=2)

# Customize the plot
plt.title('Gap Percentage Comparison: Static vs Dynamic Models', fontsize=14, pad=20)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Gap Percentage (%)', fontsize=12)
plt.xticks(x, df['Dataset'], rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Add dataset size annotations
for i, size in enumerate(df['Size']):
    plt.annotate(size, (i, df['Static_Gap_%'].max() + 0.5),
                ha='center', va='bottom', rotation=45, fontsize=8)

# Adjust layout and save
plt.tight_layout()
plt.savefig('gap_percentage_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("\nStatic Model:")
print(f"Average Gap: {df['Static_Gap_%'].mean():.2f}%")
print(f"Min Gap: {df['Static_Gap_%'].min():.2f}%")
print(f"Max Gap: {df['Static_Gap_%'].max():.2f}%")

print("\nDynamic Model:")
print(f"Average Gap: {df['Dynamic_Mean_Gap_%'].mean():.2f}%")
print(f"Min Gap: {df['Dynamic_Min'].min():.2f}%")
print(f"Max Gap: {df['Dynamic_Max'].max():.2f}%") 