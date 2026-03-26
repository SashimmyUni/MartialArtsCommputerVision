import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the runs directory
RUNS_DIR = os.path.join(
    os.path.dirname(__file__),
    'data', 'runs'
)

# Find all metrics.csv files
metrics_files = glob.glob(os.path.join(RUNS_DIR, 'run_*', 'metrics.csv'))

# Read and concatenate all metrics
dfs = []
for file in metrics_files:
    df = pd.read_csv(file)
    df['run'] = os.path.basename(os.path.dirname(file))
    dfs.append(df)

if not dfs:
    print("No metrics.csv files found.")
    exit(1)

data = pd.concat(dfs, ignore_index=True)

# Plot score distributions for each technique
plt.figure(figsize=(10, 6))
sns.boxplot(x='technique', y='score', data=data, showfliers=False)
sns.stripplot(x='technique', y='score', data=data, color='black', alpha=0.3, jitter=True)
plt.axhline(y=data['score_threshold'].iloc[0], color='red', linestyle='--', label='Threshold')
plt.title('Score Distributions by Technique')
plt.ylabel('Score')
plt.xlabel('Technique')
plt.legend()
plt.tight_layout()
plt.savefig('score_by_technique.png')
plt.show()
