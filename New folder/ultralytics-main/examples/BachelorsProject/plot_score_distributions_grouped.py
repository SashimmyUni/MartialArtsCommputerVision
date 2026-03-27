import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to runs
groups = {
    'Amateur': os.path.join('data', 'runs', 'Amateur'),
    'Professional': os.path.join('data', 'runs', 'Proffesional'),
}

all_dfs = []
for group, group_path in groups.items():
    for user_folder in os.listdir(group_path):
        user_path = os.path.join(group_path, user_folder)
        if not os.path.isdir(user_path):
            continue
        # Extract user name (first part before '_') and combine with group
        user_name = user_folder.split('_')[0]
        user_label = f"({group}) {user_name}"
        for root, dirs, files in os.walk(user_path):
            for file in files:
                if file == 'metrics.csv':
                    df = pd.read_csv(os.path.join(root, file))
                    df['group'] = group
                    df['user'] = user_label
                    all_dfs.append(df)

if not all_dfs:
    raise RuntimeError('No metrics.csv files found!')

df = pd.concat(all_dfs, ignore_index=True)

# Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='technique', y='score', hue='user')
plt.title('Score Distributions by Technique (Each User, Grouped)')
plt.ylabel('Score')
plt.xlabel('Technique')
plt.legend(title='User')
plt.tight_layout()
plt.savefig('score_by_technique_by_user.png')
plt.show()
print('Plot saved as score_by_technique_by_user.png')
