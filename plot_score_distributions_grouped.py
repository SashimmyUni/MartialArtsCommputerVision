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

# Plot with group color and hatch for Amateur
plt.figure(figsize=(14, 7))


# Assign a unique color to each user
import itertools
user_list = df['user'].unique()
color_palette = sns.color_palette('tab10', n_colors=len(user_list))
user_palette = {user: color for user, color in zip(user_list, color_palette)}

ax = sns.boxplot(
    data=df,
    x='technique',
    y='score',
    hue='user',
    palette=user_palette
)


# Add diagonal hatching only to Dicte and Emil
for patch, user in zip(ax.artists, user_list):
    if 'Dicte' in user or 'Emil' in user:
        patch.set_hatch('//')  # Diagonal stripes
    else:
        patch.set_hatch('')   # No hatch
    patch.set_edgecolor('k')
    patch.set_linewidth(1.5)

plt.title('Score Distributions by Technique')
plt.ylabel('Score')
plt.xlabel('Technique')
plt.legend(title='User', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('score_by_technique_group_user.png')
plt.show()
print('Plot saved as score_by_technique_group_user.png')