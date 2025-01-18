import os
import pandas as pd
import matplotlib.pyplot as plt

# Specify the root directory where your CSV files are stored
root_dir = "results/recruiting-ocel1"
prediction_task = "remaining_time"

# Initialize a dataframe to collect all results
all_results = []

# Walk through the directory and read CSV files
for subdir, _, files in os.walk(os.path.join(root_dir, prediction_task)):
    for file in files:
        if file.endswith(".csv"):
            parts = file.split('_')
            graph_layer, embedding_size, node_features = parts[0], parts[1], parts[2].replace('.csv', '')
            subgraph_size = int(os.path.basename(subdir))

            # Read the CSV file as text to split on repeating headers
            with open(os.path.join(subdir, file), 'r') as f:
                content = f.read().strip().split(',prediction_layer,score\n')

            # Parse each block of results
            for block in content[1:]:  # Skip the initial empty split before the first header
                data = pd.read_csv(pd.compat.StringIO(f",prediction_layer,score\n{block}"))
                for _, row in data.iterrows():
                    all_results.append({
                        'subgraph_size': subgraph_size,
                        'graph_layer': graph_layer,
                        'embedding_size': int(embedding_size),
                        'node_features': node_features,
                        'prediction_layer': row['prediction_layer'],
                        'score': row['score']
                    })

# Convert to DataFrame for easier plotting
results_df = pd.DataFrame(all_results)

# Plotting results
plt.figure(figsize=(12, 8))
for prediction_layer in results_df['prediction_layer'].unique():
    subset = results_df[results_df['prediction_layer'] == prediction_layer]
    plt.plot(subset['subgraph_size'], subset['score'], label=prediction_layer)

plt.xlabel("Subgraph Size (k)")
plt.ylabel("Score")
plt.title(f"Score vs Subgraph Size for {prediction_task}")
plt.legend()
plt.grid(True)
plt.show()