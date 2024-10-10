import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Function to read lookup table from the CSV file
def read_csv_file(csv_file):
    df = pd.read_csv(csv_file)
    lookup_dict = {}
    for _, row in df.iterrows():
        species_code = row['species_code'].strip().replace('.', '')  # Remove any trailing dots
        lookup_dict[species_code] = row['species']
    return lookup_dict, df  # Return both the lookup and the full dataframe

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load a saved model
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    model.to(device)

# Function to plot bar chart for confusion matrix counts
def plot_confusion_matrix_bars(labels, preds, lookup, output_file='confusion_matrix_bars.png'):
    cm = confusion_matrix(labels, preds)
    species_names = list(lookup.values())

    # Ensure that class_counts size matches the number of species in the lookup table
    class_counts = np.zeros(len(species_names))

    # Accumulate counts for the actual classes present in the labels
    for label in labels:
        class_counts[label] += 1

    present_classes = np.unique(labels)

    filtered_species_names = [species_names[i] for i in present_classes]
    filtered_class_counts = class_counts[present_classes]

    plt.figure(figsize=(12, 6))
    plt.bar(filtered_species_names, filtered_class_counts, color='blue')
    plt.xticks(rotation=90)
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Bar Plot (Actual Counts)')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
