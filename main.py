import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from bird_utils import read_csv_file, save_model, load_model, plot_confusion_matrix_bars
from bird_dataset import AugmentedBirdSoundDataset, collate_fn
from bird_model import BirdSoundClassifier
from bird_trainer import train_model, evaluate_model, print_sample_counts

# Hyperparameters and paths
data_dir = r'data\bird_sounds'  # Path to your .wav files
csv_file = r'data\lookup.csv'  # Path to your CSV file
num_epochs = 10
batch_size = 16
learning_rate = 0.001
model_save_path = 'saved_model.pth'  # Path where the model will be saved

# Load lookup table and DataFrame from CSV
lookup, df = read_csv_file(csv_file)

# Split data into training and testing sets
train_size = int(0.8 * len(df))  # 80% of data for training
test_size = len(df) - train_size  # 20% for testing

# Dataset and DataLoader setup with split
full_dataset = AugmentedBirdSoundDataset(data_dir, df, lookup, max_length=256, augment=False)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Print sample counts
print_sample_counts(train_dataset, test_dataset)

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, loss function, and optimizer
model = BirdSoundClassifier(num_classes=len(lookup)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, None, criterion, optimizer, num_epochs, device)

# Save the trained model
save_model(model, model_save_path)

# Load the saved model (for testing or future use)
load_model(model, model_save_path, device)

# Evaluate on test set and get the test accuracy, filenames, and true species
test_accuracy, all_labels, all_preds, all_filenames, all_true_species = evaluate_model(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Print filenames with their true and predicted labels
for filename, true_label, pred_label in zip(all_filenames, all_true_species, all_preds):
    pred_species = list(lookup.values())[pred_label]
    print(f"File: {filename} | True Label: {true_label} | Predicted: {pred_species}")

# Save and plot the confusion matrix as a bar chart
plot_confusion_matrix_bars(all_labels, all_preds, lookup, output_file='confusion_matrix_bars.png')
