#train_eval.py
import torch

# Training function with accuracy tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for batch in train_loader:
            if batch is None:
                continue  # Skip empty batches

            mel_spectrograms, labels, _, _ = batch  # Discard file name and species
            mel_spectrograms, labels = mel_spectrograms.to(device), labels.to(device)

            # Forward pass
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {train_accuracy:.2f}%')

        # Optionally, you can evaluate on a validation set
        if val_loader:
            val_accuracy = evaluate_model(model, val_loader, device)
            print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Evaluation function to check performance on test/validation data
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_filenames = []
    all_true_species = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch in test_loader:
            if batch is None:
                continue  # Skip empty batches

            mel_spectrograms, labels, filenames, true_species = batch
            mel_spectrograms, labels = mel_spectrograms.to(device), labels.to(device)
            outputs = model(mel_spectrograms)
            _, predicted = torch.max(outputs, 1)

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)  # Save filenames for reference
            all_true_species.extend(true_species)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds, all_filenames, all_true_species

# Function to print training and testing sample counts
def print_sample_counts(train_dataset, test_dataset):
    print(f"Number of files used for training: {len(train_dataset)}")
    print(f"Number of files used for testing: {len(test_dataset)}")
