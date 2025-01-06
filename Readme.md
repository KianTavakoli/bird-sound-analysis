# 🌿 **Bird Sound Classification Pipeline** 🌿

This project implements a **Bird Sound Classifier** using a neural network to classify different bird species based on audio data. The pipeline involves loading audio data, training a deep learning model, and evaluating its performance.

---

## ✨ **Features**

- **Audio Data Augmentation**: The dataset can be augmented to improve model generalization.
- **Custom Dataset Class**: `AugmentedBirdSoundDataset` is used for loading and processing audio files.
- **Training and Evaluation**: The script supports model training, evaluation, and confusion matrix visualization.
- **Model Saving and Loading**: Trained models can be saved and reloaded for future use.

---

## 🛠️ **Requirements**

Make sure you have the following dependencies installed:

- `torch`
- `torchvision`
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`

You can install the required packages using:
```bash
pip install torch torchvision pandas matplotlib numpy scikit-learn
```

---

## 🚀 **Usage**

### 1️⃣ **Data Preparation**

- Place your bird sound `.wav` files in the `data/bird_sounds` directory.
- Prepare a CSV file named `lookup.csv` in the `data` directory. The CSV should have the following format:
  ```
  filename,species
  bird1.wav,sparrow
  bird2.wav,robin
  ...
  ```

### 2️⃣ **Running the Script**

1. Set the hyperparameters and paths:
   ```python
   data_dir = r'data\\bird_sounds'  # Path to your .wav files
   csv_file = r'data\\lookup.csv'   # Path to your CSV file
   num_epochs = 10                  # Number of training epochs
   batch_size = 16                  # Batch size
   learning_rate = 0.001            # Learning rate
   model_save_path = 'saved_model.pth'  # Path to save the trained model
   ```

2. Train the model:
   ```bash
   python bird_classifier.py
   ```

3. After training, the model will be saved to the specified path.

### 3️⃣ **Evaluation**

- The model is evaluated on a test set, and the test accuracy is printed.
- A confusion matrix is generated and saved as `confusion_matrix_bars.png`.
- Sample predictions are printed with filenames, true labels, and predicted species.

---

## 🔍 **Functions**

### 1. `train_model`
Trains the model on the training dataset.

### 2. `evaluate_model`
Evaluates the model on the test dataset and returns accuracy, labels, and filenames.

### 3. `save_model`
Saves the trained model to a specified path.

### 4. `load_model`
Loads a previously saved model from a specified path.

### 5. `plot_confusion_matrix_bars`
Generates and saves a confusion matrix as a bar chart.

---

## 📊 **Results**

After training for 10 epochs with the provided dataset, the model achieved a test accuracy of approximately **X%** (replace with your result).

---

## 💡 **Notes**

- Ensure that your audio files are properly formatted and labeled in the CSV.
- For better performance, consider using more data and experimenting with different hyperparameters.

---

## 📄 **License**

This project is licensed under the **MIT License**.

