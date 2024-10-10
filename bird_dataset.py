#dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import warnings

# Dataset class for loading audio files and optionally applying augmentation
class AugmentedBirdSoundDataset(Dataset):
    def __init__(self, data_dir, df, lookup, max_length=256, augment=False):
        self.data_dir = data_dir
        self.lookup = lookup  # Lookup table from CSV
        self.df = df  # Dataframe with file info
        self.files = os.listdir(data_dir)
        self.max_length = max_length
        self.augment = augment  # Whether to apply augmentation

        # Common transforms
        warnings.filterwarnings("ignore", category=UserWarning, message=".*mel filterbank has all zero values.*")
        self.stft = T.Spectrogram(n_fft=512, hop_length=256, power=None)  # Complex STFT output
        self.mel_scale = T.MelScale(n_mels=128, sample_rate=44100, n_stft=512 // 2 + 1)
        self.time_stretch = T.TimeStretch(n_freq=512 // 2 + 1)  # Ensure matching n_freq

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        # Extract species code from the filename, remove trailing dot, and take the first two characters
        species_code = self.files[idx].split("___")[-1].split('.')[0].strip()[:2].replace('.', '')
        species_name = self.lookup.get(species_code, 'unidentified')

        if species_name == 'unidentified':
            print(f"Species code '{species_code}' not found in lookup table.")
            return None  # Skip invalid entries

        label = list(self.lookup.values()).index(species_name)

        # Apply STFT (which gives a complex-valued tensor)
        spectrogram = self.stft(waveform)

        if self.augment:
            # Apply TimeStretch (requires complex input)
            spectrogram = self.time_stretch(spectrogram, 1.2)  # Apply time-stretch with a rate of 1.2

        # Convert to MelSpectrogram
        mel_spectrogram = self.mel_scale(spectrogram.abs())  # Use magnitude for Mel scaling

        # Clip or pad the Mel spectrogram
        mel_spectrogram = self.clip_or_pad_spectrogram(mel_spectrogram, self.max_length)

        # Extract the species name from the dataframe based on the file name
        true_species = self.df[self.df['species_code'].str[:2].str.replace('.', '') == species_code]['species'].values[0]

        return mel_spectrogram, torch.tensor(label), self.files[idx], true_species  # Include file name and true species

    def clip_or_pad_spectrogram(self, spectrogram, max_length):
        time_length = spectrogram.size(-1)
        if time_length > max_length:
            spectrogram = spectrogram[:, :, :max_length]
        elif time_length < max_length:
            padding_amount = max_length - time_length
            spectrogram = F.pad(spectrogram, (0, padding_amount))
        return spectrogram

# Custom collate function to skip empty batches
def collate_fn(batch):
    # Filter out all None data (invalid entries)
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
