import pandas as pd
import librosa
import numpy as np
import librosa.display
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from typing import Tuple, List, Dict, Optional

def spec_normalization(spec: np.ndarray, err: float = 1e-6) -> np.ndarray:
    """
    Normalize spectrogram by standardization.

    Args:
        spec: Input spectrogram
        err: Small constant to avoid division by zero

    Returns:
        Normalized spectrogram
    """
    mean, std = spec.mean(), spec.std()
    return (spec - mean) / (std + err)

def spec_to_image(spec: np.ndarray) -> np.ndarray:
    """
    Convert spectrogram to image format with values in [0, 255].

    Args:
        spec: Input spectrogram

    Returns:
        Spectrogram as uint8 image with channel dimension
    """
    spec = spec_normalization(spec)
    spec_min, spec_max = spec.min(), spec.max()
    spec = 255 * (spec - spec_min) / (spec_max - spec_min)
    spec = spec.astype(np.uint8)
    return spec[np.newaxis, ...]

def apply_specaug(mel_spectrogram: np.ndarray,
                  frequency_masking_para: int = 10,
                  time_masking_para: int = 10,
                  frequency_mask_num: int = 1,
                  time_mask_num: int = 1) -> np.ndarray:
    """
    Apply SpecAugment data augmentation to mel spectrogram.

    Args:
        mel_spectrogram: Input mel spectrogram
        frequency_masking_para: Maximum frequency mask size
        time_masking_para: Maximum time mask size
        frequency_mask_num: Number of frequency masks
        time_mask_num: Number of time masks

    Returns:
        Augmented spectrogram
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Frequency masking
    for _ in range(frequency_mask_num):
        f = int(np.random.uniform(0, frequency_masking_para))
        f0 = random.randint(0, v - f)
        mel_spectrogram[:, f0:f0 + f, :] = 0

    # Time masking
    for _ in range(time_mask_num):
        t = int(np.random.uniform(0, time_masking_para))
        t0 = random.randint(0, tau - t)
        mel_spectrogram[:, :, t0:t0 + t] = 0

    return mel_spectrogram

def get_melspec(file_path: str,
                sr: int = 8000,
                top_db: int = 80,
                dataset: Optional[str] = None) -> np.ndarray:
    """
    Convert audio file to mel spectrogram.

    Args:
        file_path: Path to audio file
        sr: Sample rate
        top_db: Maximum DB difference
        dataset: Dataset type for specific processing

    Returns:
        Mel spectrogram
    """
    wav, sr = librosa.load(file_path.replace('/', '\\'), sr=sr)

    if dataset == 'wingbeats':
        # Pad the short wingbeats samples
        target_length = 2 * sr
        pad_length = int(np.ceil((target_length - wav.shape[0]) / 2))
        wav = np.pad(wav, pad_length, mode='reflect')

    # Calculate mel spectrogram
    spec = librosa.feature.melspectrogram(
        y = wav,
        sr=sr,
        n_fft=256,
        hop_length=64
    )

    # Convert to DB scale
    spec = librosa.power_to_db(spec, top_db=top_db)
    return spec

class AudioDataset(Dataset):
    """Dataset class for audio spectrograms with optional augmentation."""

    def __init__(self,
                 label_file: str,
                 label_map: Dict[str, int],
                 mode: Optional[str] = None,
                 dataset_type: str = "wingbeats"):
        """
        Initialize dataset.

        Args:
            label_file: Path to CSV file with labels
            label_map: Mapping from class names to indices
            mode: 'train' or 'val' for applying augmentation
            dataset_type: Type of dataset for specific processing
        """
        self.data = pd.read_csv(label_file)
        a = len(self.data)
        self.label_map = label_map
        self.mode = mode
        self.dataset_type = dataset_type

        # Pre-compute spectrograms and labels
        self.specs = []
        self.labels = []
        # self._precompute_data()


    def __getitem__(self, index: int) -> Tuple[np.ndarray, torch.Tensor]:
        """Get spectrogram and label for given index."""
        row = self.data.iloc[index]
        audio_path = row['Fname']
        is_wingbeats = self.dataset_type == "Wingbeats"
        spec = get_melspec(
            audio_path,
            dataset='wingbeats' if is_wingbeats else None
        )
        spec = spec_to_image(spec)

        # Get label
        label_class = row['Species']
        label = torch.tensor(self.label_map[label_class])
        # spec = self.specs[index].copy()

        # Apply augmentation for training
        if self.mode == "train":
            spec = apply_specaug(spec)
        
        return spec, label

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

