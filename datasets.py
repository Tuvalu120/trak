import torch
import os
import torchaudio
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class GTZANImageDataset(Dataset):
    def __init__(self, root):
        self._samples = []

        genres_names = sorted([
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ])
        self._genre_to_idx = {genre: i for i, genre in enumerate(genres_names)}


        genres = os.listdir(root)
        for genre in genres:
            genre_path = os.path.join(root, genre)
            for file in os.listdir(genre_path):
                if file.endswith(".png"):
                    self._samples.append((os.path.join(genre_path, file), genre))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        path, label = self._samples[index]
        label = self._genre_to_idx[label]
        image = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float()
        return image, label


class GTZANWaveDataset(Dataset):
    def __init__(self, root):
        self._samples = []

        genres_names = sorted([
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ])
        self._genre_to_idx = {genre: i for i, genre in enumerate(genres_names)}


        genres = os.listdir(root)
        for genre in genres:
            genre_path = os.path.join(root, genre)
            for file in os.listdir(genre_path):
                if file.endswith(".wav"):
                    self._samples.append((os.path.join(genre_path, file), genre))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        TARGET_LEN = 3309
        path, label = self._samples[index]
        label = torch.tensor(self._genre_to_idx[label])
        label = torch.nn.functional.one_hot(label, 10).to(torch.float32)
        waveform, sample_rate = torchaudio.load(path)
        mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=512, n_mels=64, hop_length=256)(waveform)
        mel_spectogram = F.pad(mel_spectogram, (TARGET_LEN - mel_spectogram.size(2), 0))
        return mel_spectogram, label