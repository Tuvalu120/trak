from torch.utils.data import Dataset
import os
from PIL import Image

class GTZANDataset(Dataset):
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
        path, label = self._samples[index]
        label = self._genre_to_idx[label]
        image = Image.open(path).convert("RGB")
        return image, label

