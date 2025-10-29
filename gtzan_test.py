from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision import models
import numpy as np
from trak import TRAKer

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
                if file.endswith(".png"):
                    self._samples.append((os.path.join(genre_path, file), genre))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        path, label = self._samples[index]
        label = self._genre_to_idx[label]
        image = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float()
        return image, label


model = models.vgg11()
checkpoint = model.state_dict()
device = "cuda" if torch.cuda.is_available() else "cpu"


model.to(device)


X = GTZANDataset("GTZAN/images_original")
X_train, X_test = torch.utils.data.random_split(X, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(
    X_train,
    batch_size=5,
    shuffle=True,
    pin_memory=(device == "cuda"),
    num_workers=(4 if device == "cuda" else 0)
    )

targets_loader = torch.utils.data.DataLoader(
    X_test,
    batch_size=5,
    shuffle=True,
    pin_memory=(device == "cuda"),
    num_workers=(4 if device == "cuda" else 0)
    )


traker = TRAKer(model=model, task='image_classification', train_set_size=len(X_train))

i=1
traker.load_checkpoint(checkpoint, model_id=1)
for X, y in train_loader:
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    traker.featurize(batch=(X, y), num_samples=X.shape[0])
    print(f"Batch [{i}/{len(train_loader)}] Completed")
    i+=1
traker.finalize_features()

traker.start_scoring_checkpoint('quickstart', checkpoint, model_id=1, num_targets=len(X_test))

i=1
for X, y in targets_loader:
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    traker.score(batch=(X, y), num_samples=X.shape[0])
    print(f"Batch [{i}/{len(targets_loader)}] Completed")
    i+=1
scores = traker.finalize_scores(exp_name='quickstart')

np.savetxt("test.csv", scores, delimiter=",")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
plot = plt.matshow(scores)
plt.colorbar(plot)
plt.savefig("test_gtzan.png")