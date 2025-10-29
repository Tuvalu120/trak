import torch
import numpy as np
from trak import TRAKer
from gtzan.gtzan_datasets import GTZANWaveDataset
from gtzan.gtzan_model import GTZANClassifier

model = GTZANClassifier()
model.load_state_dict(torch.load("GTZANClassifier"))
checkpoint = model.state_dict()
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


X = GTZANWaveDataset("GTZAN/genres_original")
X_train, X_test = torch.utils.data.random_split(X, [0.8, 0.2])
X_train = torch.utils.data.Subset(X_train, range(100))
X_test = torch.utils.data.Subset(X_test, range(25))

train_loader = torch.utils.data.DataLoader(
    X_train,
    batch_size=1,
    shuffle=True
    )

targets_loader = torch.utils.data.DataLoader(
    X_test,
    batch_size=1,
    shuffle=True
    )


traker = TRAKer(model=model, task='image_classification', train_set_size=len(X_train))

i=1
traker.load_checkpoint(checkpoint, model_id=1)
for X, y in train_loader:
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).to(torch.long)
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