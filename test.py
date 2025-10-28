import torch
from trak import TRAKer
from torchvision import transforms
from torchvision import models
from torchvision import datasets
import numpy as np

model = models.resnet18()
checkpoint = model.state_dict()
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

X_train = datasets.CIFAR100("CIFAR100", train=True, transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
X_train = torch.utils.data.Subset(X_train, list(range(128)))
X_test = datasets.CIFAR100("CIFAR100", train=False, transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
X_test = torch.utils.data.Subset(X_test, list(range(128)))


train_loader = torch.utils.data.DataLoader(
    X_train,
    batch_size=64,
    shuffle=True,
    pin_memory=(device == "cuda"),
    num_workers=(4 if device == "cuda" else 0))

targets_loader = torch.utils.data.DataLoader(
    X_test,
    batch_size=64,
    shuffle=True,
    pin_memory=(device == "cuda"),
    num_workers=(4 if device == "cuda" else 0))


traker = TRAKer(model=model, task='image_classification', train_set_size=len(X_train))

i=1
traker.load_checkpoint(checkpoint, model_id=0)
for X, y in train_loader:
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    traker.featurize(batch=(X, y), num_samples=X.shape[0])
    print(f"Batch [{i}/{len(train_loader)}] Completed")
    i+=1
traker.finalize_features()

traker.start_scoring_checkpoint('quickstart', checkpoint, model_id=0, num_targets=len(X_test))

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
plt.savefig("test.png")