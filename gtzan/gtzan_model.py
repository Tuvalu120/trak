import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gtzan.gtzan_datasets import GTZANWaveDataset

class GTZANClassifier(nn.Module):
    def __init__(self):
        super(GTZANClassifier, self).__init__()

        self.features = nn.Sequential(
         nn.Conv2d(1, 64, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.Conv2d(64, 64, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.Conv2d(64, 16, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4 * 827, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device.type}")
    model = GTZANClassifier().to(device)
    crtierion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())


    X = GTZANWaveDataset("GTZAN/genres_original")
    X_train, X_test = torch.utils.data.random_split(X, [0.8, 0.2])
    X_train = torch.utils.data.Subset(X_train, range(100))
    X_test = torch.utils.data.Subset(X_test, range(25))

    train_dataloader = torch.utils.data.DataLoader(
        X_train,
        batch_size=10,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=(4 if device == "cuda" else 0),
    )

    targets_dataloader = torch.utils.data.DataLoader(
        X_test,
        batch_size=10,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=(4 if device == "cuda" else 0),
    )

    N_ITERATIONS = 2

    for i in range(N_ITERATIONS):
        j=1
        for spectogram, label in train_dataloader:

            spectogram = spectogram.to(device)
            label = label.to(device)

            outputs = model(spectogram)

            loss = crtierion(label, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch [{j}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            j += 1


        print(f"Epoch [{i+1}/{N_ITERATIONS}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "GTZANClassifier")
