import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.brain import ChessMimicNet


def train_model():
    if not os.path.exists("data/dataset.npz"):
        print("Error: data/dataset.npz not found! Run preprocess.py first.")
        return

    data = np.load("data/dataset.npz")
    X = torch.tensor(data["x"], dtype=torch.float32)
    Y = torch.tensor(data["y"], dtype=torch.long)

    dataset = TensorDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training engine running on: {device}")

    model = ChessMimicNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    print("Starting model training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = train_loss / train_size
        accuracy = (correct / total) * 100
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Move Match Accuracy: {accuracy:.2f}%"
        )

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/mimic_v1.pth")
    print("Training complete! Weights saved to model/mimic_v1.pth")


if __name__ == "__main__":
    train_model()
