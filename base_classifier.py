import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

num_agents = 64
cost_alpha = 0.05
externality_alpha = 0.05
lower_bound, upper_bound = -2, 2

def make_dataset():
    # Create a synthetic dataset
    X, y = make_classification(
        n_samples=num_agents*20, 
        n_features=5, 
        n_informative=5, 
        n_redundant=0, 
        n_classes=2, 
        random_state=42,
        weights=[0.5, 0.5],
        class_sep=2.0
    )

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def make_dataloader(X_train_tensor, y_train_tensor, batch_size):
    # Create DataLoader for batching
    # Batching here is really the number of "agents" participating in the game
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader 


# Define a simple linear classifier using a weight vector
class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        self.weights = nn.Parameter(torch.normal(torch.zeros(input_dim)).float())
        self.bias = nn.Parameter(torch.normal(torch.zeros(1)).float())

    def forward(self, x):
        # Get the outcome on the NE features
        logits = torch.matmul(x, self.weights) + self.bias # Matrix multiplication
        return logits


def calculate_accuracy(outputs, labels):
    # Convert logits to predictions
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    #predictions = (outputs > 0.5).float()
    return (predictions == labels).float().mean().item()


def train_model(X_train, y_train, X_test, y_test, num_epochs=100):
    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = LinearClassifier(input_dim)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loader = make_dataloader(X_train, y_train, batch_size=num_agents)
    test_loader = make_dataloader(X_test, y_test, batch_size=num_agents)

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracy = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation phase
        if (epoch+1) % 1 == 0:
            train_losses.append(epoch_train_loss / len(train_loader))
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                for test_feat, test_label in test_loader: 
                    test_outputs = model(test_feat)
                    loss = criterion(test_outputs, test_label)
                    val_loss += loss.item()
                    val_acc += calculate_accuracy(test_outputs, test_label)
                val_losses.append(val_loss / len(test_loader))
                val_accuracy.append(val_acc / len(test_loader))

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Acc: {val_accuracy[-1]:.4f}')
        
    # Plotting the train and validation error
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_dataset()
    train_model(X_train, y_train, X_test, y_test)