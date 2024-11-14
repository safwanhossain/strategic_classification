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

batch_size = 8
num_agents = 8
cost_alpha = 0.05
externality_alpha = 0.05
lower_bound, upper_bound = -2, 2

def make_dataset():
    # Create a synthetic dataset
    X, y = make_classification(
        n_samples=num_agents*120, 
        n_features=5, 
        n_informative=3, 
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
    def __init__(self, feature_dim, batch_dim, strat):
        super(LinearClassifier, self).__init__()
        self.weights = nn.Parameter(torch.normal(torch.zeros(feature_dim)).float())  # Two classes for binary classification
        self.strat = strat
        if self.strat:
            self.train_cvx_layer = create_cvx_layer((batch_dim, feature_dim), self.weights.shape)
    
    def forward(self, x):
        if self.strat:
            chunks = torch.chunk(x, batch_size//num_agents, dim=0)
            all_features = []
            for chunk in chunks:
                NE_features, = self.train_cvx_layer(x, self.weights)
                all_features.append(NE_features)
            equi_features = torch.cat(all_features, dim=0)
            
            # Get the outcome on the NE features
            logits = torch.matmul(equi_features, self.weights)  # Matrix multiplication

            # Get the probabilities
            probs = torch.sigmoid(logits)
            return probs 
        else:
            # Get the outcome on the NE features
            logits = torch.matmul(x, self.weights)  # Matrix multiplication

            # Get the probabilities
            probs = torch.sigmoid(logits)
            return probs


def train_model(X_train, y_train, X_test, y_test, strat, num_epochs=50):
    # Initialize the model, loss function, and optimizer
    feature_dim = X_train.shape[1]
    model = LinearClassifier(feature_dim, batch_size, strat)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=0.01)
    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size)
    test_loader = make_dataloader(X_test, y_test, batch_size=num_agents)
    
    x_test, y_test = next(iter(test_loader))
    validation_cvx_layer = create_cvx_layer(x_test.shape, model.weights.shape)

    # Training loop
    train_losses = []
    val_losses = []
    strat_val_losses = []

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
        if (epoch+1) % 2 == 0:
            train_losses.append(epoch_train_loss / len(train_loader))
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for test_feat, test_label in test_loader: 
                    outputs = model(test_feat)
                    loss = criterion(outputs, test_label)
                    val_loss += loss.item()
                val_losses.append(val_loss / len(test_loader))

                # Get the strategic validation results
                curr_strat_val_losses = []
            
                for test_feat, test_label in test_loader:
                    equi_features, = validation_cvx_layer(test_feat.detach(), model.weights.detach())
                    equi_features = equi_features.detach()
                    # verify_equi(equi_features.numpy(), test_feat.numpy(), model.weights.detach().numpy())

                    strat_val_outputs = model(equi_features)
                    curr_strat_val_loss = criterion(strat_val_outputs, test_label)
                    curr_strat_val_losses.append(curr_strat_val_loss.item()) 
                strat_loss = sum(curr_strat_val_losses)/len(curr_strat_val_losses)
                strat_val_losses.append(strat_loss)

            # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Strat Validation Loss: {strat_loss}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Strat Validation Loss: {strat_loss}')
    return train_losses, val_losses, strat_val_losses


def plot_all(train, train_strat, val, val_strat):
    # Plotting the train and validation error
    plt.plot(train_strat, label='Strategic Train Loss')
    plt.plot(val, label='Val Loss (Non-Strat Training)')
    plt.plot(val_strat, label='Val Loss (Strat Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(val, label='Val Loss (Non-Strat Training)')
    plt.plot(val_strat, label='Val Loss (Strat Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.show()

def plot_non_strat(train, val, val_strat):
    # Plotting the train and validation error
    plt.plot(train, label='Non-Strategic Train Loss')
    plt.plot(val, label='Val Loss (Non-Strat Validation)')
    plt.plot(val_strat, label='Val Loss (Strategic Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Under Non Strategic Training')
    plt.legend()
    plt.show()
    plt.close()


def verify_equi(equi_X, true_X, weight):
    true_features = cp.Parameter(true_X.shape, name="true_feat", value=true_X)
    equi_features = cp.Parameter(equi_X.shape, name="equi_feat", value=equi_X)
    weight_param = cp.Parameter(weight.shape, name="weight", value=weight)
    
    # check the best response for each agent
    for i, (equi_feat, true_feat) in enumerate(zip(equi_features, true_features)):
        # Parameter setup remains the same 
        opt_feat = cp.Variable(equi_feat.shape, name="opt_feat") 
        
        gain = opt_feat @ weight_param
        cost = cp.norm2(opt_feat - true_feat)
        externality = 0

        for j, (j_equi_feat, j_true_feat) in enumerate(zip(equi_features, true_features)):
            if j == i:
                continue
            ext_ij = cp.square(cp.norm2(j_equi_feat - j_true_feat) + cp.norm2(opt_feat - true_feat))
            externality += ext_ij
        
        utility = gain - cost_alpha*cost - externality_alpha*externality
        constraints = [opt_feat >= lower_bound, opt_feat <= upper_bound]
        objective = cp.Maximize(utility)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        diff = np.sum(np.abs(opt_feat.value - equi_feat.value))
        assert diff <= 0.001
        print(f"Opt_feat: {opt_feat.value}, equi feat: {equi_feat.value}, true_feat: {true_feat.value}")


def create_cvx_layer(feature_shape, weight_shape):
    # Parameter setup remains the same
    n = feature_shape[0]
    true_features = cp.Parameter(feature_shape, name="true_feat")
    weight_param = cp.Parameter(weight_shape, name="weight")
    opt_features = cp.Variable(feature_shape, name="feature_mat")

    # the gain from classification
    gain_vec = opt_features @ weight_param
    gain = cp.sum(gain_vec)

    # the cost of manipulation to all agents
    cost = cp.sum(cp.norm2(true_features - opt_features, axis=1))

    # the externality for all agents
    diffs = cp.norm2(true_features - opt_features, axis=1)
    ext = 0
    for i in range(n):
        for j in range(i+1, n):
            ext += cp.square(diffs[i] + diffs[j])

    utility = gain - cost_alpha*cost - externality_alpha*ext 
    
    # Add regularization term to make the problem more well-conditioned
    reg_param = 1e-4
    regularization = reg_param * cp.sum_squares(opt_features)
    utility = utility - regularization
   
    constraints = [opt_features >= lower_bound, opt_features <= upper_bound]
    objective = cp.Maximize(utility)
    problem = cp.Problem(objective, constraints)    
    cvx_layer = CvxpyLayer(problem, parameters=[true_features, weight_param], variables=[opt_features])
    return cvx_layer


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_dataset()
    # train, val, strat_val = train_model(X_train, y_train, X_test, y_test, strat=False)
    # print("Completed Non-Strategic Training")
    # plot_non_strat(train, val, strat_val)

    train_strat, _, strat_val_start = train_model(X_train, y_train, X_test, y_test, strat=True)
    print("Completed Strategic Training")
    # plot_all(train, train_strat, strat_val, strat_val_start)

