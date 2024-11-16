import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import json
import pickle

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from hyperparams import (
    exp_c2_e1_n8,
    exp_c3_e15_n8,
    exp_c2_e1_n10,
    exp_c3_e15_n10, 
    exp_c2_e1_n12,
    exp_c3_e15_n12,
    exp_c2_e1_n12,
    exp_c3_e15_n12,
    exp_c2_e1_n16,
    exp_c3_e15_n16,
    exp_c2_e1_n16,
    exp_c3_e15_n16, 
)

VERBOSE = False
NUM_EPOCHS = 2
NUM_RUNS = 1

# Add this near the top of the file, after imports
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if VERBOSE:
    print(f"Running on {device}")


def make_dataset(exp):
    # Create a synthetic dataset
    X, y = make_classification(
        n_samples=exp["num_agents"]*exp["num_samples_mult"], 
        n_features=5, 
        n_informative=5, 
        n_redundant=0, 
        n_classes=2, 
        random_state=42,
        weights=[0.5, 0.5],
        class_sep=exp["class_sep"]
    )

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def make_dataloader(X_train_tensor, y_train_tensor, batch_size):
    # Create DataLoader for batching
    # Batching here is really the number of "agents" participating in the game
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader 


# Define a simple linear classifier using a weight vector
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, batch_dim, strat, init_vals, exp):
        super(LinearClassifier, self).__init__()
        if not init_vals:
            self.weights = nn.Parameter(torch.normal(torch.zeros(feature_dim)).float())
            self.bias = nn.Parameter(torch.normal(torch.zeros(1)).float())
        else:
            self.weights = nn.Parameter(init_vals[0].float())
            self.bias = nn.Parameter(init_vals[1].float())  
        
        self.exp = exp
        self.strat = strat
        if self.strat:
            self.train_cvx_layer = create_cvx_layer((batch_dim, feature_dim), self.weights.shape, exp)
    
    def get_NE_features(self, x):
        """ Given a batch of true features x, which represent the simultaneous
        participants in this round, return the features that these agents will
        play in Nash Equilibrium.
        
        This essentially means solving the convex potential game. Note that this
        must be done in a way that allows for taking gradients over convex optimization.
        
        This will require using the weights, and treating the "utility" as just the logits.
        """
        return x
    
    def forward(self, x):
        if self.strat:
            NE_features, = self.train_cvx_layer(x, self.weights, self.bias.expand(self.exp["num_agents"]))
        else:
            NE_features = x
        
        # Get the outcome on the NE features
        logits = torch.matmul(NE_features, self.weights) + self.bias # Matrix multiplication
        return logits


def calculate_accuracy(outputs, labels):
    # Convert logits to predictions
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    #predictions = (outputs > 0.5).float()
    return (predictions == labels).float().mean().item()


def train_model(X_train, y_train, X_test, y_test, strat, exp, num_epochs=70, init_vals=None):
    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = LinearClassifier(input_dim, exp["num_agents"], strat, init_vals, exp).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.01)  # Increased from 1e-4

    train_loader = make_dataloader(X_train, y_train, batch_size=exp["num_agents"])
    test_loader = make_dataloader(X_test, y_test, batch_size=exp["num_agents"])

    x_test, y_test = next(iter(test_loader))
    validation_cvx_layer = create_cvx_layer(x_test.shape, model.weights.shape, exp)

    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    strat_val_losses = []
    strat_val_accuracies = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if strat:
                norm_cost = torch.norm(model.weights, 2)**2 + torch.norm(model.bias, 2)**2
                loss = loss + exp["reg_lambda"]*norm_cost
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_acc += calculate_accuracy(outputs, y_batch)

        # Validation phase
        if (epoch+1) % 1 == 0:
            train_losses.append(epoch_train_loss / len(train_loader))
            train_accuracies.append(epoch_train_acc / len(train_loader))
            model.eval()
            with torch.no_grad():
                if not strat:
                    val_loss = 0
                    val_acc = 0
                    for test_feat, test_label in test_loader: 
                        test_outputs = model(test_feat)
                        loss = criterion(test_outputs, test_label)
                        val_loss += loss.item()
                        val_acc += calculate_accuracy(test_outputs, test_label)
                    val_losses.append(val_loss / len(test_loader))
                    val_accuracies.append(val_acc / len(test_loader))

                strat_val_loss = 0
                strat_val_acc = 0
                for test_feat, test_label in test_loader:
                    bias = model.bias.detach()
                    bias_vec = bias.expand(exp["num_agents"])
                    equi_features, = validation_cvx_layer(test_feat.detach(), model.weights.detach(), bias_vec)
                    equi_features = equi_features.detach()
                    # verify_equi(equi_features.cpu().numpy(), test_feat.cpu().numpy(), model.weights.detach().cpu().numpy(), bias.cpu().numpy())

                    strat_val_outputs = model(equi_features)
                    curr_loss = criterion(strat_val_outputs, test_label)
                    strat_val_loss += curr_loss.item()
                    strat_val_acc += calculate_accuracy(strat_val_outputs, test_label)
                strat_val_losses.append(strat_val_loss / len(test_loader))
                strat_val_accuracies.append(strat_val_acc / len(test_loader))
        
            if not strat and VERBOSE:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, Strat Val: {strat_val_losses[-1]:.4f}, Strat Val Acc: {strat_val_accuracies[-1]:.4f}')
            if strat and VERBOSE:
                print(f'Epoch [{epoch+1}/{num_epochs}], Strat Train Loss: {train_losses[-1]:.4f}, Strat Train Acc: {train_accuracies[-1]:.4f}, Strat Val Loss: {strat_val_losses[-1]:.4f}, Strat Val Acc: {strat_val_accuracies[-1]:.4f}') 
    
    ret_obj = {
        "weights" : model.weights.detach(),
        "bias" : model.bias.detach(),
        "train_losses" : train_losses, 
        "train_accuracies" : train_accuracies,
        "val_losses" : val_losses,
        "val_accuracies" : val_accuracies, 
        "strat_val_losses" : strat_val_losses, 
        "strat_val_accuracies" : strat_val_accuracies
    }
    return ret_obj
    

def verify_equi(equi_X, true_X, weight, bias, exp):
    true_features = cp.Parameter(true_X.shape, name="true_feat", value=true_X)
    equi_features = cp.Parameter(equi_X.shape, name="equi_feat", value=equi_X)
    weight_param = cp.Parameter(weight.shape, name="weight", value=weight)
    bias_param = cp.Parameter(1, name="bias", value=bias) 

    # check the best response for each agent
    for i, (equi_feat, true_feat) in enumerate(zip(equi_features, true_features)):
        # Parameter setup remains the same 
        opt_feat = cp.Variable(equi_feat.shape, name="opt_feat") 
        
        gain = opt_feat @ weight_param + bias_param
        cost = cp.norm2(opt_feat - true_feat)
        externality = 0

        for j, (j_equi_feat, j_true_feat) in enumerate(zip(equi_features, true_features)):
            if j == i:
                continue
            ext_ij = cp.square(cp.norm2(j_equi_feat - j_true_feat) + cp.norm2(opt_feat - true_feat))
            externality += ext_ij
        
        utility = gain - exp["cost_alpha"]*cost - exp["externality_alpha"]*externality
        constraints = [opt_feat >= exp["lower_bound"], opt_feat <= exp["upper_bound"]]
        objective = cp.Maximize(utility)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        diff = np.sum(np.abs(opt_feat.value - equi_feat.value))
        print(f"Opt_feat: {opt_feat.value}, equi feat: {equi_feat.value}, true_feat: {true_feat.value}")
        assert diff <= 0.05
        

def create_cvx_layer(feature_shape, weight_shape, exp):
    # Parameter setup remains the same
    n = feature_shape[0]
    true_features = cp.Parameter(feature_shape, name="true_feat")
    weight_param = cp.Parameter(weight_shape, name="weight")
    bias_param = cp.Parameter(n, name="bias") 
    opt_features = cp.Variable(feature_shape, name="feature_mat")

    # the gain from classification
    gain_vec = opt_features @ weight_param + bias_param
    gain = cp.sum(gain_vec)

    # the cost of manipulation to all agents
    cost = cp.sum(cp.norm2(true_features - opt_features, axis=1))

    # the externality for all agents
    diffs = cp.norm2(true_features - opt_features, axis=1)
    ext = 0
    for i in range(n):
        for j in range(i+1, n):
            ext += cp.square(diffs[i] + diffs[j])

    utility = exp["gain_alpha"]*gain - exp["cost_alpha"]*cost - exp["externality_alpha"]*ext 
    
    constraints = [opt_features >= exp["lower_bound"], opt_features <= exp["upper_bound"]]
    objective = cp.Maximize(utility)
    problem = cp.Problem(objective, constraints)    
    cvx_layer = CvxpyLayer(problem, parameters=[true_features, weight_param, bias_param], variables=[opt_features])
    return cvx_layer


if __name__ == "__main__":
    exps = {
        "exp_c2_e1_n10" : exp_c2_e1_n10,
        "exp_c3_e15_n10" : exp_c3_e15_n10,
        "exp_c2_e1_n12" : exp_c2_e1_n12,
        "exp_c3_e15_n12" : exp_c3_e15_n12,
        "exp_c2_e1_n16" : exp_c2_e1_n16,
        "exp_c3_e15_n16" : exp_c3_e15_n16,
        #"exp_c2_e1_n8" : exp_c2_e1_n8,
        #"exp_c3_e15_n8" : exp_c3_e15_n8,
    }
    
    for exp_name, exp_to_run in exps:
        results_dict = {
            "config" : exp_to_run,
            "num_runs" : NUM_RUNS,
            "num_epochs" : NUM_EPOCHS,
            "baseline_train_losses" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "baseline_train_accuracies" : np.zeros((NUM_RUNS, NUM_EPOCHS)), 
            "baseline_val_losses" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "baseline_val_accuracies" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "baseline_strat_val_losses" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "baseline_strat_val_accuracies" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "strategic_train_losses" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "strategic_train_accuracies" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "strategic_strat_val_losses" : np.zeros((NUM_RUNS, NUM_EPOCHS)),
            "strategic_strat_val_accuracies" : np.zeros((NUM_RUNS, NUM_EPOCHS)) 
        }

        # Run the experiments
        for run in tqdm(range(NUM_RUNS)):
            X_train, y_train, X_test, y_test = make_dataset(exp_to_run)
            baseline_ret_obj = train_model(
                X_train, y_train, X_test, y_test, 
                strat=False, exp=exp_to_run, num_epochs=NUM_EPOCHS
            )
            results_dict["baseline_train_losses"][run] = baseline_ret_obj["train_losses"]
            results_dict["baseline_train_accuracies"][run] = baseline_ret_obj["train_accuracies"]
            results_dict["baseline_val_losses"][run] = baseline_ret_obj["val_losses"]
            results_dict["baseline_val_accuracies"][run] = baseline_ret_obj["val_accuracies"]
            results_dict["baseline_strat_val_losses"][run] = baseline_ret_obj["strat_val_losses"]
            results_dict["baseline_strat_val_accuracies"][run] = baseline_ret_obj["strat_val_accuracies"]

            if VERBOSE:
                print("Finished non-strategic Train")
            
            # Adding little noise to stabalize training
            X_train += torch.normal(0, 0.01, X_train.shape).to(device)
            strat_ret_obj = train_model(
                X_train, y_train, X_test, y_test, 
                strat=True, exp=exp_to_run, num_epochs=NUM_EPOCHS
            )
            results_dict["strategic_train_losses"][run] = strat_ret_obj["train_losses"]
            results_dict["strategic_train_accuracies"][run] = strat_ret_obj["train_accuracies"]
            results_dict["strategic_strat_val_losses"][run] = strat_ret_obj["strat_val_losses"]
            results_dict["strategic_strat_val_accuracies"][run] = strat_ret_obj["strat_val_accuracies"]       

        print("Finished Experiments. Saving Results ...")
        
        # Convert numpy arrays to lists for JSON serialization
        results_dict_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in results_dict.items()}

        # Save the dictionary as a JSON file
        with open(f"{exp_name}_results.json", 'w') as json_file:
            json.dump(results_dict_serializable, json_file)

        with open(f"{exp_name}_results.pkl", 'wb') as pickle_file:
            pickle.dump(results_dict, pickle_file)

        print("Saved Results. Logging Results ...")
        wandb.init(
            project="strategic_classification",
            name=exp_name
        )
        wandb.config.update(
            exp_to_run
        )
        for epoch in range(NUM_EPOCHS):
            wandb.log({
                "baseline_train_losses" : np.mean(results_dict["baseline_train_losses"][:, epoch]),
                "baseline_train_accuracies" : np.mean(results_dict["baseline_train_accuracies"][:, epoch]), 
                "baseline_val_losses" : np.mean(results_dict["baseline_val_losses"][:, epoch]),
                "baseline_val_accuracies" : np.mean(results_dict["baseline_val_accuracies"][:, epoch]),
                "baseline_strat_val_losses" : np.mean(results_dict["baseline_strat_val_losses"][:, epoch]),
                "baseline_strat_val_accuracies" : np.mean(results_dict["baseline_strat_val_accuracies"][:, epoch]),
                "strategic_train_losses" : np.mean(results_dict["strategic_train_losses"][:, epoch]),
                "strategic_train_accuracies" : np.mean(results_dict["strategic_train_accuracies"][:, epoch]),
                "strategic_strat_val_losses" : np.mean(results_dict["strategic_strat_val_losses"][:, epoch]),
                "strategic_strat_val_accuracies" : np.mean(results_dict["strategic_strat_val_accuracies"][:, epoch]) 
            })

   
