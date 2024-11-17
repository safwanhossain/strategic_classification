import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

import os

plt.rcParams.update({
    "text.usetex": True,  # Enable LaTeX
    "font.family": "serif",  # Use serif fonts
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional: load additional LaTeX packages
})

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

def load_data(filename):
    # Load the results from the pickle file
    with open(filename, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)

    num_runs = results_dict["strategic_train_losses"].shape[0]
    #num_runs = 15

    ret_dict = {}

    mean = np.mean(results_dict["baseline_strat_val_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["baseline_strat_val_losses"], axis=0)) / np.sqrt(num_runs)
    ret_dict["baseline_strat_val_losses"] = (mean, var)

    mean = np.mean(results_dict["baseline_strat_val_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["baseline_strat_val_accuracies"], axis=0)) / np.sqrt(num_runs)
    ret_dict["baseline_strat_val_accuracies"] = (mean, var)

    mean = np.mean(results_dict["strategic_train_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_train_losses"], axis=0)) / np.sqrt(num_runs)
    ret_dict["strategic_train_losses"] = (mean, var)

    mean = np.mean(results_dict["strategic_train_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_train_accuracies"], axis=0)) / np.sqrt(num_runs)
    ret_dict["strategic_train_accuracies"] = (mean, var)

    mean = np.mean(results_dict["strategic_strat_val_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_strat_val_losses"], axis=0)) / np.sqrt(num_runs)
    ret_dict["strategic_strat_val_losses"] = (mean, var)

    mean = np.mean(results_dict["strategic_strat_val_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_strat_val_accuracies"], axis=0)) / np.sqrt(num_runs)
    ret_dict["strategic_strat_val_accuracies"] = (mean, var) 

    mean = np.mean(results_dict["cost_only_strat_val_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["cost_only_strat_val_losses"], axis=0)) / np.sqrt(num_runs)
    ret_dict["cost_only_strat_val_losses"] = (mean, var)  
    
    mean = np.mean(results_dict["cost_only_strat_val_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["cost_only_strat_val_accuracies"], axis=0)) / np.sqrt(num_runs)
    ret_dict["cost_only_strat_val_accuracies"] = (mean, var)   

    return ret_dict

def plot_results(ret_dict, ax):
    num_epochs = 40
    conf = 1.645
    alpha = 0.05
    #type, type_name = "accuracies", "Acc"
    type, type_name = "losses", "Loss"

    # Plot on log scale
    ax.set_yscale('log')

    # Increase font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)  # For major ticks
    
    # Loss plot
    mean, std = ret_dict[f"strategic_train_{type}"][0][:num_epochs], ret_dict[f"strategic_train_{type}"][1][:num_epochs]
    ax.plot(mean, color='red', label=f'Strat Training {type_name}')
    ax.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="red", alpha=alpha)

    mean, std = ret_dict[f"strategic_strat_val_{type}"][0][:num_epochs], ret_dict[f"strategic_strat_val_{type}"][1][:num_epochs]
    ax.plot(mean, color='blue', label=f'Strat Valid {type_name} (Strat Training)')
    ax.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="blue", alpha=alpha)

    mean, std = ret_dict[f"baseline_strat_val_{type}"][0][:num_epochs], ret_dict[f"baseline_strat_val_{type}"][1][:num_epochs]
    ax.plot(mean, color='green', label='Strat Valid Loss (Baseline Training)')
    ax.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="green", alpha=alpha)
    
    # mean, std = ret_dict[f"cost_only_strat_val_{type}"][0][:num_epochs], ret_dict[f"cost_only_strat_val_{type}"][1][:num_epochs] 
    # ax.plot(mean, color='green', label=f'Strat Valid {type_name} (Cost-only Training)')
    # ax.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="green", alpha=alpha)

    # Set specific ticks
    ax.set_xticks([0, 10, 20, 30, 40])  # 4 ticks on x-axis
    #ax.yaxis.set_major_locator(LinearLocator(numticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))


if __name__ == "__main__":
    exp_names = [
        "exp_c1_e05_n8",
        "exp_c2_e1_n8",
        "exp_c3_e15_n8",
        "exp_c1_e05_n10",
        "exp_c2_e1_n10",
        "exp_c3_e15_n10" ,
        "exp_c1_e05_n12" ,
        "exp_c2_e1_n12" ,
        "exp_c3_e15_n12" ,
    ]
    exp_names = [name + "_results.pkl" for name in exp_names]

    fig, axs = plt.subplots(3, 3, figsize=(12, 7))  # Create a 3x3 grid of subplots
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust spacing between plots

    # Headers for columns
    column_headers = [r"$(\alpha = 0.1, \beta = 0.05)$", r"$(\alpha = 0.3, \beta = 0.15)$", r"$(\alpha = 0.5, \beta = 0.25)$"]

    # Add headers to the subplots
    for col, header in enumerate(column_headers):
        axs[0, col].set_title(header, fontsize=14, pad=10)

    # (0,0)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[0]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[0,0])

    # (0,1)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[1]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[0,1])

    # (0,2)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[2]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[0,2])

     # (1,0)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[3]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[1,0])

    # (1,1)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[4]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[1,1])

    # (1,2)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[5]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[1,2])

     # (2,0)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[6]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[2,0])

    # (2,1)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[7]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[2,1])

    # (2,2)    
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_names[8]}")
    data_dict = load_data(filename)
    plot_results(data_dict, axs[2,2])

    # Add Labels and Legend as needed
    axs[0,0].set_ylabel("BCELoss " + r"$(k=8)$", fontsize=14)
    axs[1,0].set_ylabel("BCELoss " + r"$(k=10)$", fontsize=14)
    axs[2,0].set_ylabel("BCELoss " + r"$(k=12)$", fontsize=14)

    axs[2,0].set_xlabel("Epochs", fontsize=14)
    axs[2,1].set_xlabel("Epochs", fontsize=14)
    axs[2,2].set_xlabel("Epochs", fontsize=14)

    axs[0,2].legend(fontsize=10)
    
    for ax in axs.flat:  # Apply to all subplots
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)  # Disable scientific notation
        ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()

