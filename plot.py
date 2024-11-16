import pickle
import numpy as np

import matplotlib.pyplot as plt

def load_data(filename):
    # Load the results from the pickle file
    with open(filename, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)

    ret_dict = {}

    mean = np.mean(results_dict["baseline_strat_val_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["baseline_strat_val_losses"], axis=0))
    ret_dict["baseline_strat_val_losses"] = (mean, var)

    mean = np.mean(results_dict["baseline_strat_val_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["baseline_strat_val_accuracies"], axis=0))
    ret_dict["baseline_strat_val_accuracies"] = (mean, var)

    mean = np.mean(results_dict["strategic_train_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_train_losses"], axis=0))
    ret_dict["strategic_train_losses"] = (mean, var)

    mean = np.mean(results_dict["strategic_train_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_train_accuracies"], axis=0))
    ret_dict["strategic_train_accuracies"] = (mean, var)

    mean = np.mean(results_dict["strategic_strat_val_losses"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_strat_val_losses"], axis=0))
    ret_dict["strategic_strat_val_losses"] = (mean, var)

    mean = np.mean(results_dict["strategic_strat_val_accuracies"], axis=0)
    var = np.sqrt(np.var(results_dict["strategic_strat_val_accuracies"], axis=0))
    ret_dict["strategic_strat_val_accuracies"] = (mean, var)  

    return ret_dict

def plot_results(ret_dict):
    num_epochs = 40
    conf = 1.645
    plt.figure()
    
    # Loss plot
    mean, std = ret_dict["strategic_train_losses"][0], ret_dict["strategic_train_losses"][1]
    plt.plot(mean, color='red', label='Strategic Training Loss')
    plt.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="red", alpha=0.5)

    mean, std = ret_dict["strategic_train_losses"][0], ret_dict["strategic_train_losses"][1]
    plt.plot(mean, color='red', label='Strategic Training Loss')
    plt.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="red", alpha=0.5)

    mean, std = ret_dict["strategic_train_losses"][0], ret_dict["strategic_train_losses"][1]
    plt.plot(mean, color='red', label='Strategic Training Loss')
    plt.fill_between(range(num_epochs), mean - conf*std, mean + conf*std, color="red", alpha=0.5)

