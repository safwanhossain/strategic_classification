import pickle
import numpy as np

def load_data(filename):
    # Load the results from the pickle file
    with open(filename, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)

    ret_dict = {}

    mean = np.mean(results_dict["baseline_strat_val_losses"], axis=0)
    var = np.var(results_dict["baseline_strat_val_losses"], axis=0) 
    ret_dict["baseline_strat_val_losses"] = (mean, var)

    mean = np.mean(results_dict["baseline_strat_val_accuracies"], axis=0)
    var = np.var(results_dict["baseline_strat_val_accuracies"], axis=0) 
    ret_dict["baseline_strat_val_accuracies"] = (mean, var)

    mean = np.mean(results_dict["strategic_train_losses"], axis=0)
    var = np.var(results_dict["strategic_train_losses"], axis=0) 
    ret_dict["strategic_train_losses"] = (mean, var)

    mean = np.mean(results_dict["strategic_train_accuracies"], axis=0)
    var = np.var(results_dict["strategic_train_accuracies"], axis=0) 
    ret_dict["strategic_train_accuracies"] = (mean, var)

    mean = np.mean(results_dict["strategic_strat_val_losses"], axis=0)
    var = np.var(results_dict["strategic_strat_val_losses"], axis=0) 
    ret_dict["strategic_strat_val_losses"] = (mean, var)

    mean = np.mean(results_dict["strategic_strat_val_accuracies"], axis=0)
    var = np.var(results_dict["strategic_strat_val_accuracies"], axis=0) 
    ret_dict["strategic_strat_val_accuracies"] = (mean, var)  

    return ret_dict

def plot_results(ret_dict):
    pass
