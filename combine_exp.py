import pickle
import numpy as np

import os

def combine_exp(exp_file, cost_exp_file):
    # Load the results from the pickle file
    with open(exp_file, 'rb') as pickle_file:
        exp_results_dict = pickle.load(pickle_file)
    
    with open(cost_exp_file, 'rb') as pickle_file:
        cost_results_dict = pickle.load(pickle_file) 

    exp_results_dict["cost_only_train_losses"] = cost_results_dict["cost_only_train_losses"]
    exp_results_dict["cost_only_train_accuracies"] = cost_results_dict["cost_only_train_accuracies"]
    exp_results_dict["cost_only_strat_val_losses"] = cost_results_dict["cost_only_strat_val_losses"]
    exp_results_dict["cost_only_strat_val_accuracies"] = cost_results_dict["cost_only_strat_val_accuracies"] 

    # Overwrite the pickle file with the modified contents
    with open(exp_file, 'wb') as pickle_file:
        pickle.dump(exp_results_dict, pickle_file)

def combine():
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
    for exp in exp_names:
        exp_suite = exp + "_results.pkl"
        exp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{exp_suite}")

        cost_only_exp_suite = "cost_only_" + exp + "_results.pkl"
        cost_only_exp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{cost_only_exp_suite}")

        combine_exp(exp_file, cost_only_exp_file) 
        print(f"Combined {exp}")

if __name__ == "__main__":
    combine()
