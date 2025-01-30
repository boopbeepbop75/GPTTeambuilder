import torch
import json
import numpy as np
import Utils as U
import Data_cleanup

#Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc


def load_data():
    try:
        # Load the preprocessed data stored in .pt files
        with open(U.known_pokemon, 'r') as j:
            known_pokemon = json.load(j)

        teams_data = np.load(U.labeled_teams)

    except:
        # If the data hasn't been preprocessed, clean it, preprocess it, and save it
        print("data not found")
        Data_cleanup.clean_data()
        with open(U.known_pokemon, 'r') as j:
            known_pokemon = json.load(j)

        teams_data = np.load(U.labeled_teams)
    return known_pokemon, teams_data