import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import Data_cleanup
import HyperParameters as H
import Utils as U
import Dataset_Class
import json
import math
import random
from itertools import permutations
import Conversions
import RL_Environment
import Tokenizer
import TF_Model
import itertools
from scipy.optimize import linear_sum_assignment

device = H.device
print(device)
model = H.MODEL_NAME + '.pth'

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    with open(U.known_pokemon, 'r') as j:
        known_pokemon = json.load(j)

    teams_data = torch.load(U.team_tensors, weights_only=True)

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_cleanup.clean_data()
    with open(U.known_pokemon, 'r') as j:
        known_pokemon = json.load(j)

    teams_data = torch.load(U.team_tensors, weights_only=True)

#print(teams_data)
tokenizer = Tokenizer.tokenizer(known_pokemon)

#Model Parameters
input_size = len(tokenizer)
embedding_dim = math.floor(math.sqrt(input_size))
while embedding_dim%H.NUM_HEADS != 0:
    embedding_dim -= 1

print(f"input_size: {input_size}; embedding_dim: {embedding_dim}")

#LOAD TRAINED MODEL
Model = TF_Model.TeamBuilder(input_size, embedding_dim)
try:
    # Try loading the model weights on the same device (GPU or CPU)
    Model.load_state_dict(torch.load((U.MODEL_FOLDER / model).resolve(), weights_only=True))
except:
    # In case there's a device mismatch, load the model weights on the CPU
    Model.load_state_dict(torch.load((U.MODEL_FOLDER / model).resolve(), weights_only=True, map_location=torch.device('cpu')))
Model.to(device)

Agent = RL_Environment.Environment(Model, known_pokemon, teams_data, tokenizer)

for x in range(5):
    Agent.step()
    Agent.calculate_reward()
    print(Agent.reward)

Agent.reset()