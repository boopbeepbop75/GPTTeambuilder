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
optimizer = torch.optim.Adam(Model.parameters(), lr=(H.LEARNING_RATE/10))

EPISODES = math.ceil(len(tokenizer) / H.BATCH_SIZE) * 2
print(f"\nNum episodes per EPOCH: {EPISODES}")
best_metrics = float('-inf')  # Initialize best validation loss as infinity

for EPOCH in range(H.EPOCHS):
    print(f"\n---=== EPOCH {EPOCH + 1} ===---")
    epoch_metrics = 0
    epoch_loss = torch.tensor(0.0, device=device)
    for episode in range(EPISODES):
        Agent.step()
        Agent.calculate_reward()
        tokens, probs = Agent.get_features_for_loss() # Assuming probs are logits here, not probabilities yet.
        tokens = torch.tensor(tokens, device=device) # Ensure tokens are on the correct device
        probs = torch.stack(probs).to(device) # Ensure probs are on the correct device

        batch_loss = torch.tensor(0.0, device=device)
        for team in range(H.BATCH_SIZE):
            team_probs_logits = probs[:, team, :] # Shape: [num_tokens_per_team, vocab_size] (logits)
            # Corrected line: Take ALL tokens, not from index 1 onwards
            team_tokens = tokens[team, 1:] # Shape: [num_tokens_per_team] (indices of chosen tokens)

            # 1. Convert logits to probabilities using softmax
            team_probs = torch.softmax(team_probs_logits, dim=-1) # Softmax along the vocab_size dimension

            # 2. Get log probabilities of the chosen tokens
            log_probs = torch.log(team_probs) # Log probabilities

            # 3. Gather the log probabilities of the *chosen* tokens
            #    For each token position, get the log probability of the token in `team_tokens`
            action_log_probs = torch.gather(log_probs, dim=1, index=team_tokens.unsqueeze(1)) # Shape: [num_tokens_per_team, 1]
            action_log_probs = action_log_probs.squeeze(1) # Shape: [num_tokens_per_team]
            #print("action_log_probs shape:", action_log_probs.shape) # Shape of action log probs

            # 4. Calculate team loss: sum of log probs * reward (negated)
            team_loss = -torch.sum(action_log_probs) * Agent.reward[team]
            #team_loss = -torch.mean(action_log_probs) * Agent.reward[team]

            batch_loss += team_loss
            entropy = -torch.sum(team_probs * log_probs, dim=-1).mean()
            entropy_bonus = H.ENTROPY * entropy * (1 - Agent.reward[team])
            batch_loss += entropy_bonus

        metrics = Agent.reward.sum()/H.BATCH_SIZE
        epoch_metrics += metrics

        batch_loss /= H.BATCH_SIZE  # Average loss across batch
        epoch_loss += batch_loss

        # Backpropagation
        Agent.reset()

    epoch_metrics /= EPISODES
    epoch_loss /= EPISODES
    optimizer.zero_grad()
    epoch_loss.backward()
    optimizer.step()
    
    print(f"Epoch: {EPOCH+1}: Metrics {epoch_metrics}")
    #Evaluate model
    if epoch_metrics > best_metrics:
        best_metrics = epoch_metrics
        # Save the model's parameters (state_dict) to a file
        torch.save(Model.state_dict(), (U.MODEL_FOLDER / (H.RL_MODEL_NAME + '.pth')).resolve())
        with open((U.MODEL_FOLDER / (H.RL_MODEL_NAME + '_loss.txt')).resolve(), 'w') as f:
            f.write(str(epoch_metrics))
        print(f'Saved best model with best metrics: {epoch_metrics:.6f}')
        epochs_no_improve = 0  # Reset counter if improvement
