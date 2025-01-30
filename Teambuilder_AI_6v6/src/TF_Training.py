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
import helper_functions
import json
import math
import RL_Model
import random
from itertools import permutations
import Conversions
import Environment
import Tokenizer
import TF_Model
import itertools
from scipy.optimize import linear_sum_assignment

device = H.device
print(device)

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

teams_data = teams_data[torch.randperm(teams_data.size(0))]  # Randomly shuffle along the first dimension


print("Tokenizing teams...")
tokenizer = Tokenizer.tokenizer(known_pokemon)
print(tokenizer[0])
print(len(tokenizer))

print(tokenizer)

labels = torch.zeros(teams_data.shape[0], 6, 1) #labels

print(f"Num tokens: {len(tokenizer)}; num total pokemon: {len(teams_data) * 3}; num known pokemon: {len(known_pokemon)}")
print(teams_data.shape)
print(labels.shape)
feature_size = len(known_pokemon[0]['label'])
print(f"feature_size: {feature_size}")

input("Press enter to continue...")

#print(labels)
for i, team in enumerate(teams_data):
    for idx, mon in enumerate(team):
        labels[i, idx, 0] = tokenizer.index(mon.tolist())
###############

team_tensors = teams_data.to(torch.long) #Features

Dataset = Dataset_Class.PokemonTeamDataset(team_tensors, labels)

#Make train / test split
train_size = int(len(team_tensors) * H.train_test_split)
test_size = len(team_tensors) - train_size
train_data, test_data = random_split(Dataset, [train_size, test_size])

#Initialize loaders
train_loader = DataLoader(train_data, batch_size=H.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=H.BATCH_SIZE, shuffle=False)

#Initialize model
input_size = len(tokenizer)

embedding_dim = math.floor(math.sqrt(input_size))
while embedding_dim%2 != 0:
    embedding_dim -= 1
print(f"input_size: {input_size}; embedding_dim: {embedding_dim}")

model = TF_Model.TeamBuilder(input_size=input_size, embed_dim=embedding_dim)
model.to(device)
loss_fn = nn.CrossEntropyLoss() #Multilabel classification task
optimizer = torch.optim.Adam(model.parameters(), lr=H.LEARNING_RATE)

def set_loss(predictions, targets):
    """
    Compute set-based loss with uniqueness constraints.
    
    Args:
        predictions: tensor of shape (batch_size, team_size, num_classes) - [8, 6, 235]
        targets: tensor of shape (batch_size, team_size, 1) - [8, 6, 1]
        input_features: tensor containing the feature vectors for each prediction
    """
    total_loss = torch.tensor(0.0, device=predictions.device)
    batch_size = len(predictions)
    
    for batch_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Original set-based and cross-entropy losses
        pred_probs = F.softmax(pred, dim=-1)  # Shape: [6, ]
        target = target.squeeze(-1)  # Shape: [6]
        target_one_hot = F.one_hot(target, num_classes=pred.size(-1)).float()  # Shape: [6, ]
        
        pred_set = pred_probs.sum(dim=0)  # Shape: [235]
        target_set = target_one_hot.sum(dim=0)  # Shape: [235]
        
        pred_set = pred_set * (target_set.sum() / pred_set.sum())
        
        set_diff_loss = F.mse_loss(pred_set, target_set)
        ce_loss = F.cross_entropy(pred, target)

        team_predictions = [torch.argmax(m).item() for m in pred_probs]
        team_predictions = [tokenizer[m] for m in team_predictions]
        features = torch.from_numpy(np.array(team_predictions).astype(np.float32)).to(torch.long)
        features = features.to(device)
        #print(features)
        
        # Extract relevant features
        species = features[:, 0]
        target_labels = [tokenizer[m] for m in target]
        target_labels = torch.from_numpy(np.array(target_labels).astype(np.float32)).to(torch.long)
        act_species = target_labels[:, 0]
        wrong_species = 0
        for mon in act_species:
            if mon not in species:
                wrong_species += 1

        rain_setter = features[:, -10]
        rain_mon = features[:, -9]  
        sand_setter = features[:, -8]
        sand_mon = features[:, -7]  
        sun_setter = features[:, -6]
        sun_mon = features[:, -5]  
        snow_setter = features[:, -4]
        snow_mon = features[:, -3]  

        # Combine losses
        combined_loss = set_diff_loss + 0.15 * ce_loss 

        # Loss Penalties #
        # Species Penalty #
        species_penalty_ratio = 2

        species_penalty_weight = 1/6
        species_penalty = wrong_species * species_penalty_weight

        species_penalty = species_penalty * (species_penalty_ratio / (1 + species_penalty_ratio)) * combined_loss
        # End Species Penalty #

        # Weather Penalty #
        weather_penalty_ratio = 1
        wrong_weather = 0

        if ((rain_mon > 0).sum() > 0) and ((rain_setter > 0).sum() == 0):
            wrong_weather += 1

        if ((sand_mon > 0).sum() > 0) and ((sand_setter > 0).sum() == 0):
            wrong_weather += 1

        if ((sun_mon > 0).sum() > 0) and ((sun_setter > 0).sum() == 0):
            wrong_weather += 1
        
        if ((snow_mon > 0).sum() > 0) and ((snow_setter > 0).sum() == 0):
            wrong_weather += 1
        
        weather_penalty = wrong_weather * (weather_penalty_ratio / 1 + weather_penalty_ratio)
        # End Weather Penalty
        combined_loss += species_penalty
        combined_loss += weather_penalty
        total_loss += combined_loss
    
    return total_loss / batch_size

'''
Training Loop
#1. Forward Pass
#2. Calculate the loss on the model's predictions
#3. Optimizer
#4. Back Propagation using loss
#5. Optimizer step
'''

train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize best validation loss as infinity
epochs_no_improve = 0

def get_highest_probability_indices(probs):
    """
    Converts a tensor of probabilities to a tensor of highest probability indices.

    Args:
        probs (torch.Tensor): A tensor of probabilities with shape (batch_size, seq_len, vocab_size).

    Returns:
        torch.Tensor: A tensor of highest probability indices with shape (batch_size, seq_len).
    """
    return torch.argmax(probs, dim=-1)

print(f"Training batch size: {len(train_loader)}; Testing batch size: {len(test_loader)}")
update_interval = math.ceil(len(train_loader) * .05)

for epoch in range(H.EPOCHS):
    print(f"=== EPOCH {epoch + 1} ===")
    model.train()
    training_loss = 0 #Track training loss across that batches
    i = 0
    print("Running Training Loop")
    for batch in train_loader:
        # Shuffle each team (row) independently
        # Unpack the batch into features and labels
        features, labels = batch

        # Generate the same random permutation for each team in the batch
        batch_size = features.size(0)
        team_size = features.size(1)
        indices = torch.stack([torch.randperm(team_size) for _ in range(batch_size)])

        # Apply the permutation to both features and labels
        shuffled_features = torch.stack([team[idx] for team, idx in zip(features, indices)])
        shuffled_labels = torch.stack([team[idx] for team, idx in zip(labels, indices)])

        # Create new batch tuple
        batch = (shuffled_features, shuffled_labels)
        x = batch[0]
        y = batch[1]

        x = x.to(device)
        y = y.to(device).to(torch.long)
        
        #1. Forward pass
        y_preds = model(x)

        #calculate loss
        loss = set_loss(y_preds, y)
        training_loss += loss.item()
        #back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        '''if i % update_interval == 0:
            print(f"{i} loops in out of {len(train_loader)}")'''

    
    training_loss /= len(train_loader)
    train_losses.append(training_loss)

    testing_loss, test_acc = 0, 0
    print("Testing the model...")
    model.eval()
    for batch in test_loader:
        # Unpack the batch into features and labels
        features, labels = batch

        # Generate the same random permutation for each team in the batch
        batch_size = features.size(0)
        team_size = features.size(1)
        indices = torch.stack([torch.randperm(team_size) for _ in range(batch_size)])

        # Apply the permutation to both features and labels
        shuffled_features = torch.stack([team[idx] for team, idx in zip(features, indices)])
        shuffled_labels = torch.stack([team[idx] for team, idx in zip(labels, indices)])

        # Create new batch tuple
        batch = (shuffled_features, shuffled_labels)
        x = batch[0]
        y = batch[1].to(torch.long)

        x = x.to(device)
        y = y.to(device)
        
        #1. Forward pass
        y_preds = model(x)

        #calculate loss
        loss = set_loss(y_preds, y)
        testing_loss += loss.item()

    testing_loss /= len(test_loader)
    test_acc /= len(test_loader)
    val_losses.append(testing_loss)

    print(f"Train loss: {training_loss:.6f} | Test loss: {testing_loss:.6f}")

    #Evaluate model
    if testing_loss < best_val_loss:
        best_val_loss = testing_loss
        # Save the model's parameters (state_dict) to a file
        torch.save(model.state_dict(), (U.MODEL_FOLDER / (H.MODEL_NAME + '.pth')).resolve())
        with open((U.MODEL_FOLDER / (H.MODEL_NAME + '_loss.txt')).resolve(), 'w') as f:
            f.write(str(testing_loss))
        print(f'Saved best model with validation loss: {best_val_loss:.6f}')
        epochs_no_improve = 0  # Reset counter if improvement
    else:
        epochs_no_improve += 1
        print(f'Num epochs since improvement: {epochs_no_improve}')

        #stop training if overfitting starts to happen
        if epochs_no_improve >= H.PATIENCE:
            print("Early stopping")
            break

    '''if testing_loss < H.minimum_training_loss:
        print('Ending training')
        break'''
    if (testing_loss - training_loss > H.loss_gap_threshold) or (testing_loss / training_loss > H.loss_ratio_threshold):
        print("Stopping early due to overfitting (loss gap or ratio threshold exceeded).")
        break

# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()