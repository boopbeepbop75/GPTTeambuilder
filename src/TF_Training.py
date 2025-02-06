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

#print(tokenizer)

labels = torch.zeros(teams_data.shape[0], 6, 1) #labels

print(f"Num tokens: {len(tokenizer)}; num total pokemon: {len(teams_data) * 3}; num known pokemon: {len(known_pokemon)}")
print(teams_data.shape)
print(labels.shape)
feature_size = len(known_pokemon[0]['label'])
print(f"feature_size: {feature_size}")

#input("Press enter to continue...")

#print(labels)
error_count = 0
total_mons = 0
for i, team in enumerate(teams_data):
    for idx, mon in enumerate(team):
        total_mons += 1
        try:
            labels[i, idx, 0] = tokenizer.index(mon.tolist())
        except:
            error_count += 1

print(f"Processed {total_mons} mons with {error_count} errors ({(error_count/total_mons)*100:.2f}%)")
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
while embedding_dim%H.NUM_HEADS != 0:
    embedding_dim -= 1
print(f"input_size: {input_size}; embedding_dim: {embedding_dim}")

model = TF_Model.TeamBuilder(input_size=input_size, embed_dim=embedding_dim)
model.to(device)
loss_fn = nn.CrossEntropyLoss() #Multilabel classification task
optimizer = torch.optim.Adam(model.parameters(), lr=H.LEARNING_RATE)

def get_bonus_weight(epoch, start_epoch=0, end_epoch=15, start_weight=0.0, end_weight=.9):
    """
    Gradually increases the bonus weight from start_weight to end_weight over epochs
    """
    if epoch <= start_epoch:
        return start_weight
    elif epoch >= end_epoch:
        return end_weight
    else:
        # Linear interpolation between start and end weights
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return start_weight + (end_weight - start_weight) * progress

def set_loss(predictions, targets, epoch):
    total_loss = torch.tensor(0.0, device=predictions.device)
    batch_size = len(predictions)
    loss_fn = nn.CrossEntropyLoss()
    # Dynamic bonus weight that increases with epochs
    total_bonus_weight = get_bonus_weight(epoch)
    
    for batch_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Individual weights #
        species_weight, archetypes_weight, style_weight, weather_weight = .8, .0, .1, .1
        # Original set-based and cross-entropy losses
        pred = pred.view(-1, pred.size(-1))  # Flatten the prediction for softmax
        target = target.squeeze(1)

        # Generate all permutations of the target tokens (team)
        best_loss = torch.tensor(float('inf'), device=pred.device)  # Use tensor for best_loss
        best_perm = 0
        # Get all permutations of the target indices
        for perm in itertools.permutations(range(target.size(0))):
            permuted_target = target[list(perm)]  # Apply the permutation to the target
            
            # Calculate loss for this permutation
            loss = loss_fn(pred, permuted_target)
            
            # Update best_loss and best_perm together
            if loss < best_loss:
                best_loss = loss
                best_perm = perm  # Store the best permutation
        
        total_loss += best_loss  # Accumulate the loss (no conversion to float now)
        target = target[list(best_perm)]
        pred_probs = F.softmax(pred, dim=-1)
        predictions = [torch.argmax(p).item() for p in pred_probs]
        pred_tokens = [tokenizer[p] for p in predictions]
        target_tokens = [tokenizer[p] for p in target]
        pred_tokens = torch.from_numpy(np.array(pred_tokens).astype(np.float32)).to(device)
        target_tokens = torch.from_numpy(np.array(target_tokens).astype(np.float32)).to(device)

        # Species match bonus #
        species_bonus = torch.tensor(0.0, device=device)
        pred_species = pred_tokens[:, 0] #Extract the model's predicted species
        target_species = target_tokens[:, 0] #Extract the actual species
        used_species = set() #keep track of the model's predicted species
        correct_species = 0
        for x in pred_species:
            if x in target_species and x not in used_species:
                correct_species += 1
                used_species.add(x)

        species_bonus = (correct_species / 6)  # Normalize between 0 and 1
        # End Species Bonus #

        # Archetypes Weight #
        archetypes_bonus = torch.tensor(0.0, device=device)
        pred_archetypes = pred_tokens[:, H.archetypes]
        target_archetypes = target_tokens[:, H.archetypes]
        target_archetypes = target_archetypes[list(best_perm)]
        correct_archetypes = (pred_archetypes == target_archetypes).sum()
        archetypes_bonus = (correct_archetypes / 6) #normalize between 0 and 1
        # End Archetypes Weight

        # Style section #
        style_bonus = torch.tensor(0.0, device=device)
        pred_styles = pred_tokens[:, H.style_start:H.style_end]
        target_styles = target_tokens[:, H.style_start:H.style_end]
        correct_styles = 0
        num_styles_on = 0
        for ps, ts in zip(pred_styles, target_styles):
            '''if torch.equal(ps, ts):  # Compare the full style tensors
                correct_styles += 1'''
            #Compare the 'on' values in the style tensors
            for p, t in zip(ps, ts):
                if t == 1:
                    num_styles_on += 1
                    if p == 1:
                        correct_styles += 1

        style_bonus = (correct_styles / num_styles_on)
        # End Style Section #

        # Weather Section #
        weather_bonus = torch.tensor(0.0, device=device)
        #Extract weather mons before checking if it's a weather mon team
        rain_mon_pred = pred_tokens[:, H.rain_mon]
        rain_mon_target = target_tokens[:, H.rain_mon]
        rain_setter_pred = pred_tokens[:, H.rain_setter]
        sand_mon_pred = pred_tokens[:, H.sand_mon]
        sand_mon_target = target_tokens[:, H.sand_mon]
        sand_setter_pred = pred_tokens[:, H.sand_setter]
        sun_mon_pred = pred_tokens[:, H.sun_mon]
        sun_mon_target = target_tokens[:, H.sun_mon]
        sun_setter_pred = pred_tokens[:, H.sun_setter]
        snow_mon_pred = pred_tokens[:, H.snow_mon]
        snow_mon_target = target_tokens[:, H.snow_mon]
        snow_setter_pred = pred_tokens[:, H.snow_setter]
        weather = any([(rain_mon_target > 0).sum() > 0,
                        (sand_mon_target > 0).sum() > 0,
                        (sun_mon_target > 0).sum() > 0,
                        (snow_mon_target > 0).sum() > 0,
                       ])
        #print(rain_mon_target)

        # Weather Section #
        #Bonuses
        if ((rain_mon_target > 0).sum() > 0):
            if (rain_mon_pred > 0).sum() > 0 and (rain_setter_pred > 0).sum() > 0 and ((rain_setter_pred > 0).sum() + (sand_setter_pred > 0).sum() + (sun_setter_pred > 0).sum() + (snow_setter_pred > 0).sum() == 1):
                weather_bonus += 1 #Reward correct weather 
        elif ((sand_mon_target > 0).sum() > 0):
            if (sand_mon_pred > 0).sum() > 0 and (sand_setter_pred > 0).sum() > 0 and ((rain_setter_pred > 0).sum() + (sand_setter_pred > 0).sum() + (sun_setter_pred > 0).sum() + (snow_setter_pred > 0).sum() == 1):
                weather_bonus += 1
        elif ((sun_mon_target > 0).sum() > 0):
            if (sun_mon_pred > 0).sum() > 0 and (sun_setter_pred > 0).sum() > 0 and ((rain_setter_pred > 0).sum() + (sand_setter_pred > 0).sum() + (sun_setter_pred > 0).sum() + (snow_setter_pred > 0).sum() == 1):
                weather_bonus += 1        
        elif ((snow_mon_target > 0).sum() > 0):
            if (snow_mon_pred > 0).sum() > 0 and (snow_setter_pred > 0).sum() > 0 and ((rain_setter_pred > 0).sum() + (sand_setter_pred > 0).sum() + (sun_setter_pred > 0).sum() + (snow_setter_pred > 0).sum() == 1):
                weather_bonus += 1   
        elif weather_bonus != 0:
            weather_bonus /= weather_bonus # Make sure bonus always = 0 or 1

        #mon penaltys
        if (rain_mon_pred > 0).sum() > 0:
            if ((sand_setter_pred > 0).sum() > 0) or ((sun_setter_pred > 0).sum() > 0) or ((snow_setter_pred > 0).sum() > 0):
                weather_bonus = 0
        elif (sand_mon_pred > 0).sum() > 0:
            if ((rain_setter_pred > 0).sum() > 0) or ((sun_setter_pred > 0).sum() > 0) or ((snow_setter_pred > 0).sum() > 0):
                weather_bonus = 0
        elif (sun_mon_pred > 0).sum() > 0:
            if ((rain_setter_pred > 0).sum() > 0) or ((sand_setter_pred > 0).sum() > 0) or ((snow_setter_pred > 0).sum() > 0):
                weather_bonus = 0
        elif (snow_mon_pred > 0).sum() > 0:
            if ((rain_setter_pred > 0).sum() > 0) or ((sand_setter_pred > 0).sum() > 0) or ((sun_setter_pred > 0).sum() > 0):
                weather_bonus = 0
        # End Weather Section #

        # Apply scaled bonuses at the end #
        if not weather:
            style_weight += weather_weight
        #print(style_weight)

        #Scale bonuses
        bonus_weight_species = species_weight * total_loss.detach()  # Detach to avoid affecting gradients
        bonus_weight_archetypes = archetypes_weight * total_loss.detach()
        bonus_weight_style = style_weight * total_loss.detach()
        bonus_weight_weather = weather_weight * total_loss.detach()

        #Calculate bonuses
        species_bonus *= bonus_weight_species
        archetypes_bonus *= bonus_weight_archetypes
        style_bonus *= bonus_weight_style
        weather_bonus *= bonus_weight_weather
        
        # Incorporate all the loss changes #
        total_bonus = species_bonus + weather_bonus + style_bonus
        total_bonus *= total_bonus_weight
        total_loss -= total_bonus
    
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
        loss = set_loss(y_preds, y, epoch)
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
        loss = set_loss(y_preds, y, epoch)
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