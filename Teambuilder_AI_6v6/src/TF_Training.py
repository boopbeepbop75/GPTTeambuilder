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

def get_bonus_weight(epoch, start_epoch=0, end_epoch=15, start_weight=1, end_weight=1):
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

'''def set_loss(predictions, targets, epoch):
    total_loss = torch.tensor(0.0, device=predictions.device)
    batch_size = len(predictions)
    loss_fn = nn.CrossEntropyLoss()
    # Dynamic bonus weight that increases with epochs
    total_bonus_weight = get_bonus_weight(epoch)
    
    for batch_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Individual weights #
        species_weight, hazards_weight, removal_weight, style_weight, weather_weight = .7, .125, .125, .05, .0
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
        predictions = [torch.multinomial(p, 1).item() for p in pred_probs]
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
        correct_species_list = []
        for x in pred_species:
            if x in target_species and x not in used_species:
                correct_species += 1
                used_species.add(x)


        species_bonus = (correct_species / 6)  # Normalize between 0 and 1
        # End Species Bonus #

        # Style section #
        style_bonus = torch.tensor(0.0, device=device)
        pred_styles = pred_tokens[:, H.style_start:H.style_end]
        target_styles = target_tokens[:, H.style_start:H.style_end]
        correct_styles = 0
        num_styles_on = 0
        for ps, ts in zip(pred_styles, target_styles):
            i = 0 
            for p, t in zip(ps, ts):
                if t == 1 and i not in {H.spiker, H.rocker, H.tspiker, H.spinner, H.defogger}:
                    num_styles_on += 1
                    if p == 1:
                        correct_styles += 1
                i += 1

        style_bonus = (correct_styles / num_styles_on)
        # End Style Section #

        # Hazard Section #
        pred_hazards = pred_tokens[:, H.rocker:H.tspiker+1]
        target_hazards = target_tokens[:, H.rocker:H.tspiker+1]
        num_hazards_on, correct_hazards = 0, 0
        for ph, th in zip(pred_hazards, target_hazards):
            for p, t in zip(ph, th):
                if t == 1:
                    num_hazards_on += 1
                    if p == 1:
                        correct_hazards += 1
        if ((pred_hazards[:, 0] > 0).sum() > 1) or ((pred_hazards[:, 1] > 0).sum() > 1) or ((pred_hazards[:, 2] > 0).sum() > 1):
            correct_hazards = 0
        if num_hazards_on != 0:
            hazards_bonus = (correct_hazards / num_hazards_on)
        if correct_hazards == 0 and num_hazards_on == 0:
            hazards_bonus = 1
        # End Hazard Section #

        # Removal Section #
        pred_removal = pred_tokens[:, H.spinner:H.defogger+1]
        target_removal = target_tokens[:, H.spinner:H.defogger+1]
        num_removal_on, correct_removal = 0, 0
        for pr, tr in zip(pred_removal, target_removal):
            for p, t in zip(pr, tr):
                if t == 1:
                    num_removal_on += 1
                    if p == 1:
                        correct_removal += 1
        if num_removal_on != 0:
            removal_bonus = (correct_removal / num_removal_on)
        if correct_removal == 0 and num_removal_on == 0:
            removal_bonus = 1
        if (pred_removal[:, 0] > 0).sum() > (target_removal[:, 0] > 0).sum():
            removal_bonus = 0
        if (pred_removal[:, 1] > 0).sum() > (target_removal[:, 1] > 0).sum():
            removal_bonus = 0
        # End Removal Section #

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
        bonus_weight_species = species_weight * best_loss.detach()  # Detach to avoid affecting gradients
        bonus_weight_hazards = hazards_weight * best_loss.detach()
        bonus_weight_removal = removal_weight * best_loss.detach()
        bonus_weight_style = style_weight * best_loss.detach()
        bonus_weight_weather = weather_weight * best_loss.detach()

        #Calculate bonuses
        species_bonus *= bonus_weight_species
        hazards_bonus *= bonus_weight_hazards
        removal_weight *= bonus_weight_removal
        style_bonus *= bonus_weight_style
        weather_bonus *= bonus_weight_weather
        
        # Incorporate all the loss changes #
        total_bonus = species_bonus + hazards_bonus + removal_bonus + style_bonus + weather_bonus
        total_bonus *= total_bonus_weight
        total_loss -= total_bonus
    
    return total_loss / batch_size'''

def set_loss(predictions, targets, epoch):
    total_loss = torch.tensor(0.0, device=predictions.device)
    batch_size = len(predictions)
    loss_fn = nn.CrossEntropyLoss()
    # Dynamic bonus weight that increases with epochs
    total_bonus_weight = get_bonus_weight(epoch)
    
    for batch_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Individual weights #
        species_weight, hazards_weight, removal_weight, style_weight, weather_weight = .8, .0, .0, .2, .0
        
        # Reshape predictions and targets
        pred = pred.view(-1, pred.size(-1))  # Flatten the prediction for softmax
        target = target.squeeze(1)

        # Find best permutation for permutation-invariant loss
        best_loss = torch.tensor(float('inf'), device=pred.device)
        best_perm = 0
        
        for perm in itertools.permutations(range(target.size(0))):
            permuted_target = target[list(perm)]
            loss = loss_fn(pred, permuted_target)
            
            if loss < best_loss:
                best_loss = loss
                best_perm = perm
        
        # Add base loss and get permuted target
        total_loss += best_loss
        permuted_target = target[list(best_perm)]
        
        # Get predicted tokens using multinomial sampling for diversity
        pred_probs = F.softmax(pred, dim=-1)
        sampled_indices = torch.stack([torch.multinomial(p, 1).squeeze() for p in pred_probs])

        # Convert to actual tokens for reward calculation
        pred_tokens = torch.stack([torch.tensor(tokenizer[idx.item()], device=device) for idx in sampled_indices])
        target_tokens = torch.stack([torch.tensor(tokenizer[idx.item()], device=device) for idx in permuted_target])
        
        # Ensure tokens are properly shaped and on the right device
        pred_tokens = pred_tokens.float().to(device)
        target_tokens = target_tokens.float().to(device)
        
        # Team coherence tracking variables
        team_coherence = 1
        team_has_removal = 0
        pred_team_has_removal = 0
        num_weather_setters = 0
        has_rocker = 0
        needs_rocker = False
        needs_removal = False
        has_removal = False

        # Species match bonus
        species_bonus = torch.tensor(0.0, device=device)
        pred_species = pred_tokens[:, 0]  # Extract species from tokens
        target_species = target_tokens[:, 0]
        used_species = set()
        
        # Style section
        style_bonus = torch.tensor(0.0, device=device)
        pred_styles = pred_tokens[:, H.style_start:H.weather_end]
        target_styles = target_tokens[:, H.style_start:H.weather_end]
        
        # Extract weather mons
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

        # Weather Section
        weather_penalty = False
        weather_setter_penalty = False
        
        if (rain_mon_pred > 0).sum() > 0 and (rain_setter_pred > 0).sum() != 1:
            weather_penalty = True
        if (sand_mon_pred > 0).sum() > 0 and (sand_setter_pred > 0).sum() != 1:
            weather_penalty = True
        if (sun_mon_pred > 0).sum() > 0 and (sun_setter_pred > 0).sum() != 1:
            weather_penalty = True
        if (snow_mon_pred > 0).sum() > 0 and (snow_setter_pred > 0).sum() != 1:
            weather_penalty = True

        if ((rain_setter_pred > 0).sum() + (sand_setter_pred > 0).sum() + 
            (sun_setter_pred > 0).sum() + (snow_setter_pred > 0).sum()) > 1:
            weather_setter_penalty = True

        # Track rewards per Pokémon
        species_rewards = torch.zeros(team_size, device=device)
        style_rewards = torch.zeros(team_size, device=device)

        types_list_1 = pred_tokens[:, -2]
        types_list_2 = pred_tokens[:, -1]  # Fixed bug: this was -2 in original code

        # Calculate Species and Style Rewards
        important_styles = {H.rain_setter, H.sand_setter, H.sun_setter, H.snow_setter, 
                           H.rain_mon, H.sand_mon, H.snow_mon, H.sun_mon, H.spinner, 
                           H.defogger, H.rocker, H.spiker, H.tspiker, H.choice, 
                           H.surprise, H.scarfer, H.zmove, H.booster, H.screens, H.sasher}
        
        for i, pred in enumerate(pred_species):
            # Skip duplicate mon generations
            if pred.item() in used_species:
                continue
            else:
                used_species.add(pred.item())
            
            # Find matching Pokémon in target
            matching = (pred == target_species)
            count = torch.sum(matching).item()
            
            if count != 1:
                continue  # Skip if not exactly one match
                
            index = torch.nonzero(matching).squeeze().item()

            # Style Rewards
            give_species_reward = True
            correct_styles, num_styles_on = 0, 0
            
            for j, (ps, ts) in enumerate(zip(pred_styles[i], target_styles[index])):
                if ts == 1:  # Only count styles that should be present
                    num_styles_on += 1
                    if j in {H.defogger, H.spinner}:
                        team_has_removal += 1
                        needs_removal = True
                    if j == H.rocker:
                        needs_rocker = True
                    
                    # Check if prediction style is correct
                    if ps == 1:
                        correct_styles += 1
                        if j in {H.defogger, H.spinner}:
                            pred_team_has_removal += 1
                            has_removal = True
                        elif j in {H.rain_setter, H.sand_setter, H.sun_setter, H.snow_setter}:
                            num_weather_setters += 1
                    else:
                        # Withhold reward for incorrect important attributes
                        if j in important_styles:
                            give_species_reward = False
                            break
                
                # Check for incorrectly added important styles
                if ps == 1 and ts == 0 and j in important_styles:
                    give_species_reward = False
                    break
                
                # Track rocker presence
                if ps == 1 and j == H.rocker:
                    has_rocker += 1
            
            # Calculate style bonus ratio
            style_bonus_ratio = 1
            if num_styles_on > 0:
                style_bonus_ratio = correct_styles / num_styles_on
                
            # Apply rewards if key attributes are correct
            if give_species_reward:
                species_bonus += 1  # Reward for correct Pokémon
                style_bonus += style_bonus_ratio
                species_rewards[i] += 1
                style_rewards[i] += style_bonus_ratio
        
        # Remove rewards for rock-weak Pokémon if no hazard removal
        if needs_removal and not has_removal:
            for i, (mon_t1, mon_t2) in enumerate(zip(types_list_1, types_list_2)):
                if mon_t1 in H.types_weak_rock or mon_t2 in H.types_weak_rock:
                    species_bonus -= species_rewards[i]
                    style_bonus -= style_rewards[i]

        # Remove rewards for weather Pokémon if no weather setter
        if weather_penalty:
            for i, pred in enumerate(pred_tokens):
                pred_weather_tokens = pred[H.weather_start:H.weather_end]
                if (pred_weather_tokens > 0).sum() > 0:
                    species_bonus -= species_rewards[i]
                    style_bonus -= style_rewards[i]

        # Enforce stealth rock requirement
        if needs_rocker and has_rocker != 1:
            team_coherence -= 0.25

        # Normalize bonuses to 0-1 range
        species_bonus = species_bonus / team_size
        style_bonus = style_bonus / team_size
        
        # Scale bonuses based on loss magnitude and weights
        bonus_weight_species = species_weight * best_loss.detach()
        bonus_weight_style = style_weight * best_loss.detach()

        species_bonus *= bonus_weight_species
        style_bonus *= bonus_weight_style
        
        # Calculate final bonus
        total_bonus = (species_bonus + style_bonus) * total_bonus_weight * team_coherence
        
        # Subtract bonus from loss (reward is negative loss)
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
    '''if (testing_loss - training_loss > H.loss_gap_threshold) or (testing_loss / training_loss > H.loss_ratio_threshold):
        print("Stopping early due to overfitting (loss gap or ratio threshold exceeded).")
        break'''

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