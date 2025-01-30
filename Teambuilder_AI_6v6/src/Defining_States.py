import json
import pokemon
import Utils as U
import random
import Data_cleanup
import numpy as np
import HyperParameters as H
from preprocessing_funcs import label_teams
import torch
import itertools

'''
States:
* Bulky Offense, phys / special
* Fast Offense, phys / special
* Setup Offense, phys / special
* Stall
* Disruptor 
* Anti-Offense, phys / special
'''

def format_data():
    with open(U.cleaned_teams_loc, 'r') as json_file:
        teams_data = json.load(json_file)

    labeled_teams, known_pokemon = label_teams(teams_data) #Define pokemon archetypes and label teams
    #Double teams for extra training data
    '''l_t_copy = []
    for team in labeled_teams:
        l_t_copy.append(team)
        l_t_copy.append(team)
    labeled_teams = l_t_copy'''

    aug_labeled_teams = []
    '''for team in labeled_teams:
        # Assuming team is a list/array of Pokemon
        team_size = len(team)
        for perm in itertools.permutations(range(team_size)):
            # Access the actual team members using the permutation
            permuted_team = [list(team[i]) for i in perm]
            aug_labeled_teams.append(permuted_team)'''
    
    aug_labeled_teams = labeled_teams

    print(f"known pokemon: {len(known_pokemon)}; teams: {len(aug_labeled_teams)}; total pokemon: {len(aug_labeled_teams)*6}")
    print(labeled_teams[0])
    print(labeled_teams[0].shape)
    
    labeled_teams = np.array(aug_labeled_teams).astype(np.float32)

    team_tensors = torch.from_numpy(labeled_teams).to(torch.long)
    print(team_tensors.shape)
    #print(team_tensors)

    #Save Formatted Data
    '''print(f"Before: {len(teams_data)}, After: {len(labeled_teams)}; # of pokemon: {len(labeled_teams)*3}")
    print(f"# of unique pokemon: {len(known_pokemon)}")'''

    print('Saving data...')
    with open(U.known_pokemon, 'w') as j:
        json.dump(known_pokemon, j, indent=4) 

    torch.save(team_tensors, U.team_tensors)

if __name__ == "__main__":
    format_data()