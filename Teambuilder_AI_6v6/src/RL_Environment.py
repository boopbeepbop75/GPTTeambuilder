import torch
import Conversions
import numpy as np
import random
import HyperParameters as H
import gym
from gym import spaces
import random
device = H.device

class Environment:
    def __init__(self, Model, known_pokemon, known_teams, tokenizer, team_size=H.team_size):
        self.Model = Model
        self.known_pokemon = known_pokemon
        self.known_teams = known_teams
        self.tokenizer = tokenizer
        self.team_size = team_size
        self.max_steps = team_size - 1
        self.species_weight = 1
        self.style_weight = 1 - self.species_weight
        self.species_reward = torch.zeros(H.BATCH_SIZE).to(device)
        self.style_reward = torch.zeros(H.BATCH_SIZE).to(device)
        self.reward = torch.zeros(H.BATCH_SIZE).to(device)
        self.log_probs_list = []
        self.current_state = torch.zeros(H.BATCH_SIZE, 1, H.feature_size).to(torch.long) #Initialize empty pokemon
        for x in range(H.BATCH_SIZE):
            self.current_state[x, 0, :] = torch.from_numpy(np.array(self.known_pokemon[random.randint(0, len(self.known_pokemon)-1)]['label']).astype(np.float32)).to(torch.long)
        self.step_count = 1
        self.teams_species = known_teams[:, :, H.species]

    def reset(self):
        self.current_state = torch.zeros(H.BATCH_SIZE, 1, H.feature_size).to(torch.long) #Initialize empty pokemon
        for x in range(H.BATCH_SIZE):
            self.current_state[x, 0, :] = torch.from_numpy(np.array(self.known_pokemon[random.randint(0, len(self.known_pokemon)-1)]['label']).astype(np.float32)).to(torch.long)
        self.step_count = 1
        self.log_probs_list = []
        self.reward = self.reward * 0

        return self.current_state

    def step(self):
        for x in range(2,  7):
            self.step_count += 1
            self.current_state = self.current_state.to(device)
            #Generate next_state
            self.current_state, next_probs = self.Model.generate(self.current_state, 
                                    self.tokenizer, max_length=x,
                                    temperature=1.5,
                                    min_p=.00,
                                    top_k=len(self.tokenizer), 
                                    min_k = H.min_k, dynamic_k=False,
                                    repetition_penalty=1, 
                                    weather_repetition_penalty=1,
                                    hazard_rep_pen=1, track_gradients=True) #Default model parameters

            self.log_probs_list.append(next_probs)

    def calculate_reward(self):
        # Extract species from generated teams
        agent_species = self.current_state[:, :, 0]  # Shape: (BATCH_SIZE, team_size)
        agent_species_first_mon = agent_species[:, 0].unsqueeze(1)

        # Expand dimensions to allow broadcasting
        agent_species_first_mon = agent_species_first_mon.unsqueeze(0)  # Shape: (1, BATCH_SIZE)

        # Find teams where any Pokémon matches the first mon
        possible_best_team_mask = (self.teams_species.unsqueeze(1) == agent_species_first_mon)  # Shape: (num_known_teams, BATCH_SIZE, team_size)

        # Reduce across the last dimension (team members) to check if the first mon appears in each team
        possible_best_team_mask = possible_best_team_mask.any(dim=-1)  # Shape: (num_known_teams, BATCH_SIZE)

        # Get the indices for each batch where teams contain the first mon
        possible_best_team_indices = [mask.nonzero(as_tuple=True)[0] for mask in possible_best_team_mask.T]

        for i, team in enumerate(self.current_state):

            filtered_teams = self.known_teams[possible_best_team_indices[i]]
            filtered_teams_species = filtered_teams[..., 0]  # Extract species labels
            team_species = team[:, 0]
            # Compute match scores (count of matching species)
            match_scores = (filtered_teams_species == team_species).sum(dim=-1)

            # Find the maximum match score
            if match_scores.numel() > 0:
                max_score = match_scores.max()
            else:
                max_score = torch.tensor(0, device=device)  # Default to zero if no matches

            # Get all indices where the score equals the max score
            best_match_indices = (match_scores == max_score).nonzero(as_tuple=True)[0]
            '''print("--------------------------")
            print(f"best_match_indices: {best_match_indices}")
            print(f"Score: {max_score}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")'''

            for team_index in best_match_indices:
                #Keep track of each mons points
                current_team_species_rewards = torch.zeros(5, device=device)
                current_team_style_rewards = torch.zeros(5, device=device)
                for idx, j in enumerate(reversed(team)):
                    if idx == len(team) - 1 or not torch.isin(j[0], filtered_teams[team_index, :, 0]):
                        continue #skip to next team member if model generated an unideal partner

                    match_idx = torch.where(j[0] == filtered_teams[team_index, :, 0])[0].item()
                    # Calculate Style Points #
                    pred_mon_styles = j[1:-2]
                    target_mon_styles = filtered_teams[team_index, match_idx, 1:-2]
                    pred_mask = (pred_mon_styles > 0)

                    # Apply mask to both tensors to filter only the relevant values
                    filtered_pred = pred_mon_styles[pred_mask]  
                    filtered_target = target_mon_styles[pred_mask]  

                    # Now you can compare only the relevant values
                    comparison = (filtered_pred == filtered_target).sum()
                    style_points = comparison / len(filtered_pred)
                    current_team_species_rewards[idx] = current_team_species_rewards[idx] + 1
                    current_team_style_rewards[idx] = current_team_style_rewards[idx] + style_points
                    # Finish Calculating Style Points #

                '''print(current_team_species_rewards)
                print(current_team_style_rewards)'''
                self.species_weight = .5
                self.style_weight = .5
                reward_sum = ((current_team_species_rewards).sum() / (self.team_size-1)) * self.species_weight + ((current_team_style_rewards).sum() / (self.team_size-1)) * self.style_weight
                if reward_sum > self.reward[i]:
                    self.reward[i] = reward_sum
                #input("---------------------------")

            #self.species_reward[i] += 

        return self.reward

    def render(self):
        """
        Visualizes the environment.
        """
        print(f"Current State: {self.current_state}")
        print(f"Step Count: {self.step_count}")
        print(f"Species reward: {self.species_reward}")
        print(f"Style reward: {self.style_reward}")
        print(f"Batch rewards: {self.reward}")

    def get_features_for_loss(self):
        tokens = []
        for team in self.current_state:
            t = [self.tokenizer.index(mon.tolist()) for mon in team]
            tokens.append(t)

        return tokens, self.log_probs_list