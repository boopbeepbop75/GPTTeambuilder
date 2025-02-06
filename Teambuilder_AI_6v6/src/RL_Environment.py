import torch
import Conversions
import numpy as np
import random
import HyperParameters as H
import gym
from gym import spaces
import random

class Environment:
    def __init__(self, Model, known_pokemon, known_teams, tokenizer, team_size=H.team_size):
        self.Model = Model
        self.known_pokemon = known_pokemon
        self.known_teams = known_teams
        self.tokenizer = tokenizer
        self.team_size = team_size
        self.max_steps = team_size - 1
        self.reward = torch.zeros(H.BATCH_SIZE)
        self.log_probs_list = []
        self.current_state = torch.zeros(H.BATCH_SIZE, 1, H.feature_size).to(torch.long) #Initialize empty pokemon
        for x in range(H.BATCH_SIZE):
            self.current_state[x, 0, :] = torch.from_numpy(np.array(self.known_pokemon[random.randint(0, len(self.known_pokemon))]['label']).astype(np.float32)).to(torch.long)
        self.step_count = 1
        self.teams_species = known_teams[:, :, H.species]

    def reset(self):
        self.current_state = torch.zeros(H.BATCH_SIZE, 1, H.feature_size).to(torch.long) #Initialize empty pokemon
        for x in range(H.BATCH_SIZE):
            self.current_state[x, 0, :] = torch.from_numpy(np.array(self.known_pokemon[random.randint(0, len(self.known_pokemon))]['label']).astype(np.float32)).to(torch.long)
        self.step_count = 1
        self.log_probs_list = []
        self.reward = self.reward * 0

        return self.current_state

    def step(self):
        self.step_count += 1 #Update step count
        #Generate next_state
        self.current_state, next_probs = self.Model.generate(self.current_state, 
                                 self.tokenizer, max_length=self.step_count,
                                 temperature=1,
                                 top_p=1,
                                 top_k=len(self.tokenizer), 
                                 min_k = H.min_k, dynamic_k=False,
                                 repetition_penalty=1, 
                                 weather_repetition_penalty=1,
                                 hazard_rep_pen=1, track_gradients=True) #Default model parameters
        self.log_probs_list.append(next_probs)

    def calculate_reward(self):
        action = self.step_count-1
        self.check_valid_partner(action) #Do the species reward

        return self.reward
    
    def check_valid_partner(self, action):
        last_mon_genned_batch = self.current_state[:, action, 0]
        prev_team_batch = self.current_state[:, :action, 0]
        teams_species = self.teams_species
        #print(prev_team_batch)
        for i, t in enumerate(prev_team_batch):
            valid = False
            for team in teams_species:
                if torch.all(torch.isin(t, team)) and torch.all(torch.isin(last_mon_genned_batch[i], team)):
                    valid = True
                    break
            if valid:
                self.reward[i] += 1
            else:
                self.reward[i] -= 1


    def render(self):
        """
        Visualizes the environment.
        """
        print(f"Current Team: {self.current_team}")
        print(f"Current State: {self.current_state}")
        print(f"Step Count: {self.step_count}")