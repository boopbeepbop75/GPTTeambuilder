import torch
import Conversions
import numpy as np
import random

class Environment:
    def __init__(self, known_pokemon, all_style_labels, team_size=3, max_steps=5):
        self.known_pokemon = known_pokemon
        self.all_style_labels = all_style_labels
        self.team_size = team_size
        self.max_steps = max_steps
        self.state_space = None
        self.action_space = None
        self.current_state = None
        self.current_team = []
        self.step_count = 0

    def reset(self):
        self.current_state = torch.zeros(3, 4) #Initialize empty pokemon
        self.current_state[0, :] = Conversions.select_random_mon(self.known_pokemon, self.all_style_labels) #Initialize random starting type to build around 
        self.current_state = self.current_state.unsqueeze(0) #Add a batch dimension to the front
        self.current_team = []
        self.step_count = 0

        self.state_space = self.current_state.shape
        self.action_space = torch.empty(1, 30) # 1 batch dimension, 30 features
        return self.current_state

    def step(self, action):
        self.step_count += 1 #Update step count
        self.current_team.append(action) #Add new team member
        #Generate next_state

    def _calculate_reward(self, action):
        pass

    def render(self):
        """
        Visualizes the environment.
        """
        print(f"Current Team: {self.current_team}")
        print(f"Current State: {self.current_state}")
        print(f"Step Count: {self.step_count}")