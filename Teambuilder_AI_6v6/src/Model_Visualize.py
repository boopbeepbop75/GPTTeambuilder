import torch
import Utils as U
import HyperParameters as H
import torch.nn.functional as F
import random
import numpy as np
import json
import TF_Model
import Tokenizer
import math
import Conversions

# Load the preprocessed data stored in .pt files
with open(U.known_pokemon, 'r') as j:
    known_pokemon = json.load(j)

teams_data = torch.load(U.team_tensors, weights_only=True)
device = H.device
model = H.MODEL_NAME + '.pth'

#Initialize model
print("Initializing tokenizer...")
tokenizer = Tokenizer.tokenizer(known_pokemon)
input_size = len(tokenizer)
embedding_dim = math.floor(math.sqrt(input_size))
while embedding_dim%H.NUM_HEADS != 0:
    embedding_dim -= 1

print(f"input_size: {input_size}; embedding_dim: {embedding_dim}")

Model = TF_Model.TeamBuilder(input_size, embedding_dim)
try:
    # Try loading the model weights on the same device (GPU or CPU)
    Model.load_state_dict(torch.load((U.MODEL_FOLDER / model).resolve(), weights_only=True))
except:
    # In case there's a device mismatch, load the model weights on the CPU
    Model.load_state_dict(torch.load((U.MODEL_FOLDER / model).resolve(), weights_only=True, map_location=torch.device('cpu')))

Model.to(device)

Model.visualize_weather_parameters()