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
import preprocessing_funcs

# Load or preprocess data

# Load the preprocessed data stored in .pt files
with open(U.known_pokemon, 'r') as j:
    known_pokemon = json.load(j)

tokenizer = Tokenizer.tokenizer(known_pokemon)
input_size = len(tokenizer)

teams_data = torch.load(U.team_tensors, weights_only=True)

print(f"Num labels: {len(tokenizer)}; num pokemon: {len(known_pokemon)}")

count_dict = {}

def visualize_labels():
    text = ""
    for label in tokenizer:
        text += f"{label}:\n\n"
        for mon in known_pokemon:
            if mon['label'] == label:
                text += preprocessing_funcs.write_mon_text(mon, mon['name'])
    with open(U.labels, 'w') as f:
        f.write(text)
    print(f"Feature size: {len(tokenizer[0])}")

visualize_labels()

print(count_dict)