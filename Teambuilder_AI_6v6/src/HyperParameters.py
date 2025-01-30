import torch
import json

### Hyperparameters ###
train_test_split = .8
types = ["bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", 
            "ghost", "grass", "ground", "ice", "normal", "poison", "psychic", "rock", 
            "steel", "water", "none"]

type_mapping = []

type_num = 0
for x in range(len(types)):
    for y in range(x + 1, len(types)):  # Ensure y starts from x + 1
        type_combo = [types[x], types[y]]
        type_mapping.append(type_combo)
        type_num += 1

### MODEL HYPER PARAMETERS ###
#Model name
gen = '5'
tier = 'ou'
MODEL_NAME = 'Model_gen_' + gen
feature_size = 55
non_style_size = 6

temperature = 1
threshold=0.00
top_k = 48
min_k = 6
dynamic_k = True
repetition_penalty = 1
double_weather_penalty = 0
hazard_setter_penalty = 0

AMOUNT = 20

minimum_training_loss = .025

BATCH_SIZE = 16
DROPOUT_RATE = .5
HIDDEN_UNITS = 12
LEARNING_RATE = .0002
loss_gap_threshold = .5
loss_ratio_threshold = 2
EPOCHS = 500
PATIENCE = 5  # Number of epochs to wait before early stopping
input_dim = 30 #3 pokemon / 30 vector attributes each
input_dim_model_2 = 2*30
output_dim = 60 #Attributes for 2 pokemon

#Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"