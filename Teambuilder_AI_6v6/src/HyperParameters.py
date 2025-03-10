import torch
import json

### Hyperparameters ###
train_test_split = .8
types = ["bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", 
            "ghost", "grass", "ground", "ice", "normal", "poison", "psychic", "rock", 
            "steel", "water", "none"]

types_weak_rock = ['fire', 'bug', 'flying', 'ice']

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
version = '1'
tier = 'ou'
if tier != '1v1':
    team_size = 6
else:
    team_size = 3
MODEL_NAME = f'Model_gen_{gen}_{tier}_v{version}'
RL_MODEL_NAME = f'Model_gen_{gen}_{tier}_v{version}_RL'
feature_size = 55
non_style_size = 6

minimum_training_loss = .025

BATCH_SIZE = 16
DROPOUT_RATE = .5
HIDDEN_UNITS = 40
LEARNING_RATE = .001
NUM_HEADS = 2
NUM_LAYERS = 2
loss_gap_threshold = 2
loss_ratio_threshold = 2
EPOCHS = 50
PATIENCE = 5  # Number of epochs to wait before early stopping
ENTROPY = .1

### Sampling Settings ###
temperature = 1
temp_redux = 1
threshold=0.00
thresholds = [0.001, 0.01, 0.05]
min_p = .05
top_k = 10000
min_k = 6
dynamic_k = False
repetition_penalty = 1
double_weather_penalty = 0
hazard_setter_penalty = 0

AMOUNT = 20

### Feature Poisitions ###
style_start = 3
style_end = -10
species = 0
archetypes = 1
damage_class = 2
rocker = 11
spiker = 12
tspiker = 13
spinner = 14
defogger = 15
choice = 16
surprise = 22
scarfer = 25
zmove = 26
booster = 27
screens = 36
sasher = 37
boots = 41
weather_start = -10
weather_end = -2
rain_setter = -10
rain_mon = -9
sand_setter = -8
sand_mon = -7
sun_setter = -6
sun_mon = -5
snow_setter = -4
snow_mon = -3
type1 = -2
type2 = -1

#Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"