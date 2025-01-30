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

# Load or preprocess data

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
while embedding_dim%2 != 0:
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


def generate_teams(certain_mon=None): #Generate the teams starting with a random token
    original_mons = []
    team_batch = torch.zeros(H.BATCH_SIZE, 1, H.feature_size) #8 size, start with 1 mon per team, H features per mon
    for x in range(H.BATCH_SIZE):
        if certain_mon == None:
            ran_choice = random.choice(known_pokemon)
            random_mon = ran_choice['label']
            orig = known_pokemon.index(ran_choice)
            original_mons.append(orig)
        else: 
            random_mon = known_pokemon[certain_mon]['label']
            original_mons.append(certain_mon)
        random_mon = np.array(random_mon).astype(np.float32)
        random_mon = torch.from_numpy(random_mon)
        team_batch[x, 0] = random_mon[:]
        
    team_batch = team_batch.to(torch.long)
    team_batch = team_batch.to(device)
    #print(original_mons)
    #print(f"team_batch: {team_batch.flatten()}")
    '''print(team_batch)
    print(team_batch.shape)
    print(original_mons)
    input()'''
    teams_batch = Model.generate(team_batch, 
                                 tokenizer, 
                                 temperature=H.temperature, 
                                 threshold=H.threshold, top_k=H.top_k, 
                                 min_k = H.min_k, dynamic_k=H.dynamic_k,
                                 repetition_penalty=H.repetition_penalty, 
                                 weather_repetition_penalty=H.double_weather_penalty,
                                 hazard_rep_pen=H.hazard_setter_penalty)
    return teams_batch, original_mons

def convert_teams(teams, original_mons): #Transform team predictions to readable output
    #print(f"EFEF {original_mons}")
    teams = teams.tolist()
    avg_num_choices = 0
    for i, team in enumerate(teams): 
        for idx, mon in enumerate(team):

            possible_pokes = []
            for pkmn in known_pokemon: #Find all pokemon under that label
                if mon == pkmn['label']:
                    #print(f"count: {pkmn['count']}")
                    '''if H.gen != '6':
                        for count in range(pkmn['count']):
                            possible_pokes.append(pkmn)
                    else:'''
                    possible_pokes.append(pkmn)
            
            mon = random.choice(possible_pokes) #randomly choose a match
            pkmn_names = [m['name'] for m in possible_pokes]
            #print(pkmn_names)
            avg_num_choices += len(possible_pokes)
            if idx == 0:
                mon = known_pokemon[original_mons[i]]
            teams[i][idx] = mon
    print(f"Average num mons per prediction: {avg_num_choices/(H.BATCH_SIZE*3)}")
    return teams

def write_mon_text(mon, name):
    t = ""
    if mon['name'] == 'darmanitan-galar-standard ':
        mon['name'] = 'darmanitan-galar '
    t += f"AI mon {name} ({mon['name']}) @ {mon['item']} \nAbility: {mon['abilities']} \nEVs: {mon['evs']} \n{mon['nature']} Nature "
    for move in mon['moves']:
        t += f"\n- {move['name']} "
    t += "\n\n"
    return t

def write_team_text(teams):
    text = ""
    for team in teams:
        text += f"=== [gen{H.gen}{H.tier}] DGPT {team[0]['name']}, {team[1]['name']}, {team[2]['name']}, {team[3]['name']}, {team[4]['name']}, {team[5]['name']}; temp: {H.temperature}; "
        if H.dynamic_k:
            text += f"k: Dynamic === \n\n"
        else:
            text += f"k: {H.top_k} === \n\n"
        for i, mon in enumerate(team):
            text += write_mon_text(mon, i+1)
        text += "\n"
    with open(U.generated_teams, 'a') as f:
        f.write(text)

if __name__ == "__main__":
    with open(U.generated_teams, 'w') as f:
        f.write("")
    num_batches = 0
    i = 1
    #while num_batches < H.AMOUNT:
    #try:
    random_mon = random.randint(0, len(known_pokemon))
    random_mon = 1
    print(random_mon)

    teams_batch, original_mons = generate_teams(certain_mon=None)

    teams = convert_teams(teams_batch, original_mons)
    num_batches = H.BATCH_SIZE * i
    i+=1
    '''except Exception as e:
        print(e)'''
    write_team_text(teams)
