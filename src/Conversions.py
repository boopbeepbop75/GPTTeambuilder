import random
import HyperParameters as H
import numpy as np
import torch

def select_random_mon(known_pokemon, all_style_labels):
    i = random.randint(0, len(known_pokemon)-1)
    mon = np.array([known_pokemon[i]['arch_label'], known_pokemon[i]['dc_label'], all_style_labels.index(known_pokemon[i]['style_label']), known_pokemon[i]['type_label']]).astype(np.float32)
    mon = torch.from_numpy(mon).to(torch.long)
    return mon, i

def select_mon(mon, all_style_labels):
    i = mon
    mon = np.array([mon['arch_label'], mon['dc_label'], all_style_labels.index(mon['style_label']), mon['type_label']]).astype(np.float32)
    mon = torch.from_numpy(mon).to(torch.long)
    return mon, i

def show_attributes(pkmn, all_style_labels):
    if pkmn[0] == 0:
        arch = 'offense'
    else:
        arch = 'stall'
    
    if pkmn[1] == 0:
        dc = 'physical'
    elif pkmn[1] == 1:
        dc = 'special'
    else:
        dc = 'mixed'

    style = all_style_labels[pkmn[2]]
    types = H.type_mapping[pkmn[3]]
    
    return [arch, dc, style, types]

def normalize_mon(pkmn, all_style_labels):
    try:
        pkmn[2] = all_style_labels[pkmn[2]]
    except:
        #print(f"ERROR: {pkmn}")
        pass
    return pkmn

