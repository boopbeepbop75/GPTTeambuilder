import numpy as np
import HyperParameters as H
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import Utils as U

weather_moves = [['none'], ['hurricane', 'thunder'], ['none'], ['solar beam', 'solar blade', 'moonlight'], ['blizzard']]
weathers = ['none', 'drizzle', 'sand stream', 'drought', 'snow warning']
weather_abilities = [['none'], ['swift swim', 'rain dish', 'dry skin'], ['sand force', 'sand rush'], ['chlorophyll'], ['slush rush', 'ice body'], ['slush rush', 'ice body']]

def make_state(pkmn):
    archetypes = {'offense': 0, 'stall': 0}
    damage_type = {'physical': 0, 'special': 0}
    #ADD PRIORITY
    style = {'Bulky': 0, 'fast': 0, 'mega': 0, 'setup_physical': 0, 'setup_special': 0, 'setup_speed': 0, 'rocker': 0, 'spiker': 0, 'tspiker': 0, 'spinner': 0, 'defogger': 0, 'choice': 0, 'anti_offense': 0, 'disruptor': 0, 'mixed': 0, 'pivot': 0, 'anti_stall': 0, 'surprise': 0, 'trapper': 0, 'anti': '', 'scarf': 0, 'z_move': '', 'be': 0, 'types': [], 'bulky_physical': 0, 'bulky_special': 0, 'bulky_mixed': 0, 'paralysis': 0, 'burn': 0, 'toxic': 0, 'support': 0, 'screens': 0, 'sash': 0, 'lefties': 0, 'regen': 0, 'levitate': 0, 'boots': 0, 'sturdy': 0, 'unaware': 0, "m_guard": 0, 'elec_immune': 0, 'water_immune': 0, 'fire_immune': 0, 'status_absorber': 0, 'life orb': 0, 'trick_room': 0}
    weather = {'rain_setter': 0, 'rain': 0, 'sand_setter': 0, 'sand': 0, 'sun_setter': 0, 'sun': 0, 'snow_setter': 0, 'snow': 0} #label for each weather
    terrain = {'electric': 0, 'psychic': 0, 'misty': 0, 'grassy': 0}
    if 'choice' in pkmn['item'].lower():
        style['choice'] += 1
        if 'scarf' in pkmn['item'].lower():
            style['scarf'] += 1
    for move in pkmn['moves']:
        try:
            move['name'] = move['name'].lower()
            try:
                move['effect'] = move['effect'].lower()
            except:
                pass
            '''if move['name'] == 'bullet punch':
                print(move)
                input()'''
            try:
                for i, w in enumerate(weather_moves):
                    if move['name'] in w:
                        #print(f"weather effect: {w}; {move['name']}")
                        if i == 1:
                            weather['rain'] = 1
                        elif i == 2:
                            weather['sand'] = 1
                        elif i == 3:
                            weather['sun'] = 1 
                        elif i == 4:
                            weather['snow'] = 1
                        break
            except:
                pass
            if move['bp'] != 0:
                if move['damage_class'] == 'special':
                    damage_type['special'] += 1
                else:
                    damage_type['physical'] += 1
                archetypes['offense'] += 1
                try:
                    ### Attack move exceptions ###
                    if 'switches' in move['effect'] and 'user' in move['effect'] and move['name'].lower() != 'pursuit':
                        #print(f"pivot move: {move['name']}")
                        style['pivot'] += 1
                    elif move['name'] in {'pursuit', 'magma storm', 'fire spin', 'whirlpool', 'thousand waves'}:
                        style['trapper'] += 1
                    elif move['name'] == 'circle throw':
                        style['anti_offense'] += 1
                    elif move['name'] == 'rapid spin':
                        style['spinner'] += 1
                    elif 'stealth rock' in move['effect']:
                        style['rocker'] += 1
                    elif 'spikes' in move['effect'] and 'layer' in move['effect']:
                        style['spiker'] += 1
                    elif move['name'] in {'body slam', 'discharge', 'nuzzle'}:
                        style['paralysis'] += 1
                    elif move['name'] in {'lava plume', 'scald'}:
                        style['burn'] += 1
                    elif move['name'] in {'sludge bomb', 'malignant chain'}:
                        style['toxic'] += 1
                    ### End attack exceptions ###
                except:
                    pass
                if 'hidden' in move['name'].lower():
                    style['anti'] = f"anti_{move['typing']}"
                '''if move['typing'] not in pkmn['types']:
                    style['types'].append(move['typing'])'''
            else:
                ### move exceptions ###
                if move['name'].lower() in {'counter', 'mirror coat', 'metal burst'}:
                    stall_check = False
                    for m in pkmn['moves']:
                        try:
                            if ('regains' in m['effect'].lower() or 'heals' in m['effect'].lower()):
                                stall_check = True
                                #print(f"heals 2 {move['name']}")
                                break
                        except:
                            pass
                    if stall_check:
                        archetypes['stall'] += 1
                    else:
                        archetypes['offense'] += 1
                    style['anti_offense'] += 1
                    
                        #print(f"anti offense {move['name']}")
                elif move['name'] == 'defog':
                    style['defogger'] += 1
                elif move['name'] == 'spikes':
                    style['spiker'] += 1
                elif move['name'] == 'toxic spikes':
                    style['tspiker'] += 1
                elif move['name'] == 'stealth rock':
                    style['rocker'] += 1
                elif move['name'].lower() == 'seismic toss' or move['name'].lower() == 'night shade'or move['name'].lower() == 'toxic':
                    archetypes['stall'] += 1
                elif move['name'].lower() in {'taunt', 'trick', 'encore'}:
                    style['disruptor'] += 1
                    style['anti_stall'] += 1
                elif move['name'].lower() == 'gyro ball' or move['name'].lower() == 'heavy slam' or move['name'].lower() == 'low kick':
                    archetypes['offense'] += 1
                    damage_type['physical'] += 1
                elif move['name'].lower() == 'electro ball' or move['name'].lower() == 'grass knot':
                    archetypes['offense'] += 1
                    damage_type['special'] += 1
                elif move['name'].lower() == 'shell smash':
                    style['setup_physical'] += 1
                    style['setup_special'] += 1
                elif move['name'].lower() == 'substitute':
                    style['disruptor'] += 1
                elif move['name'].lower() == 'will-o-wisp':
                    offense_check = False
                    for m in pkmn['moves']:
                        if m['name'].lower() == 'hex':
                            offense_check = True
                            break
                    if offense_check:
                        archetypes['offense'] += 1
                    style['burn'] += 1
                elif move['name'] == 'thunder wave':
                    style['paralysis'] += 1
                elif move['name'].lower() == 'take heart':
                    style['setup_special'] += 1
                    damage_type['special'] += 1
                elif move['name'] == 'toxic':
                    style['toxic'] += 1
                elif move['name'] in {'healing wish', 'memento', 'lunar dance', 'wish'}:
                    style['support'] += 1
                elif move['name'] in {'light screen', 'reflect'}:
                    style['screens'] += 1
                elif move['name'] == 'trick room':
                    style['trick_room'] += 1
                ### end move exceptions ###

                elif 'lowers' in move['effect'] and 'target' in move['effect']:
                    if 'special attack' in move['effect'] or 'attack' in move['effect']:
                        archetypes['stall'] += 1
                    elif 'special' in move['effect']:
                        style['setup_special'] += 1
                        archetypes['offense'] += 1
                    else:
                        style['setup_physical'] += 1
                        archetypes['offense'] += 1
                        #print(f"lowers opp/ setup: {move['name']}")
                elif 'raises' in move['effect'] and 'user' in move['effect']:
                    #SETUP
                    #print(move['effect'])
                    move['effect'] = move['effect']
                    if 'special' in move['effect']:
                        archetypes['offense'] += 1
                        damage_type['special'] += 1
                        style['setup_special'] += 1
                        #print(f"setup special {move['name']}")
                    elif 'attack' in move['effect']:
                        archetypes['offense'] += 1
                        damage_type['physical'] += 1
                        style['setup_physical'] += 1
                        #print(f"setup phys {move['name']}")
                    elif 'defense' in move['effect']:
                        body_press_check = False
                        for m in pkmn['moves']:
                            if (m['name'].lower() == 'body press'):
                                body_press_check = True
                                break
                        if body_press_check:
                            archetypes['offense'] += 1
                            style['Bulky'] += 1
                            style['setup_pyshical'] += 1
                        else:
                            archetypes['stall'] += 1
                    elif 'special defense' in move['effect']:
                        archetypes['stall'] += 1
                    if 'speed' in move['effect']:
                        style['setup_speed'] += 1
                        #print(f"setup blank {move['name']}")
                elif ('regains' in move['effect'] or 'heals' in move['effect']):
                    archetypes['stall'] += 1
                    #print(f"heals {move['name']}")
                
                elif move['name'].lower() == 'endure':
                    style['anti_offense'] += 1
                elif 'switches the target out' in move['effect']:
                    style['anti_offense'] += 1
                else:
                    style['disruptor'] += 1
                    #print(f"disrupter {move['name']}")
        except Exception as e:
            print(f"Error on move: {move['name']}, {pkmn['name']}")
            print(e)
    try:
        # Important items #
        try:
            if pkmn['item'].lower() == 'custap berry' or pkmn['abilities'].lower() == "sturdy":
                style['anti_offense'] += 1

            elif ('Berry' in pkmn['item']) and (pkmn['item'] not in {'Sitrus Berry', 'Iapapa Berry', 'Kee Berry', 'Maranga Berry', 'Wiki Berry', 'Lum Berry', 'Liechi Berry', 'Petaya Berry'}):
                style['surprise'] += 1
            elif pkmn['item'] == 'Assault Vest':
                style['bulky'] += 1
            elif pkmn['item'] == 'Weakness Policy':
                style['surprise'] += 1
            elif 'Z' in pkmn['item']:
                style['z_move'] = pkmn['item']
            elif pkmn['item'] == 'Booster Energy':
                style['be'] += 1
            elif pkmn['item'] == 'Focus Sash':
                style['sash'] += 1
            elif pkmn['item'] == 'Leftovers' or pkmn['item'] == 'Black Sludge':
                style['lefties'] += 1
            elif pkmn['item'] == 'Heavy-Duty Boots':
                style['boots'] += 1
            elif pkmn['item'] == 'Eject Button':
                style['pivot'] += 1
            elif pkmn['item'] == 'Life Orb':
                style['life orb'] += 1
        except Exception as e:
            print(f"Item error: {e}")
        # End important items #
        try:
            # Weather check / abilities check #
            if pkmn['abilities'].lower() == 'drizzle':
                weather['rain'] = 1
                weather['rain_setter'] = 1
            elif pkmn['abilities'].lower() == 'sand stream':
                weather['sand'] = 1
                weather['sand_setter'] = 1
            elif pkmn['abilities'].lower() == 'drought':
                weather['sun'] = 1
                weather['sun_setter'] = 1
            elif pkmn['abilities'].lower() == 'snow warning':
                weather['snow'] = 1
                weather['snow_setter'] = 1
            elif pkmn['abilities'].lower() in weather_abilities[1]:
                weather['rain'] = 1
            elif pkmn['abilities'].lower() in weather_abilities[2]:
                weather['sand'] = 1
            elif pkmn['abilities'].lower() in weather_abilities[3]:
                weather['sun'] = 1
            elif pkmn['abilities'].lower() in weather_abilities[4]:
                weather['snow'] = 1
            # End weather abilities check #
            
            elif pkmn['abilities'].lower() == 'magnet pull': 
                style['trapper'] += 1
            elif pkmn['abilities'].lower() == 'regenerator':
                style['regen'] += 1
            elif pkmn['abilities'].lower() == 'levitate':
                style['levitate'] += 1
            elif pkmn['abilities'].lower() == 'unaware':
                style['unaware'] += 1
            elif pkmn['abilities'].lower() == 'magic guard':
                style['m_guard'] += 1
            elif pkmn['abilities'].lower() in {'volt absorb', 'motor drive', 'lightning rod'}:
                style['elec_immune'] += 1
            elif pkmn['abilities'].lower() in {'water absorb', 'storm drain', 'dry skin'}:
                style['water_immune'] += 1
            elif pkmn['abilities'].lower() in {'flash fire'}:
                style['fire_immune'] += 1
            elif pkmn['abilities'].lower() in {'purifying salt', 'poison heal', 'limber'}:
                style['status_absorber'] += 1

        except Exception as e:
            print(f"Ability error {e}")
            #print(move['name'])
            #print(archetypes)
        # important abilities #
    except:
        print(f"Error on {pkmn['item']}")
    if style['disruptor'] > archetypes['offense'] and archetypes['offense'] == 1:
        archetypes['stall'] += 1
    if archetypes['offense'] == archetypes['stall']:
        archetypes['stall'] += 1
    try:
        pkmn['evs'] = pkmn['evs'].lower()
    except:
        pass
    p_evs = pkmn['evs'].split('/')
    bulk = 0
    speed = 0
    for stat in p_evs:
        if 'spe' in stat:
            speed += int(stat.strip().split(' ')[0])
        elif 'hp' in stat or 'def' in stat or 'spd' in stat:
            bulk += int(stat.strip().split(' ')[0])
    if bulk > speed:
        style['Bulky'] += 1
        if speed > 200:
            style['fast'] += 1
    else:
        style['fast'] += 1
    # Define whether the mon is physically or specially bulky if it is bulky
    defense = 0
    sp_defense = 0
    for stat in p_evs:
        #Grab the defensive EVs
        if 'def' in stat:
            defense += int(stat.strip().split(' ')[0])
        elif 'spd' in stat or 'def' in stat or 'spd' in stat:
            sp_defense += int(stat.strip().split(' ')[0])
        ###
    # Compare the defense evs
    if (defense >= 100 and (defense - sp_defense) >= 50) or (defense - sp_defense) >= 80:
        style['bulky_physical'] += 1
    elif sp_defense >= 100 and (sp_defense - defense) >= 50 or (sp_defense - defense) >= 80:
        style['bulky_special'] += 1
    elif defense >= 68 and sp_defense >= 68:
        style['bulky_mixed'] += 1
    
    #Check Mixed Attacker


    '''if damage_type['physical'] == damage_type['special'] and damage_type['physical'] != 0:
        style['mixed'] += 1''' #old mixed
    if damage_type['physical'] > 0 and damage_type['special'] > 0:
        style['mixed'] += 1

    #print(f"{pkmn['name']}, {weather}, {style}\n")
    
    return archetypes, damage_type, style, weather


def make_label(mon):
    try:
        archetype, damage_class, style, weather = make_state(mon)
    except Exception as e:
        print(mon)
        print(e)
    archetype_encoding = 0
    dc_encoding = 0
    style_label = []
    #Archetype encoding
    if archetype['offense'] > archetype['stall']:
        archetype_encoding = 0
    else:
        archetype_encoding = 1
        
    #Damage class encoding
    if damage_class['physical'] > damage_class['special']:
        dc_encoding = 0
    else:
        dc_encoding = 1
    '''print("POOPY")
    print(style)'''

    #Style encoding
    for key, value in style.items():
        '''print(style_label)
        input()'''
        if key == 'types':
            for val in value:
                #style_label += val + " "
                pass
        elif key == 'anti' or key == 'z_move':
            #style_label += f'{value} '
            pass
        elif key == 'mixed':
            if value >= 1:
                dc_encoding = 2
        elif value > 0:
            style_label.append(1)
        else:
            style_label.append(0)

    for key, value in weather.items():
        style_label.append(value)

    '''print(archetype_encoding, dc_encoding, style_label)
    input()'''
    
    return archetype_encoding, dc_encoding, style_label


def get_type_encoding(mon):
    '''mon_types_1 = [mon['types'][0], mon['types'][1]]
    mon_types_2 = [mon['types'][1], mon['types'][0]]'''
    #print(mon_types_1)
    '''for i, types in enumerate(H.type_mapping):
        if mon_types_1 == types or mon_types_2 == types:
            return i'''
    return [H.types.index(mon['types'][0]), H.types.index(mon['types'][1])]

def flatten_mon_label(mon_array):
    flattened_label = []
    for element in mon_array:
        for val in element:
            flattened_label.append(val)

    return flattened_label

def label_teams(teams_data):
    #all_labels = set()
    labeled_teams = []  # Initialize as an empty list
    type_errors = set()
    label_errors = set()

    known_pokemon = []
    mon_counts = []

    mon_species = []

    for team in teams_data:
        for mon in team:
            if mon['name'] not in mon_species:
                mon_species.append(mon['name'])

    with open(U.species_number, 'w') as f:
        f.write(str(len(mon_species)))

    for index, team in enumerate(teams_data):
        t = []
        error = False
        for mon in team:
            try:
                arch, dc, style = make_label(mon) #Get mon state label
            except Exception as e:
                #print('label error')
                label_errors.add(mon['name'])
                error = True
                print(e)
                break
            try:
                if len(mon['types']) == 1: #Type observations
                    mon['types'].append('none')
                type_encoding = get_type_encoding(mon)
                species_label = mon_species.index(mon['name'])
                mon_label = flatten_mon_label([[species_label, arch, dc], style, type_encoding])
                t.append(mon_label) 

                mon['label'] = mon_label
                if mon not in known_pokemon:
                    mon_counts.append(1)
                    known_pokemon.append(mon)
                else:
                    mon_counts[known_pokemon.index(mon)] += 1
            except:
                error = True
                type_errors.add(mon['name'])
                break
        if error:
            continue
        labeled_teams.append(t) #Add each team as a new array to labeled_teams
        '''print(f"{t[0]}\n{t[1]}\n{t[2]}")
        input()'''
        if (index+1)%100 == 0:
            print(f"processed {index} teams...")

    #print(mon_counts)
    for m, c in zip(known_pokemon, mon_counts):
        m['count'] = c

    #labeled_teams = np.array(labeled_teams) #Convert into a numpy array
    '''print(len(labeled_teams))
    print(labeled_teams[0])'''
    labeled_teams = np.array(labeled_teams)

    ###DEBUG
    print(f"Type Errors: {type_errors}")
    print(f"Label Errors: {label_errors}")

    return labeled_teams, known_pokemon

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