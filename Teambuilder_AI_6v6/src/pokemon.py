import requests
#from pokepy import V2Client
#from collections import MutableMapping

class Pokemon:
    def __init__(self, name, types):
        self.name = name
        self.types = types
        self.item = None
        self.ability = None
        self.evs = None
        self.nature = None
        self.moves = []

    def to_dict(self):
        return {
            "name": self.name,
            "item": self.item,
            "types": self.types,
            "abilities": self.ability,
            "evs": self.evs,
            "nature": self.nature,
            "moves": self.moves
        }

    def __str__(self):
        return f"Pokemon({self.name}, {self.item}, {self.types}, {self.ability}, {self.evs}, {self.nature}, {self.moves[0]['name']}, {self.moves[1]['name']}, {self.moves[2]['name']}, {self.moves[3]['name']})"
    
class Move:
    def __init__(self, name, formatted_name, bp, effect, damage_class, typing):
        self.name = name
        self.formatted_name = formatted_name
        self.bp = bp
        self.effect = effect
        self.damage_class = damage_class
        self.typing = typing

    def to_dict(self):
        return {
            "name": self.name,
            "formatted_name": self.formatted_name,
            "bp": self.bp,
            "effect": self.effect,
            "damage_class": self.damage_class,
            'typing': self.typing
        }

    def __str__(self):
        return f"Move({self.name}, {self.bp} bp; {self.effect})"

def get_pokemon(name):
    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        types = [t['type']['name'] for t in data['types']]
        #stats = {s['stat']['name']: s['base_stat'] for s in data['stats']}
        #abilities = [a['ability']['name'] for a in data['abilities']]
        return Pokemon(name=data['name'], types=types)

def get_move(name):
    formatted_name = name.replace(' ', '-').replace("'", "")
    url = f"https://pokeapi.co/api/v2/move/{formatted_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        #print(data)
        power = data['power']
        if power != None:
            power = int(power)
        else:
            power = 0
        try:
            effect = data['effect_entries'][0]['effect']
        except:
            effect = None
        try:
            damage_class = data['damage_class']['name']
        except: 
            pass
        try:
            typing = data['type']['name']
        except Exception as e:
            print(f"move type error: {e}")
        #abilities = [a['ability']['name'] for a in data['abilities']]
        return Move(name, formatted_name, power, effect, damage_class, typing)
    
#print(get_move('vine-whip'))

# Example usage
#pikachu = get_pokemon("regidrago")
#print(pikachu)

# Fetch a Pokémon
#pokemon = client.get_pokemon("charizard")