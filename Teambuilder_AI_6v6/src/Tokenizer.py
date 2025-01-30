import Conversions

def tokenizer(known_pokemon):
    map = []
    for mon in known_pokemon:
        try:
            if mon['label'][0] not in map:
                map.append(mon['label'])
        except Exception as e:
            print(f"Token Error: {e}")
    return map