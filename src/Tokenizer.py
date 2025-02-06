import Conversions

def tokenizer(known_pokemon):
    map = []
    duplicates = []
    
    for mon in known_pokemon:
        try:
            if mon['label'] not in map:
                map.append(mon['label'])
            else:
                # Track duplicates for debugging
                duplicates.append(mon['label'])
        except Exception as e:
            print(f"Token Error: {e}")
            print(f"Problematic mon: {mon}")
    
    # Print some debugging info
    print(f"Total unique labels: {len(map)}")
    print(f"Total duplicates found: {len(duplicates)}")
    return map