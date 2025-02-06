import HyperParameters as H

#gen_9 = []
gen_8 = []
gen_7 = []
gen_6 = []
gen_5 = ['latios', 'tyranitar', 'landorus-therian', 'excadrill', 'ferrothorn', 'keldeo', 'keldeo-resolute', 'reuniclus', 'alakazam', 'thundurus-therian', 'politoed', 'skarmory', 'gliscor', 'jirachi', 'garchomp', 'dragonite', 'starmie', 'rotom-wash', 'tentacruel', 'scizor', 'mamoswine', 'volcarona', 'terrakion', 'hippowdon', 'breloom', 'jellicent', 'magnezone', 'celebi', 'latias', 'clefable', 'heatran', 'kyurem-black', 'abomasnow', 'gastrodon', 'tornadus', 'slowbro', 'blissey', 'amoonguss', 'gyarados', 'forretress', 'bronzong', 'chansey', 'conkeldurr', 'mew', 'slowking', 'kyurem', 'salamence', 'seismitoad', 'xatu', 'aerodactyl', 'hydreigon', 'ninetales', 'cresselia', 'zapdos', 'ditto', 'lucario', 'victini', 'tangrowth', 'froslass', 'rotom', 'alomomola', 'cresselia', 'mienshao', 'toxicroak', 'gengar', 'milotic', 'moltres', 'haxorus', 'scolipede', 'mandibuzz', 'donphan', 'sableye', 'bisharp', 'weavile', 'magneton', 'raikou', 'quagsire', 'azelf']

def get_vr():
    if H.gen == '9':
        return gen_9
    elif H.gen == '8':
        return gen_8
    elif H.gen == '7':
        return gen_7
    elif H.gen == '6':
        return gen_6
    elif H.gen == '5':
        return gen_5