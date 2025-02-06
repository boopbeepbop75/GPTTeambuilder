from pathlib import Path
import HyperParameters as H
import Utils as U

#Dataset location information 
PROJECT_DIR = Path(__file__).parent.parent
CLEAN_DATA_FOLDER = (PROJECT_DIR / 'data_clean').resolve()
RAW_DATA_FOLDER = (PROJECT_DIR / 'data_raw').resolve()
TEST_DATA_FOLDER = (PROJECT_DIR / 'data_test').resolve()
GENERATED_TEAMS = (PROJECT_DIR / 'generated_teams').resolve()
MODEL_FOLDER = (PROJECT_DIR / 'Model').resolve()

teams = (U.RAW_DATA_FOLDER / f'teams_gen_{H.gen}.txt').resolve()
processed_teams = (U.CLEAN_DATA_FOLDER / f'processed_teams{H.gen}.json').resolve()
cleaned_teams_loc = (U.CLEAN_DATA_FOLDER / f'cleaned_teams{H.gen}.json').resolve()
known_pokemon = (U.CLEAN_DATA_FOLDER / f'known_pokemon{H.gen}.json').resolve()
all_style_labels = (U.CLEAN_DATA_FOLDER / f'all_style_labels{H.gen}.json').resolve()
team_tensors = (U.CLEAN_DATA_FOLDER / f'team_tensors{H.gen}.pt').resolve()
generated_teams = (U.GENERATED_TEAMS / f'generated_teams{H.gen}.txt').resolve()
labels = (U.CLEAN_DATA_FOLDER / f'label_visualization{H.gen}.txt').resolve()
species_number = (U.CLEAN_DATA_FOLDER / f'species{H.gen}.txt').resolve()