import json

import requests
import yaml
from tqdm import tqdm
from uncertainties import ufloat
from gia.utils import ufloat_decoder, ufloat_encoder


with open("data/envs/metaworld/scores.json", "r") as f:
    scores = json.load(f, object_hook=ufloat_decoder)
# patch
scores = {env_id: {"random": score} for env_id, score in scores.items()}

# Add expert data (from APPO training)
for env_id in tqdm(scores):
    url = f"https://huggingface.co/qgallouedec/sample-factory-{env_id}-v2/raw/main/README.md"
    documents = requests.get(url).text.split("---\n")
    data = yaml.safe_load(documents[1])
    mean_reward = data["model-index"][0]["results"][0]["metrics"][0]["value"]
    mean, uncertainty = map(float, mean_reward.split(" +/- "))
    scores[env_id]["expert"] = ufloat(mean, uncertainty)


# Saving
with open("data/envs/metaworld/scores.json", "w") as f:
    json.dump(scores, f, default=ufloat_encoder, indent=4)


# Loading
with open("data/envs/metaworld/scores.json", "r") as f:
    json.load(f, object_hook=ufloat_decoder)
