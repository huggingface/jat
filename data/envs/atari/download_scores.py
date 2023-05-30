import json

import requests
import yaml
from tqdm import tqdm
from uncertainties import ufloat
from uncertainties.core import Variable

scores = { # data from Agent 57 paper https://arxiv.org/abs/2003.13350
    "alien": {
        "human": 7127.7,
        "random": 227.8,
        "agent57": ufloat(297638.17, 37054.55),
        "r2d2": ufloat(464232.43, 7988.66),
        "muzero": 741812.63,
    },
    "amidar": {
        "human": 1719.5,
        "random": 5.8,
        "agent57": ufloat(29660.08, 880.39),
        "r2d2": ufloat(31331.37, 817.79),
        "muzero": 28634.39,
    },
    "assault": {
        "human": 742.0,
        "random": 222.4,
        "agent57": ufloat(67212.67, 6150.59),
        "r2d2": ufloat(110100.04, 346.06),
        "muzero": 143972.03,
    },
    "asterix": {
        "human": 8503.3,
        "random": 210.0,
        "agent57": ufloat(991384.42, 9493.32),
        "r2d2": ufloat(999354.03, 12.94),
        "muzero": 998425.0,
    },
    "asteroids": {
        "human": 47388.7,
        "random": 719.1,
        "agent57": ufloat(150854.61, 16116.72),
        "r2d2": ufloat(431072.45, 1799.13),
        "muzero": 6785558.64,
    },
    "atlantis": {
        "human": 29028.1,
        "random": 12850.0,
        "agent57": ufloat(1528841.76, 28282.53),
        "r2d2": ufloat(1660721.85, 14643.83),
        "muzero": 1674767.2,
    },
    "bankheist": {
        "human": 753.1,
        "random": 14.2,
        "agent57": ufloat(23071.50, 15834.73),
        "r2d2": ufloat(27117.85, 963.12),
        "muzero": 1278.98,
    },
    "battlezone": {
        "human": 37187.5,
        "random": 2360.0,
        "agent57": ufloat(934134.88, 38916.03),
        "r2d2": ufloat(992600.31, 1096.19),
        "muzero": 848623.0,
    },
    "beamrider": {
        "human": 16926.5,
        "random": 363.9,
        "agent57": ufloat(300509.80, 13075.35),
        "r2d2": ufloat(390603.06, 23304.09),
        "muzero": 4549993.53,
    },
    "berzerk": {
        "human": 2630.4,
        "random": 123.7,
        "agent57": ufloat(61507.83, 26539.54),
        "r2d2": ufloat(77725.62, 4556.93),
        "muzero": 85932.6,
    },
    "bowling": {
        "human": 160.7,
        "random": 23.1,
        "agent57": ufloat(251.18, 13.22),
        "r2d2": ufloat(161.77, 99.84),
        "muzero": 260.13,
    },
    "boxing": {
        "human": 12.1,
        "random": 0.1,
        "agent57": ufloat(100.00, 0.00),
        "r2d2": ufloat(100.00, 0.00),
        "muzero": 100.0,
    },
    "breakout": {
        "human": 30.5,
        "random": 1.7,
        "agent57": ufloat(790.40, 60.05),
        "r2d2": ufloat(863.92, 0.08),
        "muzero": 864.0,
    },
    "centipede": {
        "human": 12017.0,
        "random": 2090.9,
        "agent57": ufloat(412847.86, 26087.14),
        "r2d2": ufloat(908137.24, 7330.99),
        "muzero": 1159049.27,
    },
    "choppercommand": {
        "human": 7387.8,
        "random": 811.0,
        "agent57": ufloat(999900.00, 0.00),
        "r2d2": ufloat(999900.00, 0.00),
        "muzero": 991039.7,
    },
    "crazyclimber": {
        "human": 35829.4,
        "random": 10780.5,
        "agent57": ufloat(565909.85, 89183.85),
        "r2d2": ufloat(729482.83, 87975.74),
        "muzero": 458315.4,
    },
    "defender": {
        "human": 18688.9,
        "random": 2874.5,
        "agent57": ufloat(677642.78, 16858.59),
        "r2d2": ufloat(730714.53, 715.54),
        "muzero": 839642.95,
    },
    "demonattack": {
        "human": 1971.0,
        "random": 152.1,
        "agent57": ufloat(143161.44, 220.32),
        "r2d2": ufloat(143913.32, 92.93),
        "muzero": 143964.26,
    },
    "doubledunk": {
        "human": -16.4,
        "random": -18.6,
        "agent57": ufloat(23.93, 0.06),
        "r2d2": ufloat(24.00, 0.00),
        "muzero": 23.94,
    },
    "enduro": {
        "human": 860.5,
        "random": 0.0,
        "agent57": ufloat(2367.71, 8.69),
        "r2d2": ufloat(2378.66, 3.66),
        "muzero": 2382.44,
    },
    "fishingderby": {
        "human": -38.7,
        "random": -91.7,
        "agent57": ufloat(86.97, 3.25),
        "r2d2": ufloat(90.34, 2.66),
        "muzero": 91.16,
    },
    "freeway": {
        "human": 29.6,
        "random": 0.0,
        "agent57": ufloat(32.59, 0.71),
        "r2d2": ufloat(34.00, 0.00),
        "muzero": 33.03,
    },
    "frostbite": {
        "human": 4334.7,
        "random": 65.2,
        "agent57": ufloat(541280.88, 17485.76),
        "r2d2": ufloat(309077.30, 274879.03),
        "muzero": 631378.53,
    },
    "gopher": {
        "human": 2412.5,
        "random": 257.6,
        "agent57": ufloat(117777.08, 3108.06),
        "r2d2": ufloat(129736.13, 653.03),
        "muzero": 130345.58,
    },
    "gravitar": {
        "human": 3351.4,
        "random": 173.0,
        "agent57": ufloat(19213.96, 348.25),
        "r2d2": ufloat(21068.03, 497.25),
        "muzero": 6682.7,
    },
    "hero": {
        "human": 30826.4,
        "random": 1027.0,
        "agent57": ufloat(114736.26, 49116.60),
        "r2d2": ufloat(49339.62, 4617.76),
        "muzero": 49244.11,
    },
    "icehockey": {
        "human": 0.9,
        "random": -11.2,
        "agent57": ufloat(63.64, 6.48),
        "r2d2": ufloat(86.59, 0.59),
        "muzero": 67.04,
    },
    "jamesbond": {
        "human": 302.8,
        "random": 29.0,
        "agent57": ufloat(135784.96, 9132.28),
        "r2d2": ufloat(158142.36, 904.45),
        "muzero": 41063.25,
    },
    "kangaroo": {
        "human": 3035.0,
        "random": 52.0,
        "agent57": ufloat(24034.16, 12565.88),
        "r2d2": ufloat(18284.99, 817.25),
        "muzero": 16763.6,
    },
    "krull": {
        "human": 2665.5,
        "random": 1598.0,
        "agent57": ufloat(251997.31, 20274.39),
        "r2d2": ufloat(245315.44, 48249.07),
        "muzero": 269358.27,
    },
    "kungfumaster": {
        "human": 22736.3,
        "random": 258.5,
        "agent57": ufloat(206845.82, 11112.10),
        "r2d2": ufloat(267766.63, 2895.73),
        "muzero": 204824.0,
    },
    "montezumarevenge": {
        "human": 4753.3,
        "random": 0.0,
        "agent57": ufloat(9352.01, 2939.78),
        "r2d2": ufloat(3000.00, 0.00),
        "muzero": 0.0,
    },
    "mspacman": {
        "human": 6951.6,
        "random": 307.3,
        "agent57": ufloat(63994.44, 6652.16),
        "r2d2": ufloat(62595.90, 1755.82),
        "muzero": 243401.1,
    },
    "namethisgame": {
        "human": 8049.0,
        "random": 2292.3,
        "agent57": ufloat(54386.77, 6148.50),
        "r2d2": ufloat(138030.67, 5279.91),
        "muzero": 157177.85,
    },
    "phoenix": {
        "human": 7242.6,
        "random": 761.4,
        "agent57": ufloat(908264.15, 28978.92),
        "r2d2": ufloat(990638.12, 6278.77),
        "muzero": 955137.84,
    },
    "pitfall": {
        "human": 6463.7,
        "random": -229.4,
        "agent57": ufloat(18756.01, 9783.91),
        "r2d2": ufloat(0.00, 0.00),
        "muzero": 0.0,
    },
    "pong": {
        "human": 14.6,
        "random": -20.7,
        "agent57": ufloat(20.67, 0.47),
        "r2d2": ufloat(21.00, 0.00),
        "muzero": 21.0,
    },
    "privateeye": {
        "human": 69571.3,
        "random": 24.9,
        "agent57": ufloat(79716.46, 29515.48),
        "r2d2": ufloat(40700.00, 0.00),
        "muzero": 15299.98,
    },
    "qbert": {
        "human": 13455.0,
        "random": 163.9,
        "agent57": ufloat(580328.14, 151251.66),
        "r2d2": ufloat(777071.30, 190653.94),
        "muzero": 72276.0,
    },
    "riverraid": {
        "human": 17118.0,
        "random": 1338.5,
        "agent57": ufloat(63318.67, 5659.55),
        "r2d2": ufloat(93569.66, 13308.08),
        "muzero": 323417.18,
    },
    "roadrunner": {
        "human": 7845.0,
        "random": 11.5,
        "agent57": ufloat(243025.80, 79555.98),
        "r2d2": ufloat(593186.78, 88650.69),
        "muzero": 613411.8,
    },
    "robotank": {
        "human": 11.9,
        "random": 2.2,
        "agent57": ufloat(127.32, 12.50),
        "r2d2": ufloat(144.00, 0.00),
        "muzero": 131.13,
    },
    "seaquest": {
        "human": 42054.7,
        "random": 68.4,
        "agent57": ufloat(999997.63, 1.42),
        "r2d2": ufloat(999999.00, 0.00),
        "muzero": 999976.52,
    },
    "skiing": {
        "human": -4336.9,
        "random": -17098.1,
        "agent57": ufloat(-4202.60, 607.85),
        "r2d2": ufloat(-3851.44, 517.52),
        "muzero": -29968.36,
    },
    "solaris": {
        "human": 12326.7,
        "random": 1236.3,
        "agent57": ufloat(44199.93, 8055.50),
        "r2d2": ufloat(67306.29, 10378.22),
        "muzero": 56.62,
    },
    "spaceinvaders": {
        "human": 1668.7,
        "random": 148.0,
        "agent57": ufloat(48680.86, 5894.01),
        "r2d2": ufloat(67898.71, 1744.74),
        "muzero": 74335.3,
    },
    "stargunner": {
        "human": 10250.0,
        "random": 664.0,
        "agent57": ufloat(839573.53, 67132.17),
        "r2d2": ufloat(998600.28, 218.66),
        "muzero": 549271.7,
    },
    "surround": {
        "human": 6.5,
        "random": -10.0,
        "agent57": ufloat(9.50, 0.19),
        "r2d2": ufloat(10.00, 0.00),
        "muzero": 9.99,
    },
    "tennis": {
        "human": -8.3,
        "random": -23.8,
        "agent57": ufloat(23.84, 0.10),
        "r2d2": ufloat(24.00, 0.00),
        "muzero": 0.0,
    },
    "timepilot": {
        "human": 5229.2,
        "random": 3568.0,
        "agent57": ufloat(405425.31, 17044.45),
        "r2d2": ufloat(460596.49, 3139.33),
        "muzero": 476763.9,
    },
    "tutankham": {
        "human": 167.6,
        "random": 11.4,
        "agent57": ufloat(2354.91, 3421.43),
        "r2d2": ufloat(483.78, 37.90),
        "muzero": 491.48,
    },
    "upndown": {
        "human": 11693.2,
        "random": 533.4,
        "agent57": ufloat(623805.73, 23493.75),
        "r2d2": ufloat(702700.36, 8937.59),
        "muzero": 715545.61,
    },
    "venture": {
        "human": 1187.5,
        "random": 0.0,
        "agent57": ufloat(2623.71, 442.13),
        "r2d2": ufloat(2258.93, 29.90),
        "muzero": 0.4,
    },
    "videopinball": {
        "human": 17667.9,
        "random": 0.0,
        "agent57": ufloat(992340.74, 12867.87),
        "r2d2": ufloat(999645.92, 57.93),
        "muzero": 981791.88,
    },
    "wizardofwor": {
        "human": 4756.5,
        "random": 563.5,
        "agent57": ufloat(157306.41, 16000.00),
        "r2d2": ufloat(183090.81, 6070.10),
        "muzero": 197126.0,
    },
    "yarsrevenge": {
        "human": 54576.9,
        "random": 3092.9,
        "agent57": ufloat(998532.37, 375.82),
        "r2d2": ufloat(999807.02, 54.85),
        "muzero": 553311.46,
    },
    "zaxxon": {
        "human": 9173.3,
        "random": 32.5,
        "agent57": ufloat(249808.90, 58261.59),
        "r2d2": ufloat(370649.03, 19761.32),
        "muzero": 725853.9,
    },
}

env_id_patch = {
    "asteroids": "asteroid",
    "montezumarevenge": "montezuma",
    "privateeye": "privateye",
    "kungfumaster": "kongfumaster",
}

# Add expert data (from APPO training)
for env_id in tqdm(scores):
    # patch env_id
    if env_id in env_id_patch:
        url_env_id = env_id_patch[env_id]
    elif env_id == "surround": # FIXME: trained agent not available
        scores[env_id]["expert"] = ufloat(10.0, 0.0)
        continue
    else:
        url_env_id = env_id
    url = f"https://huggingface.co/edbeeching/atari_2B_atari_{url_env_id}_1111/raw/main/README.md"
    documents = requests.get(url).text.split("---\n")
    data = yaml.safe_load(documents[1])
    mean_reward = data["model-index"][0]["results"][0]["metrics"][0]["value"]
    mean, uncertainty = map(float, mean_reward.split(" +/- "))
    scores[env_id]["expert"] = ufloat(mean, uncertainty)


def custom_encoder(obj):
    if isinstance(obj, Variable):
        return {"__ufloat__": True, "n": obj.n, "s": obj.s}
    return obj


def custom_decoder(dct):
    if "__ufloat__" in dct:
        return ufloat(dct["n"], dct["s"])
    return dct


# Saving
with open("data/envs/atari/scores.json", "w") as f:
    json.dump(scores, f, default=custom_encoder, indent=4)


# Loading
with open("data/envs/atari/scores.json", "r") as f:
    scores = json.load(f, object_hook=custom_decoder)
