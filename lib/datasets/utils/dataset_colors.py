import random
from functools import lru_cache
import numpy as np

dataset_color_dict = {
    "potsdam" : [[1], [2], [3], [4], [5]],
    "vaihingen" : [[255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 255, 255]],
    "deepglobe" : [[255,255,255], [0,0,255], [0,255,0],[255,0,255], [255,255,0], [0,255,255]],
    "fbp" : [[i+1] for i in range(24)],
    "loveda" : [[i+2, i+2, i+2] for i in range(6)],
    "isaid" : [[i+1] for i in range(15)],
    "pastis-mm" : [[i+1] for i in range(18)],
    "dynamic-mm" : [[i] for i in range(7)], 
    "c2seg-ab" : [[i+1] for i in range(13)],
    "flood3i": [[i+1] for i in range(9)],
    "jl16-mm": [[i] for i in range(16)],
    "flair-mm": [[i+1] for i in range(18)],
    "dfc20": [[i+1] for i in range(10)]
}


modal_norm_dict = {
    'hr' : {
        'div' : 255.,
        'mean' : [0.485, 0.456, 0.406],
        'std' : [0.229, 0.224, 0.225]
    },
    'anno' : {
        'div' : 255.,
        'mean' : [0.485, 0.456, 0.406],
        'std' : [0.229, 0.224, 0.225]
    },
    's2' : {
        'div' : 1.,
        'mean' : [884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2368.51236873, 1805.06846033],
        'std' : [1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1455.52084939, 1343.48379601]
    },
    's1' : {
        'div' : 1.,
        'mean' : [-12.54847273, -20.19237134],
        'std' : [5.25697717, 5.91150917]
    },
}

@lru_cache()
def get_painter_color_map_list(num_locations = 300):
    
    num_sep_per_channel = int(num_locations ** (1 / 3)) + 1  # 19
    separation_per_channel = 256 // num_sep_per_channel

    color_list = []
    for location in range(num_locations):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel) and (num_seq_g <= num_sep_per_channel) \
               and (num_seq_b <= num_sep_per_channel)

        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (R < 256) and (G < 256) and (B < 256)
        assert (R >= 0) and (G >= 0) and (B >= 0)
        assert (R, G, B) not in color_list

        color_list.append((R, G, B))

    return color_list


def get_real_random_color_list(num_locations):
    random_color_list = np.random.randint(0, 256, (num_locations, 3))
    while np.sum(random_color_list) == 0:
        print('random_color_list is 0!')
        random_color_list = np.random.randint(0, 256, (num_locations, 3))
    random_color_list = random_color_list.tolist()
    return random_color_list # [:num_locations]
