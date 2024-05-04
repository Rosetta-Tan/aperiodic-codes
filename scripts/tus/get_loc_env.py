from typing import List, Tuple
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from typing import List, Tuple, Dict
from src.helpers import *
import src.helpers as helpers
from config import init_map_v2f

if __name__ == '__main__':
    loc_envs = init_map_v2f
    for k, faces in loc_envs.items():
        faces = helpers.GeometryUtils.subdivide(faces)
        faces = helpers.GeometryUtils.subdivide(faces)
        loc_envs[k] = faces

    # fig, ax = plt.subplots(1, len(init_map_v2f))
    # for k, faces in loc_envs.items():
    #     if k == 11:
    #         Tiling.draw(faces, Tiling.uniq_vertices(faces), ax=ax[k-1], show=False)
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    faces = loc_envs[1]
    Tiling.draw(faces, Tiling.uniq_vertices(faces), ax=ax, show=True)


