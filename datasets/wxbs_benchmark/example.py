import os
from pathlib import Path

ROOT_PATH = Path("/teamspace/studios/this_studio/image-matching-webui")
prefix = "datasets/wxbs_benchmark/.WxBS/v1.1"
wxbs_path = ROOT_PATH / prefix

pairs = []
for catg in os.listdir(wxbs_path):
    catg_path = wxbs_path / catg
    if not catg_path.is_dir():
        continue
    for scene in os.listdir(catg_path):
        scene_path = catg_path / scene
        if not scene_path.is_dir():
            continue
        img1_path = scene_path / "01.png"
        img2_path = scene_path / "02.png"
        pairs.append([str(img1_path), str(img2_path)])
