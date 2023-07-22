import torch
import numpy as np
import cv2
from hloc import matchers, extractors
from hloc.utils.base_model import dynamic_load
from hloc import match_dense, match_features, extract_features
from .plotting import draw_matches, fig2im
from .visualize_util import plot_images, plot_color_line_matches

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(match_conf):
    Model = dynamic_load(matchers, match_conf["model"]["name"])
    model = Model(match_conf["model"]).eval().to(device)
    return model


def get_feature_model(conf):
    Model = dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)
    return model


def display_matches(pred: dict):
    img0 = pred["image0_orig"]
    img1 = pred["image1_orig"]

    num_inliers = 0
    if "keypoints0_orig" in pred.keys() and "keypoints1_orig" in pred.keys():
        mkpts0 = pred["keypoints0_orig"]
        mkpts1 = pred["keypoints1_orig"]
        num_inliers = len(mkpts0)
        if "mconf" in pred.keys():
            mconf = pred["mconf"]
        else:
            mconf = np.ones(len(mkpts0))
        fig_mkpts = draw_matches(
            mkpts0,
            mkpts1,
            img0,
            img1,
            mconf,
            dpi=300,
            titles=["Image 0 - matched keypoints", "Image 1 - matched keypoints"],
        )
        fig = fig_mkpts
    if "line0_orig" in pred.keys() and "line1_orig" in pred.keys():
        # lines
        mtlines0 = pred["line0_orig"]
        mtlines1 = pred["line1_orig"]
        num_inliers = len(mtlines0)
        fig_lines = plot_images(
            [img0.squeeze(), img1.squeeze()],
            ["Image 0 - matched lines", "Image 1 - matched lines"],
            dpi=300,
        )
        fig_lines = plot_color_line_matches([mtlines0, mtlines1], lw=2)
        fig_lines = fig2im(fig_lines)

        # keypoints
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]

        if mkpts0 is not None and mkpts1 is not None:
            num_inliers = len(mkpts0)
            if "mconf" in pred.keys():
                mconf = pred["mconf"]
            else:
                mconf = np.ones(len(mkpts0))
            fig_mkpts = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi=300)
            fig_lines = cv2.resize(fig_lines, (fig_mkpts.shape[1], fig_mkpts.shape[0]))
            fig = np.concatenate([fig_mkpts, fig_lines], axis=0)
        else:
            fig = fig_lines
    return fig, num_inliers


# Matchers collections
matcher_zoo = {
    "gluestick": {"config": match_dense.confs["gluestick"], "dense": True},
    "sold2": {"config": match_dense.confs["sold2"], "dense": True},
    # 'dedode-sparse': {
    #     'config': match_dense.confs['dedode_sparse'],
    #     'dense': True  # dense mode, we need 2 images
    # },
    "loftr": {"config": match_dense.confs["loftr"], "dense": True},
    "topicfm": {"config": match_dense.confs["topicfm"], "dense": True},
    "aspanformer": {"config": match_dense.confs["aspanformer"], "dense": True},
    "dedode": {
        "config": match_features.confs["Dual-Softmax"],
        "config_feature": extract_features.confs["dedode"],
        "dense": False,
    },
    "superpoint+superglue": {
        "config": match_features.confs["superglue"],
        "config_feature": extract_features.confs["superpoint_max"],
        "dense": False,
    },
    "superpoint+lightglue": {
        "config": match_features.confs["superpoint-lightglue"],
        "config_feature": extract_features.confs["superpoint_max"],
        "dense": False,
    },
    "disk": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["disk"],
        "dense": False,
    },
    "disk+dualsoftmax": {
        "config": match_features.confs["Dual-Softmax"],
        "config_feature": extract_features.confs["disk"],
        "dense": False,
    },
    "superpoint+dualsoftmax": {
        "config": match_features.confs["Dual-Softmax"],
        "config_feature": extract_features.confs["superpoint_max"],
        "dense": False,
    },
    "disk+lightglue": {
        "config": match_features.confs["disk-lightglue"],
        "config_feature": extract_features.confs["disk"],
        "dense": False,
    },
    "superpoint+mnn": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["superpoint_max"],
        "dense": False,
    },
    "sift+sgmnet": {
        "config": match_features.confs["sgmnet"],
        "config_feature": extract_features.confs["sift"],
        "dense": False,
    },
    "sosnet": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["sosnet"],
        "dense": False,
    },
    "hardnet": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["hardnet"],
        "dense": False,
    },
    "d2net": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["d2net-ss"],
        "dense": False,
    },
    "d2net-ms": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["d2net-ms"],
        "dense": False,
    },
    "alike": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["alike"],
        "dense": False,
    },
    "lanet": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["lanet"],
        "dense": False,
    },
    "r2d2": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["r2d2"],
        "dense": False,
    },
    "darkfeat": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["darkfeat"],
        "dense": False,
    },
    "sift": {
        "config": match_features.confs["NN-mutual"],
        "config_feature": extract_features.confs["sift"],
        "dense": False,
    },
    "roma": {"config": match_dense.confs["roma"], "dense": True},
    "DKMv3": {"config": match_dense.confs["dkm"], "dense": True},
}
