"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""

confs = {
    # sparse matching methods below
    "superglue": {
        "output": "matches-superglue",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "superglue-fast": {
        "output": "matches-superglue-it5",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 5,
            "match_threshold": 0.2,
        },
    },
    "superpoint-lightglue": {
        "output": "matches-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "superpoint",
            "model_name": "superpoint_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "disk-lightglue": {
        "output": "matches-disk-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "disk",
            "model_name": "disk_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "aliked-lightglue": {
        "output": "matches-aliked-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "aliked",
            "model_name": "aliked_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "sift-lightglue": {
        "output": "matches-sift-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "sift",
            "add_scale_ori": True,
            "model_name": "sift_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "sgmnet": {
        "output": "matches-sgmnet",
        "model": {
            "name": "sgmnet",
            "seed_top_k": [256, 256],
            "seed_radius_coe": 0.01,
            "net_channels": 128,
            "layer_num": 9,
            "head": 4,
            "seedlayer": [0, 6],
            "use_mc_seeding": True,
            "use_score_encoding": False,
            "conf_bar": [1.11, 0.1],
            "sink_iter": [10, 100],
            "detach_iter": 1000000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "NN-superpoint": {
        "output": "matches-NN-mutual-dist.7",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
            "match_threshold": 0.2,
        },
    },
    "NN-ratio": {
        "output": "matches-NN-mutual-ratio.8",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
            "match_threshold": 0.2,
        },
    },
    "NN-mutual": {
        "output": "matches-NN-mutual",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "match_threshold": 0.2,
        },
    },
    "Dual-Softmax": {
        "output": "matches-Dual-Softmax",
        "model": {
            "name": "dual_softmax",
            "match_threshold": 0.01,
            "inv_temperature": 20,
        },
    },
    "adalam": {
        "output": "matches-adalam",
        "model": {
            "name": "adalam",
            "match_threshold": 0.2,
        },
    },
    "imp": {
        "output": "matches-imp",
        "model": {
            "name": "imp",
            "match_threshold": 0.2,
        },
    },
    # dense matching methods below
    # Best quality but loads of points. Only use for small scenes
    "loftr": {
        "output": "matches-loftr",
        "model": {
            "name": "loftr",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "minima_loftr": {
        "output": "matches-minima_loftr",
        "model": {
            "name": "loftr",
            "weights": "outdoor",
            "model_name": "minima_loftr.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": False,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "eloftr": {
        "output": "matches-eloftr",
        "model": {
            "name": "eloftr",
            "model_name": "eloftr_outdoor.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 32,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "xoftr": {
        "output": "matches-xoftr",
        "model": {
            "name": "xoftr",
            "weights": "weights_xoftr_640.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.3,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "jamma": {
        "output": "matches-jamma",
        "model": {
            "name": "jamma",
            "weights": "jamma_weight.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.3,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 16,
            "width": 832,
            "height": 832,
            "force_resize": True,
        },
    },
    # "loftr_quadtree": {
    #     "output": "matches-loftr-quadtree",
    #     "model": {
    #         "name": "quadtree",
    #         "weights": "outdoor",
    #         "max_keypoints": 2000,
    #         "match_threshold": 0.2,
    #     },
    #     "preprocessing": {
    #         "grayscale": True,
    #         "resize_max": 1024,
    #         "dfactor": 8,
    #         "width": 640,
    #         "height": 480,
    #         "force_resize": True,
    #     },
    #     "max_error": 1,  # max error for assigned keypoints (in px)
    #     "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    # },
    "cotr": {
        "output": "matches-cotr",
        "model": {
            "name": "cotr",
            "weights": "out/default",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    # Semi-scalable loftr which limits detected keypoints
    "loftr_aachen": {
        "output": "matches-loftr_aachen",
        "model": {
            "name": "loftr",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {"grayscale": True, "resize_max": 1024, "dfactor": 8},
        "max_error": 2,  # max error for assigned keypoints (in px)
        "cell_size": 8,  # size of quantization patch (max 1 kp/patch)
    },
    # Use for matching superpoint feats with loftr
    "loftr_superpoint": {
        "output": "matches-loftr_aachen",
        "model": {
            "name": "loftr",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 4,  # max error for assigned keypoints (in px)
        "cell_size": 4,  # size of quantization patch (max 1 kp/patch)
    },
    # Use topicfm for matching feats
    "topicfm": {
        "output": "matches-topicfm",
        "model": {
            "name": "topicfm",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
        },
    },
    # Use aspanformer for matching feats
    "aspanformer": {
        "output": "matches-aspanformer",
        "model": {
            "name": "aspanformer",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "duster": {
        "output": "matches-duster",
        "model": {
            "name": "duster",
            "weights": "vit_large",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 512,
            "dfactor": 16,
        },
    },
    "mast3r": {
        "output": "matches-mast3r",
        "model": {
            "name": "mast3r",
            "weights": "vit_large",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 512,
            "dfactor": 16,
        },
    },
    "xfeat_lightglue": {
        "output": "matches-xfeat_lightglue",
        "model": {
            "name": "xfeat_lightglue",
            "max_keypoints": 8000,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": False,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "xfeat_dense": {
        "output": "matches-xfeat_dense",
        "model": {
            "name": "xfeat_dense",
            "max_keypoints": 8000,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": False,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "dkm": {
        "output": "matches-dkm",
        "model": {
            "name": "dkm",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 80,
            "height": 60,
            "dfactor": 8,
        },
    },
    "roma": {
        "output": "matches-roma",
        "model": {
            "name": "roma",
            "weights": "outdoor",
            "model_name": "roma_outdoor.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "dad_roma": {
        "output": "matches-dad_roma",
        "model": {
            "name": "dad_roma",
            "weights": "outdoor",
            "model_name": "roma_outdoor.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "gim_roma": {
        "output": "matches-gim_roma",
        "model": {
            "name": "roma",
            "model_name": "gim_roma_100h.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "rdd_dense": {
        "output": "matches-rdd_dense",
        "model": {
            "name": "rdd_dense",
            "model_name": "RDD-v2.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "minima_roma": {
        "output": "matches-minima_roma",
        "model": {
            "name": "roma",
            "weights": "outdoor",
            "model_name": "minima_roma.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "gim(dkm)": {
        "output": "matches-gim",
        "model": {
            "name": "gim",
            "model_name": "gim_dkm_100h.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "omniglue": {
        "output": "matches-omniglue",
        "model": {
            "name": "omniglue",
            "match_threshold": 0.2,
            "max_keypoints": 2000,
            "features": "null",
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
            "width": 640,
            "height": 480,
        },
    },
    "sold2": {
        "output": "matches-sold2",
        "model": {
            "name": "sold2",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "gluestick": {
        "output": "matches-gluestick",
        "model": {
            "name": "gluestick",
            "use_lines": True,
            "max_keypoints": 1000,
            "max_lines": 300,
            "force_num_keypoints": False,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
}
