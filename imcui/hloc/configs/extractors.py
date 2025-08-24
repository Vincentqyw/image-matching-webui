"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
"""

confs = {
    "superpoint_aachen": {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
            "keypoint_threshold": 0.005,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1600,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "output": "feats-superpoint-n4096-rmax1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
            "keypoint_threshold": 0.005,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1600,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "superpoint_inloc": {
        "output": "feats-superpoint-n4096-r1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
            "keypoint_threshold": 0.005,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "r2d2": {
        "output": "feats-r2d2-n5000-r1024",
        "model": {
            "name": "r2d2",
            "max_keypoints": 5000,
            "reliability_threshold": 0.7,
            "repetability_threshold": 0.7,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "d2net-ss": {
        "output": "feats-d2net-ss-n5000-r1600",
        "model": {
            "name": "d2net",
            "multiscale": False,
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "d2net-ms": {
        "output": "feats-d2net-ms-n5000-r1600",
        "model": {
            "name": "d2net",
            "multiscale": True,
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "rord": {
        "output": "feats-rord-ss-n5000-r1600",
        "model": {
            "name": "rord",
            "multiscale": False,
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "rootsift": {
        "output": "feats-rootsift-n5000-r1600",
        "model": {
            "name": "dog",
            "descriptor": "rootsift",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1600,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "sift": {
        "output": "feats-sift-n5000-r1600",
        "model": {
            "name": "sift",
            "rootsift": True,
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1600,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "sosnet": {
        "output": "feats-sosnet-n5000-r1600",
        "model": {
            "name": "dog",
            "descriptor": "sosnet",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "force_resize": True,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "hardnet": {
        "output": "feats-hardnet-n5000-r1600",
        "model": {
            "name": "dog",
            "descriptor": "hardnet",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "force_resize": True,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "disk": {
        "output": "feats-disk-n5000-r1600",
        "model": {
            "name": "disk",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "xfeat": {
        "output": "feats-xfeat-n5000-r1600",
        "model": {
            "name": "xfeat",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "liftfeat": {
        "output": "feats-liftfeat-n5000-r1600",
        "model": {
            "name": "liftfeat",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "rdd": {
        "output": "feats-rdd-n5000-r1600",
        "model": {
            "name": "rdd",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "ripe": {
        "output": "feats-ripe-n2048-r1600",
        "model": {
            "name": "ripe",
            "max_keypoints": 2048,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "aliked-n16-rot": {
        "output": "feats-aliked-n16-rot",
        "model": {
            "name": "aliked",
            "model_name": "aliked-n16rot",
            "max_num_keypoints": -1,
            "detection_threshold": 0.2,
            "nms_radius": 2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    "aliked-n16": {
        "output": "feats-aliked-n16",
        "model": {
            "name": "aliked",
            "model_name": "aliked-n16",
            "max_num_keypoints": -1,
            "detection_threshold": 0.2,
            "nms_radius": 2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    "alike": {
        "output": "feats-alike-n5000-r1600",
        "model": {
            "name": "alike",
            "max_keypoints": 5000,
            "use_relu": True,
            "multiscale": False,
            "detection_threshold": 0.5,
            "top_k": -1,
            "sub_pixel": False,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "lanet": {
        "output": "feats-lanet-n5000-r1600",
        "model": {
            "name": "lanet",
            "keypoint_threshold": 0.1,
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "darkfeat": {
        "output": "feats-darkfeat-n5000-r1600",
        "model": {
            "name": "darkfeat",
            "max_keypoints": 5000,
            "reliability_threshold": 0.7,
            "repetability_threshold": 0.7,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1600,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "dedode": {
        "output": "feats-dedode-n5000-r1600",
        "model": {
            "name": "dedode",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1600,
            "width": 768,
            "height": 768,
            "dfactor": 8,
        },
    },
    "example": {
        "output": "feats-example-n2000-r1024",
        "model": {
            "name": "example",
            "keypoint_threshold": 0.1,
            "max_keypoints": 2000,
            "model_name": "model.pth",
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 768,
            "height": 768,
            "dfactor": 8,
        },
    },
    "sfd2": {
        "output": "feats-sfd2-n4096-r1600",
        "model": {
            "name": "sfd2",
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1600,
            "width": 640,
            "height": 480,
            "conf_th": 0.001,
            "multiscale": False,
            "scales": [1.0],
        },
    },
    # Global descriptors
    "dir": {
        "output": "global-feats-dir",
        "model": {"name": "dir"},
        "preprocessing": {"resize_max": 1024},
    },
    "netvlad": {
        "output": "global-feats-netvlad",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 1024},
    },
    "openibl": {
        "output": "global-feats-openibl",
        "model": {"name": "openibl"},
        "preprocessing": {"resize_max": 1024},
    },
    "cosplace": {
        "output": "global-feats-cosplace",
        "model": {"name": "cosplace"},
        "preprocessing": {"resize_max": 1024},
    },
    "eigenplaces": {
        "output": "global-feats-eigenplaces",
        "model": {"name": "eigenplaces"},
        "preprocessing": {"resize_max": 1024},
    },
}
