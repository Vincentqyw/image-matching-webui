import sys
import argparse
import numpy as np
import torch
import gradio as gr

from hloc import match_dense, match_features, matchers,extractors, extract_features
from hloc.utils.base_model import dynamic_load
from utils.plotting import draw_matches
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# @brief: Load model from configuration file
def get_model(match_conf):
    Model = dynamic_load(matchers, match_conf['model']['name'])
    model = Model(match_conf['model']).eval().to(device)
    return model

def get_feature_model(conf):
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    return model

matcher_zoo = {
    'loftr': {
        'config': match_dense.confs['loftr'],
        'model': get_model(match_dense.confs['loftr']),
        'dense': True
    },
    'topicfm': {
        'config': match_dense.confs['topicfm'],
        'model': get_model(match_dense.confs['topicfm']),
        'dense': True
    },
    'aspanformer': {
        'config': match_dense.confs['aspanformer'],
        'model': get_model(match_dense.confs['aspanformer']),
        'dense': True
    },
    'superglue': {
        'config': match_features.confs['superglue'],
        'config_feature': extract_features.confs['superpoint_max'],
        'model': get_model(match_features.confs['superglue']),
        'model_feature': get_feature_model(extract_features.confs['superpoint_max']),
        'dense': False
    },
    'd2net': {
        'config': match_features.confs['NN-mutual'],
        'config_feature': extract_features.confs['d2net-ss'],
        'model': get_model(match_features.confs['NN-mutual']),
        'model_feature': get_feature_model(extract_features.confs['d2net-ss']),
        'dense': False
    },
    'disk': {
        'config': match_features.confs['NN-mutual'],
        'config_feature': extract_features.confs['disk'],
        'model': get_model(match_features.confs['NN-mutual']),
        'model_feature': get_feature_model(extract_features.confs['disk']),
        'dense': False
    },
    'r2d2': {
        'config': match_features.confs['NN-mutual'],
        'config_feature': extract_features.confs['r2d2'],
        'model': get_model(match_features.confs['NN-mutual']),
        'model_feature': get_feature_model(extract_features.confs['r2d2']),
        'dense': False
    },
    'sift': {
        'config': match_features.confs['NN-mutual'],
        'config_feature': extract_features.confs['sift'],
        'model': get_model(match_features.confs['NN-mutual']),
        'model_feature': get_feature_model(extract_features.confs['sift']),
        'dense': False
    },
    'DKMv3': {
        'config': match_dense.confs['dkm'],
        'model': get_model(match_dense.confs['dkm']),
        'dense': True
    },
}

def run_matching(key, image0, image1):
    model = matcher_zoo[key]
    matcher = model['model']
    match_conf = model['config']
    if model['dense']:
        pred = match_dense.match_images(matcher,image0, image1, match_conf['preprocessing'])
    else:
        extractor = model['model_feature']
        extract_conf = model['config_feature']
        pred0 = extract_features.extract(extractor, image0, extract_conf['preprocessing'])
        pred1 = extract_features.extract(extractor, image1, extract_conf['preprocessing'])
        pred = match_features.match_images(matcher, pred0, pred1)

    mkpts0 = pred['keypoints0_orig']
    mkpts1 = pred['keypoints1_orig']
    img0 = pred['image0_orig']
    img1 = pred['image1_orig']

    if 'mconf' in pred.keys():
        mconf = pred['mconf']
    else:
        mconf = np.ones(len(mkpts0))
    fig = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi = 300)
    return fig, len(mkpts0)

def run(config):
    matcher_list = gr.Dropdown(choices=list(matcher_zoo.keys()))
    input_image0 = gr.Image(label="image 0", type="numpy")
    input_image1 = gr.Image(label="image 1", type="numpy")
    output_path = gr.Image(label="image", type="numpy")
    matches_info = gr.Textbox(label="matches info", type="text")

    demo = gr.Interface(
        fn=run_matching,
        inputs=[matcher_list, input_image0, input_image1],
        outputs=[output_path, matches_info],
        title="Image Matching Toolbox",
        description="Image Matching Toolbox",
        theme="compact"
    )
    demo.launch(share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    args = parser.parse_args()
    config = None
    run(config)