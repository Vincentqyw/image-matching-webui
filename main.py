import sys
import argparse
import numpy as np
import torch
import gradio as gr

from hloc import match_dense, match_features, matchers,extractors, extract_features
from hloc.utils.base_model import dynamic_load
from utils.plotting import draw_matches, fig2im, draw_image_pairs
from utils.visualize_util import plot_images, plot_lines, \
    plot_line_matches, plot_color_line_matches, plot_keypoints

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
    'sold2': {
        'config': match_dense.confs['sold2'],
        'model': get_model(match_dense.confs['sold2']),
        'dense': True
    },
    'gluestick': {
        'config': match_dense.confs['gluestick'],
        'model': get_model(match_dense.confs['gluestick']),
        'dense': True
    },
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
        pred = match_dense.match_images(matcher,image0, image1, match_conf['preprocessing'],device=device)
    else:
        extractor = model['model_feature']
        extract_conf = model['config_feature']
        pred0 = extract_features.extract(extractor, image0, extract_conf['preprocessing'])
        pred1 = extract_features.extract(extractor, image1, extract_conf['preprocessing'])
        pred = match_features.match_images(matcher, pred0, pred1)

    img0 = pred['image0_orig']
    img1 = pred['image1_orig']

    num_inliers = 0
    if 'keypoints0_orig' in pred.keys() and 'keypoints1_orig' in pred.keys():
        mkpts0 = pred['keypoints0_orig']
        mkpts1 = pred['keypoints1_orig']
        num_inliers = len(mkpts0)
        if 'mconf' in pred.keys():
            mconf = pred['mconf']
        else:
            mconf = np.ones(len(mkpts0))
        fig_mkpts = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi = 300)
        fig_lines = draw_image_pairs(img0, img1)
    if 'line0_orig' in pred.keys() and 'line1_orig' in pred.keys():
        mtlines0 = pred['line0_orig']
        mtlines1 = pred['line1_orig']
        lines0 = pred['line0']
        lines1 = pred['line1']
        num_inliers = len(mtlines0)

        # fig0 = plot_images([img0.squeeze(), img1.squeeze()], ['Image 1 - detected lines', 'Image 2 - detected lines'])
        # fig0 = plot_lines([lines0[:, :, ::-1], lines1[:, :, ::-1]], ps=3, lw=2)
        # fig0 = fig2im(fig0)
        fig_lines = plot_images([img0.squeeze(), img1.squeeze()], ['Image 1 - matched lines', 'Image 2 - matched lines'])
        fig_lines = plot_color_line_matches([mtlines0, mtlines1], lw=2)
        fig_lines = fig2im(fig_lines)

        # lines
        mkpts0 = pred['line_keypoints0_orig']
        mkpts1 = pred['line_keypoints1_orig']
        if 'mconf' in pred.keys():
            mconf = pred['mconf']
        else:
            mconf = np.ones(len(mkpts0))
        fig_mkpts = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi = 300)
    return fig_mkpts, fig_lines, num_inliers

def run(config):
    matcher_list = gr.Dropdown(choices=list(matcher_zoo.keys()))
    input_image0 = gr.Image(label="image 0", type="numpy")
    input_image1 = gr.Image(label="image 1", type="numpy")
    output_mkpts = gr.Image(label="image", type="numpy")
    output_lines = gr.Image(label="image", type="numpy")
    matches_info = gr.Textbox(label="matches info", type="text")

    demo = gr.Interface(
        fn=run_matching,
        inputs=[matcher_list, input_image0, input_image1],
        outputs=[output_mkpts, output_lines, matches_info],
        title="Image Matching Toolbox",
        description="Image Matching Toolbox",
        theme="compact"
    )
    demo.launch(share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    args = parser.parse_args()
    config = None
    run(config)