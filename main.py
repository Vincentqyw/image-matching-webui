import argparse
import numpy as np
import gradio as gr

from hloc import match_dense, match_features, extract_features
from utils.plotting import draw_matches, fig2im, draw_image_pairs
from utils.visualize_util import plot_images, plot_lines, \
    plot_line_matches, plot_color_line_matches, plot_keypoints
from utils.utils import matcher_zoo, device, match_dense, match_features,\
        get_model, get_feature_model   
 
# TODO: add model selection
def run_select_model(key):
    model = matcher_zoo[key]
    match_conf = model['config']
    matcher = get_model(match_conf)
    if not matcher['dense']:
        extract_conf = model['config_feature']
        local_feature_extractor = get_feature_model(extract_conf)
    return matcher, local_feature_extractor

def run_matching(in0, in1, in2, key, image0, image1):
    model = matcher_zoo[key]
    # matcher = model['model']
    match_conf = model['config']
    matcher = get_model(match_conf)
    if model['dense']:
        pred = match_dense.match_images(matcher,image0,\
            image1, match_conf['preprocessing'],device=device)
        del matcher
        extract_conf = None
    else:
        extract_conf = model['config_feature']
        extractor = get_feature_model(extract_conf)
        pred0 = extract_features.extract(extractor, \
                    image0, extract_conf['preprocessing'])
        pred1 = extract_features.extract(extractor, \
                    image1, extract_conf['preprocessing'])
        pred = match_features.match_images(matcher, pred0, pred1)
        del extractor
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
        # lines
        mtlines0 = pred['line0_orig']
        mtlines1 = pred['line1_orig']
        num_inliers = len(mtlines0)
        fig_lines = plot_images([img0.squeeze(), img1.squeeze()], \
                        ['Image 1 - matched lines', 'Image 2 - matched lines'],
                        pad=0)
        fig_lines = plot_color_line_matches([mtlines0, mtlines1], lw=2)
        fig_lines = fig2im(fig_lines)

        # keypoints
        mkpts0 = pred['line_keypoints0_orig']
        mkpts1 = pred['line_keypoints1_orig']
        if 'mconf' in pred.keys():
            mconf = pred['mconf']
        else:
            mconf = np.ones(len(mkpts0))
        fig_mkpts = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi = 300, pad = 0)
    del pred
    return fig_mkpts, fig_lines, {"matches number": num_inliers}, \
        {'match_conf': match_conf, 'extractor_conf': extract_conf}

def run(config):
    with gr.Blocks(theme=gr.themes.Monochrome(),
        css="footer {visibility: hidden}") as block:
        gr.Markdown("# Image Matching Toolbox")
        gr.HTML("<hr> Image matching toolbox webui is a web-based tool for image matching.\
                You can use it to match two images and visualize the results.")
        with gr.Row(equal_height=False):
            with gr.Column():
                matcher_list = gr.Dropdown(choices=list(matcher_zoo.keys()), \
                    value='topicfm', label="Select Model", interactive=True)
                with gr.Row():
                    match_setting_resize = gr.Slider(minimum=0.1, maximum=1, \
                            step=0.1, label="Resize ratio")
                    match_setting_max_num_features = gr.Slider(minimum=100, \
                            maximum=10000, step=100, label="Max number of features")
                match_setting_force_resize = gr.Checkbox(label="Force resize")
                
                input_image0 = gr.Image(label="Image 0", type="numpy")
                input_image1 = gr.Image(label="Image 1", type="numpy")
                with gr.Row():
                    button_clear = gr.Button(label="Clear",value="Clear")
                    button_run = gr.Button(label="Run Match", value="Run Match")
            with gr.Column():
                output_mkpts = gr.Image(label="Keypoints Matching", type="numpy")
                output_lines = gr.Image(label="Lines Matching", type="numpy")
                matches_result_info = gr.JSON(label="Matches Statistics")
                matcher_info = gr.JSON(label="Match info")

            # collect inputs and outputs
            inputs = [
                match_setting_resize,
                match_setting_max_num_features,
                match_setting_force_resize,
                matcher_list,
                input_image0,
                input_image1,
            ]
            outputs = [output_mkpts,
                         output_lines, 
                         matches_result_info, 
                         matcher_info,
            ]
            # button callbacks
            button_run.click(
                fn=run_matching,
                inputs=inputs,
                outputs=outputs
            )
    block.launch(share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    args = parser.parse_args()
    config = None
    run(config)