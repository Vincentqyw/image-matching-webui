import argparse
import gradio as gr
import numpy as np
import cv2
from hloc import extract_features
from extra_utils.plotting import draw_matches, fig2im
from extra_utils.utils import (
    matcher_zoo, device, match_dense, match_features,
    get_model, get_feature_model,
)
from extra_utils.visualize_util import plot_images, plot_color_line_matches

def run_matching(match_threshold, extract_max_keypoints, 
                 keypoint_threshold, key, image0, image1):
    if image0 is None or image1 is None:
        return np.zeros([2,2]), {"matches number": -1}, \
        {'match_conf': -1, 'extractor_conf': -1}
    
    model = matcher_zoo[key]
    match_conf = model['config']
    # update match config
    match_conf['model']['match_threshold'] = match_threshold
    match_conf['model']['max_keypoints'] = extract_max_keypoints

    matcher = get_model(match_conf)
    if model['dense']:
        pred = match_dense.match_images(
            matcher, image0, \
            image1, match_conf['preprocessing'],
            device=device
            )
        del matcher
        extract_conf = None
    else:
        extract_conf = model['config_feature']
        # update extract config
        extract_conf['model']['max_keypoints'] = extract_max_keypoints
        extract_conf['model']['keypoint_threshold'] = keypoint_threshold
        extractor = get_feature_model(extract_conf)
        pred0 = extract_features.extract(
            extractor, \
            image0, extract_conf['preprocessing']
            )
        pred1 = extract_features.extract(
            extractor, \
            image1, extract_conf['preprocessing']
            )
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
        fig_mkpts = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi=300, 
              titles=['Image 0 - matched keypoints',
                     'Image 1 - matched keypoints']
        )
        fig = fig_mkpts
    if 'line0_orig' in pred.keys() and 'line1_orig' in pred.keys():
        # lines
        mtlines0 = pred['line0_orig']
        mtlines1 = pred['line1_orig']
        num_inliers = len(mtlines0)
        fig_lines = plot_images(
            [img0.squeeze(), img1.squeeze()], \
            ['Image 0 - matched lines',
             'Image 1 - matched lines'], dpi=300
            )
        fig_lines = plot_color_line_matches([mtlines0, mtlines1], lw=2)
        fig_lines = fig2im(fig_lines)

        # keypoints
        mkpts0 = pred['line_keypoints0_orig']
        mkpts1 = pred['line_keypoints1_orig']
        if mkpts0 is not None and mkpts1 is not None:
            num_inliers = len(mkpts0)
            if 'mconf' in pred.keys():
                mconf = pred['mconf']
            else:
                mconf = np.ones(len(mkpts0))
            fig_mkpts = draw_matches(mkpts0, mkpts1, img0, img1, mconf, dpi=300)
            fig_lines = cv2.resize(fig_lines, (fig_mkpts.shape[1], fig_mkpts.shape[0]))
            fig = np.concatenate([fig_mkpts, fig_lines], axis=0)
        else:
            fig = fig_lines
    del pred
    return fig, {"matches number": num_inliers}, \
        {'match_conf': match_conf, 'extractor_conf': extract_conf}

def change_imagebox(choice):
    return {"value": None, "source": choice, "__type__": "update"}

def run(config):
    with gr.Blocks(
            theme=gr.themes.Monochrome(),
            css="footer {visibility: hidden}"
            ) as app:
        gr.Markdown(
            """
            <p align="center">
            <h1 align="center">Image Matching WebUI</h1> 
            </p>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    matcher_list = gr.Dropdown(
                        choices=list(matcher_zoo.keys()), \
                        value='topicfm',
                        label="Select Model",
                        interactive=True
                    )
                    match_image_src = gr.Radio(["upload", "webcam", "canvas"],
                        label="Image Source",value="upload")

                with gr.Row():
                    match_setting_threshold = gr.Slider(
                        minimum=0.1, maximum=1, \
                        step=0.01,
                        label="Match threshold",
                        value=0.2,
                    )
                    match_setting_max_num_features = gr.Slider(
                        minimum=100, \
                        maximum=10000,
                        step=100,
                        label="Max number of features",
                        value=1000,
                    )
                # TODO: add line settings
                with gr.Row():
                    detect_keypoints_threshold = gr.Slider(
                        minimum=0.1, maximum=1, \
                        step=0.01,
                        label="Keypoint threshold",
                        value=0.2,
                    )
                    detect_line_threshold = gr.Slider(
                        minimum=0.1, maximum=1, \
                        step=0.01,
                        label="Line threshold",
                        value=0.2,
                    )

                input_image0 = gr.Image(label="Image 0", type="numpy", interactive=True)
                input_image1 = gr.Image(label="Image 1", type="numpy", interactive=True)

                with gr.Row():
                    button_clear = gr.Button(label="Clear", value="Clear")
                    button_run = gr.Button(label="Run Match", value="Run Match")
                    button_clear.click(fn=change_imagebox, inputs=match_image_src, outputs=input_image0)
                    button_clear.click(fn=change_imagebox, inputs=match_image_src, outputs=input_image1)

                with gr.Accordion("Open for More!", open = False):
                    gr.Markdown(
                        f"""
                        <h3>Supported Algorithms</h3>
                        """
                    )
                    
    
            with gr.Column():
                output_mkpts = gr.Image(
                    label="Keypoints Matching",
                    type="numpy"
                )
                matches_result_info = gr.JSON(label="Matches Statistics")
                matcher_info = gr.JSON(label="Match info")
            

            # callbacks
            match_image_src.change(fn=change_imagebox, inputs=match_image_src, outputs=input_image0)
            match_image_src.change(fn=change_imagebox, inputs=match_image_src, outputs=input_image1)
            
            # collect inputs and outputs
            inputs = [
                match_setting_threshold,
                match_setting_max_num_features,
                detect_keypoints_threshold,
                matcher_list,
                input_image0,
                input_image1,
            ]
            outputs = [
                output_mkpts,
                matches_result_info,
                matcher_info,
            ]
            # button callbacks
            button_run.click(
                fn=run_matching,
                inputs=inputs,
                outputs=outputs
            )
    app.launch(share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', type=str,
        default='config.yaml', help='configuration file path'
        )
    args = parser.parse_args()
    config = None
    run(config)
