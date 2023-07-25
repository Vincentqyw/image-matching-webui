import argparse
import gradio as gr
import numpy as np
import cv2
from hloc import extract_features
from extra_utils.utils import (
    matcher_zoo,
    device,
    match_dense,
    match_features,
    get_model,
    get_feature_model,
    display_matches
)

def wrap_images(mkpts0, mkpts1, img0, img1, estimate_geom):
    h1, w1, d = img0.shape
    h2, w2, d = img1.shape
    result_matrix = None
    if estimate_geom=="Fundamental":
        # Estimate the fundamental matrix using RANSAC with 1000 iterations
        F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_LMEDS, 1.0, 0.99)
        # Rectify the images
        _, H1, H2 = cv2.stereoRectifyUncalibrated(mkpts0.reshape(-1, 2), mkpts1.reshape(-1, 2), F, imgSize=(w1, h1))
        rectified_image1 = cv2.warpPerspective(img0, H1, (w1, h1))
        rectified_image2 = cv2.warpPerspective(img1, H2, (w2, h2))
        result_matrix = F

    if estimate_geom=="Homography":
        # Estimate the homography matrix
        # Calculate the homography matrix using RANSAC
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC)
        rectified_image1 = img0
        # Warp the second image onto the first image using the calculated homography
        rectified_image2 = cv2.warpPerspective(img1, H, (img0.shape[1]+img1.shape[1], img0.shape[0]))
        result_matrix = H
    if result_matrix is not None:
        # Calculate the dimensions of the images
        height1, width1 = rectified_image1.shape[:2]
        height2, width2 = rectified_image2.shape[:2]

        # Calculate the maximum dimensions
        max_height = max(height1, height2)
        max_width = max(width1, width2)

        # Create black images with the maximum dimensions
        padded_image1 = np.zeros((max_height+100, max_width+100, 3), dtype=np.uint8)
        padded_image2 = np.zeros((max_height+100, max_width+100, 3), dtype=np.uint8)
        padded_image1[padded_image1==0] = 255
        padded_image2[padded_image2==0] = 255
        # Calculate the starting point to paste the original images
        start1 = (max_width - width1) // 2
        start2 = (max_width - width2) // 2

        # Paste the original images into the padded images
        padded_image1[:height1, start1:start1+width1] = rectified_image1
        padded_image2[:height2, start2:start2+width2] = rectified_image2


        # Create a visualization of the two images side by side
        result = np.hstack((padded_image1, padded_image2))
        dictionary = {
            'row1': result_matrix[0].tolist(),
            'row2': result_matrix[1].tolist(),
            'row3': result_matrix[2].tolist()
        }
    return result, dictionary


def run_matching(
    match_threshold, extract_max_keypoints, keypoint_threshold, key, image0, image1, estimate_geom
):
    # image0 and image1 is RGB mode
    if image0 is None or image1 is None:
        raise gr.Error("Error: No images found! Please upload two images.")

    model = matcher_zoo[key]
    match_conf = model["config"]
    # update match config
    match_conf["model"]["match_threshold"] = match_threshold
    match_conf["model"]["max_keypoints"] = extract_max_keypoints

    matcher = get_model(match_conf)
    if model["dense"]:
        pred = match_dense.match_images(
            matcher, image0, image1, match_conf["preprocessing"], device=device
        )
        del matcher
        extract_conf = None
    else:
        extract_conf = model["config_feature"]
        # update extract config
        extract_conf["model"]["max_keypoints"] = extract_max_keypoints
        extract_conf["model"]["keypoint_threshold"] = keypoint_threshold
        extractor = get_feature_model(extract_conf)
        pred0 = extract_features.extract(
            extractor, image0, extract_conf["preprocessing"]
        )
        pred1 = extract_features.extract(
            extractor, image1, extract_conf["preprocessing"]
        )
        pred = match_features.match_images(matcher, pred0, pred1)
        del extractor
    fig, num_inliers = display_matches(pred)
    mkpts0 = pred['keypoints0_orig']
    mkpts1 = pred['keypoints1_orig']
    img0 = pred['image0_orig']
    img1 = pred['image1_orig']
    del pred
    wrapped_images = np.array([1])
    geom_rec = [0,0,0]
    if estimate_geom!="No":
        wrapped_images, geom_rec = wrap_images(mkpts0, mkpts1, img0, img1, estimate_geom)
    return fig, {"matches number": num_inliers}, \
        {'match_conf': match_conf, 'extractor_conf': extract_conf}, wrapped_images, {estimate_geom:geom_rec}


def ui_change_imagebox(choice):
    return {"value": None, "source": choice, "__type__": "update"}

def change_estimate_geom(choice):
    pass
    # print("Selected choice:", choice)

def ui_reset_state(
    match_threshold, extract_max_keypoints, keypoint_threshold, key, image0, image1, estimate_geom, output_wrapped, geometry_result
):
    match_threshold = 0.2
    extract_max_keypoints = 1000
    keypoint_threshold = 0.015
    key = list(matcher_zoo.keys())[0]
    image0 = None
    image1 = None
    estimate_geom = None
    output_wrapped = None
    geometry_result = None
    return match_threshold, extract_max_keypoints, \
        keypoint_threshold, key, image0, image1, \
        {"value": None, "source": "upload", "__type__": "update"}, \
        {"value": None, "source": "upload", "__type__": "update"}, \
        "upload", None, {}, {}, estimate_geom, output_wrapped, geometry_result


def run(config):
    with gr.Blocks(
        theme=gr.themes.Monochrome(), css="footer {visibility: hidden}"
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
                        choices=list(matcher_zoo.keys()),
                        value="disk+lightglue",
                        label="Matching Model",
                        interactive=True,
                    )
                    match_image_src = gr.Radio(
                        ["upload", "webcam", "canvas"],
                        label="Image Source",
                        value="upload",
                    )
                    estimate_geom = gr.Radio(["No", "Fundamental", "Homography"],
                        label="Reconstruct Geometry",value="No")

                with gr.Row():
                    match_setting_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.001,
                        label="Match threshold",
                        value=0.1,
                    )
                    match_setting_max_features = gr.Slider(
                        minimum=10,
                        maximum=10000,
                        step=10,
                        label="Max number of features",
                        value=1000,
                    )
                # TODO: add line settings
                with gr.Row():
                    detect_keypoints_threshold = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.001,
                        label="Keypoint threshold",
                        value=0.015,
                    )
                    detect_line_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1,
                        step=0.01,
                        label="Line threshold",
                        value=0.2,
                    )
                    # matcher_lists = gr.Radio(
                    #     ["NN-mutual", "Dual-Softmax"],
                    #     label="Matcher mode",
                    #     value="NN-mutual",
                    # )
                with gr.Row():
                    input_image0 = gr.Image(
                        label="Image 0",
                        type="numpy",
                        interactive=True,
                        image_mode="RGB",
                    )
                    input_image1 = gr.Image(
                        label="Image 1",
                        type="numpy",
                        interactive=True,
                        image_mode="RGB",
                    )

                with gr.Row():
                    button_reset = gr.Button(label="Reset", value="Reset")
                    button_run = gr.Button(
                        label="Run Match", value="Run Match", variant="primary"
                    )

                with gr.Accordion("Open for More!", open=False):
                    gr.Markdown(
                        f"""
                        <h3>Supported Algorithms</h3>
                        {", ".join(matcher_zoo.keys())}
                        """
                    )
            with gr.Column():
                output_wrapped = gr.Image(
                    label="Wrapped Pair",
                    type="numpy"
                )
                geometry_result = gr.JSON(label="Reconstructed Geometry")
                # collect inputs
                inputs = [
                    match_setting_threshold,
                    match_setting_max_features,
                    detect_keypoints_threshold,
                    matcher_list,
                    input_image0,
                    input_image1,
                    estimate_geom,
                ]

                # Add some examples
                with gr.Row():
                    examples = [
                        [
                            0.1,
                            2000,
                            0.015,
                            "disk+lightglue",
                            "datasets/sacre_coeur/mapping/71295362_4051449754.jpg",
                            "datasets/sacre_coeur/mapping/93341989_396310999.jpg",
                        ],
                        [
                            0.1,
                            2000,
                            0.015,
                            "loftr",
                            "datasets/sacre_coeur/mapping/03903474_1471484089.jpg",
                            "datasets/sacre_coeur/mapping/02928139_3448003521.jpg",
                        ],
                        [
                            0.1,
                            2000,
                            0.015,
                            "disk",
                            "datasets/sacre_coeur/mapping/10265353_3838484249.jpg",
                            "datasets/sacre_coeur/mapping/51091044_3486849416.jpg",
                        ],
                        [
                            0.1,
                            2000,
                            0.015,
                            "topicfm",
                            "datasets/sacre_coeur/mapping/44120379_8371960244.jpg",
                            "datasets/sacre_coeur/mapping/93341989_396310999.jpg",
                        ],
                        [
                            0.1,
                            2000,
                            0.015,
                            "superpoint+superglue",
                            "datasets/sacre_coeur/mapping/17295357_9106075285.jpg",
                            "datasets/sacre_coeur/mapping/44120379_8371960244.jpg",
                        ],
                    ]
                    # Example inputs
                    gr.Examples(
                        examples=examples,
                        inputs=inputs,
                        outputs=[],
                        fn=run_matching,
                        cache_examples=False,
                        label="Examples (click one of the images below to Run Match)",
                    )

            with gr.Column():
                output_mkpts = gr.Image(label="Keypoints Matching", type="numpy")
                matches_result_info = gr.JSON(label="Matches Statistics")
                matcher_info = gr.JSON(label="Match info")

            # callbacks
            match_image_src.change(
                fn=ui_change_imagebox, inputs=match_image_src, outputs=input_image0
            )
            match_image_src.change(
                fn=ui_change_imagebox, inputs=match_image_src, outputs=input_image1
            )
            estimate_geom.change(fn=change_estimate_geom, 
                inputs=estimate_geom, 
                outputs=None)
            # collect outputs
            outputs = [
                output_mkpts,
                matches_result_info,
                matcher_info,
                output_wrapped,
                geometry_result,
            ]
            # button callbacks
            button_run.click(fn=run_matching, inputs=inputs, outputs=outputs)

            # Reset images
            reset_outputs = [
                match_setting_threshold,
                match_setting_max_features,
                detect_keypoints_threshold,
                matcher_list,
                input_image0,
                input_image1,
                input_image0,
                input_image1,
                match_image_src,
                output_mkpts,
                matches_result_info,
                matcher_info,
                estimate_geom,
                output_wrapped,
                geometry_result,
            ]
            button_reset.click(fn=ui_reset_state, inputs=inputs, outputs=reset_outputs)

    app.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="config.yaml", help="configuration file path"
    )
    args = parser.parse_args()
    config = None
    run(config)
