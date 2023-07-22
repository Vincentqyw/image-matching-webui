import argparse
import gradio as gr

from hloc import extract_features
from extra_utils.utils import (
    matcher_zoo,
    device,
    match_dense,
    match_features,
    get_model,
    get_feature_model,
)

# from extra_utils.plotting import draw_matches, fig2im
# from extra_utils.visualize_util import plot_images, plot_color_line_matches
from extra_utils.utils import display_matches

description = "<p style='text-align: center'>\
    Optical flow and stereo matching demo for <a href='https://haofeixu.github.io/unimatch/' \
        target='_blank'>Unifying Flow, Stereo and Depth Estimation</a> | <a href='https://arxiv.org/abs/2211.05783' target='_blank'>Paper</a> | <a href='https://github.com/autonomousvision/unimatch' target='_blank'>Code</a> | <a href='https://colab.research.google.com/drive/1r5m-xVy3Kw60U-m5VB-aQ98oqqg_6cab?usp=sharing' target='_blank'>Colab</a><br>Task <strong>flow</strong>: Image1: <strong>video frame t</strong>, Image2: <strong>video frame t+1</strong>; Task <strong>stereo</strong>: Image1: <strong>left</strong> image, Image2: <strong>right</strong> image<br>Simply upload your images or click one of the provided examples.<br><strong>Select the task type according to your input images</strong>.</p>"


def run_matching(
    match_threshold, extract_max_keypoints, keypoint_threshold, key, image0, image1
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
    del pred
    return (
        fig,
        {"matches number": num_inliers},
        {"match_conf": match_conf, "extractor_conf": extract_conf},
    )


def ui_change_imagebox(choice):
    return {"value": None, "source": choice, "__type__": "update"}


def ui_reset_state(
    match_threshold, extract_max_keypoints, keypoint_threshold, key, image0, image1
):
    match_threshold = 0.2
    extract_max_keypoints = 1000
    keypoint_threshold = 0.015
    key = list(matcher_zoo.keys())[0]
    image0 = None
    image1 = None
    return (
        match_threshold,
        extract_max_keypoints,
        keypoint_threshold,
        key,
        image0,
        image1,
        {"value": None, "source": "upload", "__type__": "update"},
        {"value": None, "source": "upload", "__type__": "update"},
        "upload",
        None,
        {},
        {},
    )


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
                        value=list(matcher_zoo.keys())[0],
                        label="Matching Model",
                        interactive=True,
                    )
                    match_image_src = gr.Radio(
                        ["upload", "webcam", "canvas"],
                        label="Image Source",
                        value="upload",
                    )

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

                # collect inputs
                inputs = [
                    match_setting_threshold,
                    match_setting_max_features,
                    detect_keypoints_threshold,
                    matcher_list,
                    input_image0,
                    input_image1,
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

            # collect outputs
            outputs = [
                output_mkpts,
                matches_result_info,
                matcher_info,
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
