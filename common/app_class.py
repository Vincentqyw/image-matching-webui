import argparse
import numpy as np
import gradio as gr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from common.utils import (
    ransac_zoo,
    change_estimate_geom,
    load_config,
    get_matcher_zoo,
    run_matching,
    gen_examples,
    GRADIO_VERSION,
)

DESCRIPTION = """
# Image Matching WebUI
This Space demonstrates [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui) by vincent qin. Feel free to play with it, or duplicate to run image matching without a queue!
<br/>
🔎 For more details about supported local features and matchers, please refer to https://github.com/Vincentqyw/image-matching-webui

🚀 All algorithms run on CPU for inference, causing slow speeds and high latency. For faster inference, please download the [source code](https://github.com/Vincentqyw/image-matching-webui) for local deployment.

🐛 Your feedback is valuable to me. Please do not hesitate to report any bugs [here](https://github.com/Vincentqyw/image-matching-webui/issues).
"""


class ImageMatchingApp:
    def __init__(self, server_name="0.0.0.0", server_port=7860, **kwargs):
        self.server_name = server_name
        self.server_port = server_port
        self.config_path = kwargs.get(
            "config", Path(__file__).parent / "config.yaml"
        )
        self.cfg = load_config(self.config_path)
        self.matcher_zoo = get_matcher_zoo(self.cfg["matcher_zoo"])
        # self.ransac_zoo = get_ransac_zoo(self.cfg["ransac_zoo"])
        self.app = None
        self.init_interface()
        # print all the keys

    def init_interface(self):
        with gr.Blocks() as self.app:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(
                        str(Path(__file__).parent.parent / "assets/logo.webp"),
                        elem_id="logo-img",
                        show_label=False,
                        show_share_button=False,
                        show_download_button=False,
                    )
                with gr.Column(scale=3):
                    gr.Markdown(DESCRIPTION)
            with gr.Row(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        matcher_list = gr.Dropdown(
                            choices=list(self.matcher_zoo.keys()),
                            value="disk+lightglue",
                            label="Matching Model",
                            interactive=True,
                        )
                        match_image_src = gr.Radio(
                            (
                                ["upload", "webcam", "clipboard"]
                                if GRADIO_VERSION > "3"
                                else ["upload", "webcam", "canvas"]
                            ),
                            label="Image Source",
                            value="upload",
                        )
                    with gr.Row():
                        input_image0 = gr.Image(
                            label="Image 0",
                            type="numpy",
                            image_mode="RGB",
                            height=300 if GRADIO_VERSION > "3" else None,
                            interactive=True,
                        )
                        input_image1 = gr.Image(
                            label="Image 1",
                            type="numpy",
                            image_mode="RGB",
                            height=300 if GRADIO_VERSION > "3" else None,
                            interactive=True,
                        )

                    with gr.Row():
                        button_reset = gr.Button(value="Reset")
                        button_run = gr.Button(
                            value="Run Match", variant="primary"
                        )

                    with gr.Accordion("Advanced Setting", open=False):
                        with gr.Accordion("Matching Setting", open=True):
                            with gr.Row():
                                match_setting_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=1,
                                    step=0.001,
                                    label="Match thres.",
                                    value=0.1,
                                )
                                match_setting_max_features = gr.Slider(
                                    minimum=10,
                                    maximum=10000,
                                    step=10,
                                    label="Max features",
                                    value=1000,
                                )
                            # TODO: add line settings
                            with gr.Row():
                                detect_keypoints_threshold = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    step=0.001,
                                    label="Keypoint thres.",
                                    value=0.015,
                                )
                                detect_line_threshold = gr.Slider(
                                    minimum=0.1,
                                    maximum=1,
                                    step=0.01,
                                    label="Line thres.",
                                    value=0.2,
                                )
                            # matcher_lists = gr.Radio(
                            #     ["NN-mutual", "Dual-Softmax"],
                            #     label="Matcher mode",
                            #     value="NN-mutual",
                            # )
                        with gr.Accordion("RANSAC Setting", open=True):
                            with gr.Row(equal_height=False):
                                ransac_method = gr.Dropdown(
                                    choices=ransac_zoo.keys(),
                                    value=self.cfg["defaults"]["ransac_method"],
                                    label="RANSAC Method",
                                    interactive=True,
                                )
                            ransac_reproj_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=12,
                                step=0.01,
                                label="Ransac Reproj threshold",
                                value=8.0,
                            )
                            ransac_confidence = gr.Slider(
                                minimum=0.0,
                                maximum=1,
                                step=0.00001,
                                label="Ransac Confidence",
                                value=self.cfg["defaults"]["ransac_confidence"],
                            )
                            ransac_max_iter = gr.Slider(
                                minimum=0.0,
                                maximum=100000,
                                step=100,
                                label="Ransac Iterations",
                                value=self.cfg["defaults"]["ransac_max_iter"],
                            )

                        with gr.Accordion("Geometry Setting", open=False):
                            with gr.Row(equal_height=False):
                                choice_estimate_geom = gr.Radio(
                                    ["Fundamental", "Homography"],
                                    label="Reconstruct Geometry",
                                    value=self.cfg["defaults"][
                                        "setting_geometry"
                                    ],
                                )

                    # collect inputs
                    inputs = [
                        input_image0,
                        input_image1,
                        match_setting_threshold,
                        match_setting_max_features,
                        detect_keypoints_threshold,
                        matcher_list,
                        ransac_method,
                        ransac_reproj_threshold,
                        ransac_confidence,
                        ransac_max_iter,
                        choice_estimate_geom,
                        gr.State(self.matcher_zoo),
                    ]

                    # Add some examples
                    with gr.Row():
                        # Example inputs
                        gr.Examples(
                            examples=gen_examples(),
                            inputs=inputs,
                            outputs=[],
                            fn=run_matching,
                            cache_examples=False,
                            label=(
                                "Examples (click one of the images below to Run"
                                " Match)"
                            ),
                        )
                    with gr.Accordion("Open for More!", open=False):
                        gr.Markdown(
                            f"""
                            <h3>Supported Algorithms</h3>
                            {", ".join(self.matcher_zoo.keys())}
                            """
                        )

                with gr.Column():
                    output_keypoints = gr.Image(label="Keypoints", type="numpy")
                    output_matches_raw = gr.Image(
                        label="Raw Matches", type="numpy"
                    )
                    output_matches_ransac = gr.Image(
                        label="Ransac Matches", type="numpy"
                    )
                    with gr.Accordion(
                        "Open for More: Matches Statistics", open=False
                    ):
                        matches_result_info = gr.JSON(
                            label="Matches Statistics"
                        )
                        matcher_info = gr.JSON(label="Match info")

                    with gr.Accordion(
                        "Open for More: Warped Image", open=False
                    ):
                        output_wrapped = gr.Image(
                            label="Wrapped Pair", type="numpy"
                        )
                        with gr.Accordion(
                            "Open for More: Geometry info", open=False
                        ):
                            geometry_result = gr.JSON(
                                label="Reconstructed Geometry"
                            )

                # callbacks
                match_image_src.change(
                    fn=self.ui_change_imagebox,
                    inputs=match_image_src,
                    outputs=input_image0,
                )
                match_image_src.change(
                    fn=self.ui_change_imagebox,
                    inputs=match_image_src,
                    outputs=input_image1,
                )

                # collect outputs
                outputs = [
                    output_keypoints,
                    output_matches_raw,
                    output_matches_ransac,
                    matches_result_info,
                    matcher_info,
                    geometry_result,
                    output_wrapped,
                ]
                # button callbacks
                button_run.click(
                    fn=run_matching, inputs=inputs, outputs=outputs
                )

                # Reset images
                reset_outputs = [
                    input_image0,
                    input_image1,
                    match_setting_threshold,
                    match_setting_max_features,
                    detect_keypoints_threshold,
                    matcher_list,
                    input_image0,
                    input_image1,
                    match_image_src,
                    output_keypoints,
                    output_matches_raw,
                    output_matches_ransac,
                    matches_result_info,
                    matcher_info,
                    output_wrapped,
                    geometry_result,
                    ransac_method,
                    ransac_reproj_threshold,
                    ransac_confidence,
                    ransac_max_iter,
                    choice_estimate_geom,
                ]
                button_reset.click(
                    fn=self.ui_reset_state, inputs=inputs, outputs=reset_outputs
                )

                # estimate geo
                choice_estimate_geom.change(
                    fn=change_estimate_geom,
                    inputs=[
                        input_image0,
                        input_image1,
                        geometry_result,
                        choice_estimate_geom,
                    ],
                    outputs=[output_wrapped, geometry_result],
                )

    def run(self):
        self.app.queue().launch(
            server_name=self.server_name,
            server_port=self.server_port,
            share=False,
        )

    def ui_change_imagebox(self, choice):
        """
        Updates the image box with the given choice.

        Args:
            choice (list): The list of image sources to be displayed in the image box.

        Returns:
            dict: A dictionary containing the updated value, sources, and type for the image box.
        """
        ret_dict = {
            "value": None,  # The updated value of the image box
            "__type__": "update",  # The type of update for the image box
        }
        if GRADIO_VERSION > "3":
            return {
                **ret_dict,
                "sources": choice,  # The list of image sources to be displayed
            }
        else:
            return {
                **ret_dict,
                "source": choice,  # The list of image sources to be displayed
            }

    def ui_reset_state(
        self,
        *args: Any,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
        int,
        float,
        str,
        Dict[str, Any],
        Dict[str, Any],
        str,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Dict[str, Any],
        Dict[str, Any],
        Optional[np.ndarray],
        Dict[str, Any],
        str,
        int,
        float,
        int,
    ]:
        """
        Reset the state of the UI.

        Returns:
            tuple: A tuple containing the initial values for the UI state.
        """
        key: str = list(self.matcher_zoo.keys())[
            0
        ]  # Get the first key from matcher_zoo
        return (
            None,  # image0: Optional[np.ndarray]
            None,  # image1: Optional[np.ndarray]
            self.cfg["defaults"][
                "match_threshold"
            ],  # matching_threshold: float
            self.cfg["defaults"]["max_keypoints"],  # max_features: int
            self.cfg["defaults"][
                "keypoint_threshold"
            ],  # keypoint_threshold: float
            key,  # matcher: str
            self.ui_change_imagebox("upload"),  # input image0: Dict[str, Any]
            self.ui_change_imagebox("upload"),  # input image1: Dict[str, Any]
            "upload",  # match_image_src: str
            None,  # keypoints: Optional[np.ndarray]
            None,  # raw matches: Optional[np.ndarray]
            None,  # ransac matches: Optional[np.ndarray]
            {},  # matches result info: Dict[str, Any]
            {},  # matcher config: Dict[str, Any]
            None,  # warped image: Optional[np.ndarray]
            {},  # geometry result: Dict[str, Any]
            self.cfg["defaults"]["ransac_method"],  # ransac_method: str
            self.cfg["defaults"][
                "ransac_reproj_threshold"
            ],  # ransac_reproj_threshold: float
            self.cfg["defaults"][
                "ransac_confidence"
            ],  # ransac_confidence: float
            self.cfg["defaults"]["ransac_max_iter"],  # ransac_max_iter: int
            self.cfg["defaults"]["setting_geometry"],  # geometry: str
        )
