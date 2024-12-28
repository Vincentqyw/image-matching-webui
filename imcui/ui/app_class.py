import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parents[1]))

from ui.sfm import SfmEngine
from ui.utils import (
    GRADIO_VERSION,
    gen_examples,
    generate_warp_images,
    get_matcher_zoo,
    load_config,
    ransac_zoo,
    run_matching,
    run_ransac,
    send_to_match,
)

DESCRIPTION = """
# Image Matching WebUI
This Space demonstrates [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui) by vincent qin. Feel free to play with it, or duplicate to run image matching without a queue!
<br/>
🔎 For more details about supported local features and matchers, please refer to https://github.com/Vincentqyw/image-matching-webui

🚀 All algorithms run on CPU for inference, causing slow speeds and high latency. For faster inference, please download the [source code](https://github.com/Vincentqyw/image-matching-webui) for local deployment.

🐛 Your feedback is valuable to me. Please do not hesitate to report any bugs [here](https://github.com/Vincentqyw/image-matching-webui/issues).
"""

CSS = """
#warning {background-color: #FFCCCB}
.logs_class textarea {font-size: 12px !important}
"""


class ImageMatchingApp:
    def __init__(self, server_name="0.0.0.0", server_port=7860, **kwargs):
        self.server_name = server_name
        self.server_port = server_port
        self.config_path = kwargs.get("config", Path(__file__).parent / "config.yaml")
        self.cfg = load_config(self.config_path)
        self.matcher_zoo = get_matcher_zoo(self.cfg["matcher_zoo"])
        self.app = None
        self.init_interface()
        # print all the keys

    def init_matcher_dropdown(self):
        algos = []
        for k, v in self.cfg["matcher_zoo"].items():
            if v.get("enable", True):
                algos.append(k)
        return algos

    def init_interface(self):
        with gr.Blocks(css=CSS) as self.app:
            with gr.Tab("Image Matching"):
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
                                choices=self.init_matcher_dropdown(),
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
                            button_run = gr.Button(value="Run Match", variant="primary")

                        with gr.Accordion("Advanced Setting", open=False):
                            with gr.Accordion("Image Setting", open=True):
                                with gr.Row():
                                    image_force_resize_cb = gr.Checkbox(
                                        label="Force Resize",
                                        value=False,
                                        interactive=True,
                                    )
                                    image_setting_height = gr.Slider(
                                        minimum=48,
                                        maximum=2048,
                                        step=16,
                                        label="Image Height",
                                        value=480,
                                        visible=False,
                                    )
                                    image_setting_width = gr.Slider(
                                        minimum=64,
                                        maximum=2048,
                                        step=16,
                                        label="Image Width",
                                        value=640,
                                        visible=False,
                                    )
                            with gr.Accordion("Matching Setting", open=True):
                                with gr.Row():
                                    match_setting_threshold = gr.Slider(
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.001,
                                        label="Match threshold",
                                        value=0.1,
                                    )
                                    match_setting_max_keypoints = gr.Slider(
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
                                        label="Keypoint threshold",
                                        value=0.015,
                                    )
                                    detect_line_threshold = (  # noqa: F841
                                        gr.Slider(
                                            minimum=0.1,
                                            maximum=1,
                                            step=0.01,
                                            label="Line threshold",
                                            value=0.2,
                                        )
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
                                button_ransac = gr.Button(
                                    value="Rerun RANSAC", variant="primary"
                                )
                            with gr.Accordion("Geometry Setting", open=False):
                                with gr.Row(equal_height=False):
                                    choice_geometry_type = gr.Radio(
                                        ["Fundamental", "Homography"],
                                        label="Reconstruct Geometry",
                                        value=self.cfg["defaults"]["setting_geometry"],
                                    )
                        # image resize
                        image_force_resize_cb.select(
                            fn=self._on_select_force_resize,
                            inputs=image_force_resize_cb,
                            outputs=[image_setting_width, image_setting_height],
                        )
                        # collect inputs
                        state_cache = gr.State({})
                        inputs = [
                            input_image0,
                            input_image1,
                            match_setting_threshold,
                            match_setting_max_keypoints,
                            detect_keypoints_threshold,
                            matcher_list,
                            ransac_method,
                            ransac_reproj_threshold,
                            ransac_confidence,
                            ransac_max_iter,
                            choice_geometry_type,
                            gr.State(self.matcher_zoo),
                            image_force_resize_cb,
                            image_setting_width,
                            image_setting_height,
                        ]

                        # Add some examples
                        with gr.Row():
                            # Example inputs
                            with gr.Accordion("Open for More: Examples", open=True):
                                gr.Examples(
                                    examples=gen_examples(),
                                    inputs=inputs,
                                    outputs=[],
                                    fn=run_matching,
                                    cache_examples=False,
                                    label=(
                                        "Examples (click one of the images below to Run"
                                        " Match). Thx: WxBS"
                                    ),
                                )
                        with gr.Accordion("Supported Algorithms", open=False):
                            # add a table of supported algorithms
                            self.display_supported_algorithms()

                    with gr.Column():
                        with gr.Accordion("Open for More: Keypoints", open=True):
                            output_keypoints = gr.Image(label="Keypoints", type="numpy")
                        with gr.Accordion(
                            (
                                "Open for More: Raw Matches"
                                " (Green for good matches, Red for bad)"
                            ),
                            open=False,
                        ):
                            output_matches_raw = gr.Image(
                                label="Raw Matches",
                                type="numpy",
                            )
                        with gr.Accordion(
                            (
                                "Open for More: Ransac Matches"
                                " (Green for good matches, Red for bad)"
                            ),
                            open=True,
                        ):
                            output_matches_ransac = gr.Image(
                                label="Ransac Matches", type="numpy"
                            )
                        with gr.Accordion(
                            "Open for More: Matches Statistics", open=False
                        ):
                            output_pred = gr.File(label="Outputs", elem_id="download")
                            matches_result_info = gr.JSON(label="Matches Statistics")
                            matcher_info = gr.JSON(label="Match info")

                        with gr.Accordion("Open for More: Warped Image", open=True):
                            output_wrapped = gr.Image(
                                label="Wrapped Pair", type="numpy"
                            )
                            # send to input
                            button_rerun = gr.Button(
                                value="Send to Input Match Pair",
                                variant="primary",
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
                        state_cache,
                        output_pred,
                    ]
                    # button callbacks
                    button_run.click(fn=run_matching, inputs=inputs, outputs=outputs)
                    # Reset images
                    reset_outputs = [
                        input_image0,
                        input_image1,
                        match_setting_threshold,
                        match_setting_max_keypoints,
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
                        choice_geometry_type,
                        output_pred,
                        image_force_resize_cb,
                    ]
                    button_reset.click(
                        fn=self.ui_reset_state,
                        inputs=None,
                        outputs=reset_outputs,
                    )

                    # run ransac button action
                    button_ransac.click(
                        fn=run_ransac,
                        inputs=[
                            state_cache,
                            choice_geometry_type,
                            ransac_method,
                            ransac_reproj_threshold,
                            ransac_confidence,
                            ransac_max_iter,
                        ],
                        outputs=[
                            output_matches_ransac,
                            matches_result_info,
                            output_wrapped,
                            output_pred,
                        ],
                    )

                    # send warped image to match
                    button_rerun.click(
                        fn=send_to_match,
                        inputs=[state_cache],
                        outputs=[input_image0, input_image1],
                    )

                    # estimate geo
                    choice_geometry_type.change(
                        fn=generate_warp_images,
                        inputs=[
                            input_image0,
                            input_image1,
                            geometry_result,
                            choice_geometry_type,
                        ],
                        outputs=[output_wrapped, geometry_result],
                    )
            with gr.Tab("Structure from Motion(under-dev)"):
                sfm_ui = AppSfmUI(  # noqa: F841
                    {
                        **self.cfg,
                        "matcher_zoo": self.matcher_zoo,
                        "outputs": "experiments/sfm",
                    }
                )
                sfm_ui.call_empty()

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

    def _on_select_force_resize(self, visible: bool = False):
        return gr.update(visible=visible), gr.update(visible=visible)

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
        bool,
    ]:
        """
        Reset the state of the UI.

        Returns:
            tuple: A tuple containing the initial values for the UI state.
        """
        key: str = list(self.matcher_zoo.keys())[
            0
        ]  # Get the first key from matcher_zoo
        # flush_logs()
        return (
            None,  # image0: Optional[np.ndarray]
            None,  # image1: Optional[np.ndarray]
            self.cfg["defaults"]["match_threshold"],  # matching_threshold: float
            self.cfg["defaults"]["max_keypoints"],  # max_keypoints: int
            self.cfg["defaults"]["keypoint_threshold"],  # keypoint_threshold: float
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
            self.cfg["defaults"]["ransac_confidence"],  # ransac_confidence: float
            self.cfg["defaults"]["ransac_max_iter"],  # ransac_max_iter: int
            self.cfg["defaults"]["setting_geometry"],  # geometry: str
            None,  # predictions
            False,
        )

    def display_supported_algorithms(self, style="tab"):
        def get_link(link, tag="Link"):
            return "[{}]({})".format(tag, link) if link is not None else "None"

        data = []
        cfg = self.cfg["matcher_zoo"]
        if style == "md":
            markdown_table = "| Algo. | Conference | Code | Project | Paper |\n"
            markdown_table += "| ----- | ---------- | ---- | ------- | ----- |\n"

            for k, v in cfg.items():
                if not v["info"]["display"]:
                    continue
                github_link = get_link(v["info"]["github"])
                project_link = get_link(v["info"]["project"])
                paper_link = get_link(
                    v["info"]["paper"],
                    (
                        Path(v["info"]["paper"]).name[-10:]
                        if v["info"]["paper"] is not None
                        else "Link"
                    ),
                )

                markdown_table += "{}|{}|{}|{}|{}\n".format(
                    v["info"]["name"],  # display name
                    v["info"]["source"],
                    github_link,
                    project_link,
                    paper_link,
                )
            return gr.Markdown(markdown_table)
        elif style == "tab":
            for k, v in cfg.items():
                if not v["info"].get("display", True):
                    continue
                data.append(
                    [
                        v["info"]["name"],
                        v["info"]["source"],
                        v["info"]["github"],
                        v["info"]["paper"],
                        v["info"]["project"],
                    ]
                )
            tab = gr.Dataframe(
                headers=["Algo.", "Conference", "Code", "Paper", "Project"],
                datatype=["str", "str", "str", "str", "str"],
                col_count=(5, "fixed"),
                value=data,
                # wrap=True,
                # min_width = 1000,
                # height=1000,
            )
            return tab


class AppBaseUI:
    def __init__(self, cfg: Dict[str, Any] = {}):
        self.cfg = OmegaConf.create(cfg)
        self.inputs = edict({})
        self.outputs = edict({})
        self.ui = edict({})

    def _init_ui(self):
        NotImplemented

    def call(self, **kwargs):
        NotImplemented

    def info(self):
        gr.Info("SFM is under construction.")


class AppSfmUI(AppBaseUI):
    def __init__(self, cfg: Dict[str, Any] = None):
        super().__init__(cfg)
        assert "matcher_zoo" in self.cfg
        self.matcher_zoo = self.cfg["matcher_zoo"]
        self.sfm_engine = SfmEngine(cfg)
        self._init_ui()

    def init_retrieval_dropdown(self):
        algos = []
        for k, v in self.cfg["retrieval_zoo"].items():
            if v.get("enable", True):
                algos.append(k)
        return algos

    def _update_options(self, option):
        if option == "sparse":
            return gr.Textbox("sparse", visible=True)
        elif option == "dense":
            return gr.Textbox("dense", visible=True)
        else:
            return gr.Textbox("not set", visible=True)

    def _on_select_custom_params(self, value: bool = False):
        return gr.update(visible=value)

    def _init_ui(self):
        with gr.Row():
            # data settting and camera settings
            with gr.Column():
                self.inputs.input_images = gr.File(
                    label="SfM",
                    interactive=True,
                    file_count="multiple",
                    min_width=300,
                )
                # camera setting
                with gr.Accordion("Camera Settings", open=True):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                self.inputs.camera_model = gr.Dropdown(
                                    choices=[
                                        "PINHOLE",
                                        "SIMPLE_RADIAL",
                                        "OPENCV",
                                    ],
                                    value="PINHOLE",
                                    label="Camera Model",
                                    interactive=True,
                                )
                            with gr.Column():
                                gr.Checkbox(
                                    label="Shared Params",
                                    value=True,
                                    interactive=True,
                                )
                                camera_custom_params_cb = gr.Checkbox(
                                    label="Custom Params",
                                    value=False,
                                    interactive=True,
                                )
                        with gr.Row():
                            self.inputs.camera_params = gr.Textbox(
                                label="Camera Params",
                                value="0,0,0,0",
                                interactive=False,
                                visible=False,
                            )
                        camera_custom_params_cb.select(
                            fn=self._on_select_custom_params,
                            inputs=camera_custom_params_cb,
                            outputs=self.inputs.camera_params,
                        )

                with gr.Accordion("Matching Settings", open=True):
                    # feature extraction and matching setting
                    with gr.Row():
                        # matcher setting
                        self.inputs.matcher_key = gr.Dropdown(
                            choices=self.matcher_zoo.keys(),
                            value="disk+lightglue",
                            label="Matching Model",
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Column():
                                with gr.Row():
                                    # matching setting
                                    self.inputs.max_keypoints = gr.Slider(
                                        label="Max Keypoints",
                                        minimum=100,
                                        maximum=10000,
                                        value=1000,
                                        interactive=True,
                                    )
                                    self.inputs.keypoint_threshold = gr.Slider(
                                        label="Keypoint Threshold",
                                        minimum=0,
                                        maximum=1,
                                        value=0.01,
                                    )
                                with gr.Row():
                                    self.inputs.match_threshold = gr.Slider(
                                        label="Match Threshold",
                                        minimum=0.01,
                                        maximum=12.0,
                                        value=0.2,
                                    )
                                    self.inputs.ransac_threshold = gr.Slider(
                                        label="Ransac Threshold",
                                        minimum=0.01,
                                        maximum=12.0,
                                        value=4.0,
                                        step=0.01,
                                        interactive=True,
                                    )

                                with gr.Row():
                                    self.inputs.ransac_confidence = gr.Slider(
                                        label="Ransac Confidence",
                                        minimum=0.01,
                                        maximum=1.0,
                                        value=0.9999,
                                        step=0.0001,
                                        interactive=True,
                                    )
                                    self.inputs.ransac_max_iter = gr.Slider(
                                        label="Ransac Max Iter",
                                        minimum=1,
                                        maximum=100,
                                        value=100,
                                        step=1,
                                        interactive=True,
                                    )
                with gr.Accordion("Scene Graph Settings", open=True):
                    # mapping setting
                    self.inputs.scene_graph = gr.Dropdown(
                        choices=["all", "swin", "oneref"],
                        value="all",
                        label="Scene Graph",
                        interactive=True,
                    )

                    # global feature setting
                    self.inputs.global_feature = gr.Dropdown(
                        choices=self.init_retrieval_dropdown(),
                        value="netvlad",
                        label="Global features",
                        interactive=True,
                    )
                    self.inputs.top_k = gr.Slider(
                        label="Number of Images per Image to Match",
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1,
                    )
                # button_match = gr.Button("Run Matching", variant="primary")

            # mapping setting
            with gr.Column():
                with gr.Accordion("Mapping Settings", open=True):
                    with gr.Row():
                        with gr.Accordion("Buddle Settings", open=True):
                            with gr.Row():
                                self.inputs.mapper_refine_focal_length = gr.Checkbox(
                                    label="Refine Focal Length",
                                    value=False,
                                    interactive=True,
                                )
                                self.inputs.mapper_refine_principle_points = (
                                    gr.Checkbox(
                                        label="Refine Principle Points",
                                        value=False,
                                        interactive=True,
                                    )
                                )
                                self.inputs.mapper_refine_extra_params = gr.Checkbox(
                                    label="Refine Extra Params",
                                    value=False,
                                    interactive=True,
                                )
                    with gr.Accordion("Retriangluation Settings", open=True):
                        gr.Textbox(
                            label="Retriangluation Details",
                        )
                    self.ui.button_sfm = gr.Button("Run SFM", variant="primary")
                self.outputs.model_3d = gr.Model3D(
                    interactive=True,
                )
                self.outputs.output_image = gr.Image(
                    label="SFM Visualize",
                    type="numpy",
                    image_mode="RGB",
                    interactive=False,
                )

    def call_empty(self):
        self.ui.button_sfm.click(fn=self.info, inputs=[], outputs=[])

    def call(self):
        self.ui.button_sfm.click(
            fn=self.sfm_engine.call,
            inputs=[
                self.inputs.matcher_key,
                self.inputs.input_images,  # images
                self.inputs.camera_model,
                self.inputs.camera_params,
                self.inputs.max_keypoints,
                self.inputs.keypoint_threshold,
                self.inputs.match_threshold,
                self.inputs.ransac_threshold,
                self.inputs.ransac_confidence,
                self.inputs.ransac_max_iter,
                self.inputs.scene_graph,
                self.inputs.global_feature,
                self.inputs.top_k,
                self.inputs.mapper_refine_focal_length,
                self.inputs.mapper_refine_principle_points,
                self.inputs.mapper_refine_extra_params,
            ],
            outputs=[self.outputs.model_3d, self.outputs.output_image],
        )
