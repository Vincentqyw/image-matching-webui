import argparse
import base64
import os
import pickle
import time
from typing import Dict, List

import cv2
import numpy as np
import requests

ENDPOINT = "http://127.0.0.1:8001"
if "REMOTE_URL_RAILWAY" in os.environ:
    ENDPOINT = os.environ["REMOTE_URL_RAILWAY"]

print(f"API ENDPOINT: {ENDPOINT}")

API_VERSION = f"{ENDPOINT}/version"
API_URL_MATCH = f"{ENDPOINT}/v1/match"
API_URL_EXTRACT = f"{ENDPOINT}/v1/extract"


def read_image(path: str) -> str:
    """
    Read an image from a file, encode it as a JPEG and then as a base64 string.

    Args:
        path (str): The path to the image to read.

    Returns:
        str: The base64 encoded image.
    """
    # Read the image from the file
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Encode the image as a png, NO COMPRESSION!!!
    retval, buffer = cv2.imencode(".png", img)

    # Encode the JPEG as a base64 string
    b64img = base64.b64encode(buffer).decode("utf-8")

    return b64img


def do_api_requests(url=API_URL_EXTRACT, **kwargs):
    """
    Helper function to send an API request to the image matching service.

    Args:
        url (str): The URL of the API endpoint to use. Defaults to the
            feature extraction endpoint.
        **kwargs: Additional keyword arguments to pass to the API.

    Returns:
        List[Dict[str, np.ndarray]]: A list of dictionaries containing the
            extracted features. The keys are "keypoints", "descriptors", and
            "scores", and the values are ndarrays of shape (N, 2), (N, ?),
            and (N,), respectively.
    """
    # Set up the request body
    reqbody = {
        # List of image data base64 encoded
        "data": [],
        # List of maximum number of keypoints to extract from each image
        "max_keypoints": [100, 100],
        # List of timestamps for each image (not used?)
        "timestamps": ["0", "1"],
        # Whether to convert the images to grayscale
        "grayscale": 0,
        # List of image height and width
        "image_hw": [[640, 480], [320, 240]],
        # Type of feature to extract
        "feature_type": 0,
        # List of rotation angles for each image
        "rotates": [0.0, 0.0],
        # List of scale factors for each image
        "scales": [1.0, 1.0],
        # List of reference points for each image (not used)
        "reference_points": [[640, 480], [320, 240]],
        # Whether to binarize the descriptors
        "binarize": True,
    }
    # Update the request body with the additional keyword arguments
    reqbody.update(kwargs)
    try:
        # Send the request
        r = requests.post(url, json=reqbody)
        if r.status_code == 200:
            # Return the response
            return r.json()
        else:
            # Print an error message if the response code is not 200
            print(f"Error: Response code {r.status_code} - {r.text}")
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"An error occurred: {e}")


def send_request_match(path0: str, path1: str) -> Dict[str, np.ndarray]:
    """
    Send a request to the API to generate a match between two images.

    Args:
        path0 (str): The path to the first image.
        path1 (str): The path to the second image.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the generated matches.
            The keys are "keypoints0", "keypoints1", "matches0", and "matches1",
            and the values are ndarrays of shape (N, 2), (N, 2), (N, 2), and
            (N, 2), respectively.
    """
    files = {"image0": open(path0, "rb"), "image1": open(path1, "rb")}
    try:
        # TODO: replace files with post json
        response = requests.post(API_URL_MATCH, files=files)
        pred = {}
        if response.status_code == 200:
            pred = response.json()
            for key in list(pred.keys()):
                pred[key] = np.array(pred[key])
        else:
            print(f"Error: Response code {response.status_code} - {response.text}")
    finally:
        files["image0"].close()
        files["image1"].close()
    return pred


def send_request_extract(
    input_images: str, viz: bool = False
) -> List[Dict[str, np.ndarray]]:
    """
    Send a request to the API to extract features from an image.

    Args:
        input_images (str): The path to the image.

    Returns:
        List[Dict[str, np.ndarray]]: A list of dictionaries containing the
            extracted features. The keys are "keypoints", "descriptors", and
            "scores", and the values are ndarrays of shape (N, 2), (N, 128),
            and (N,), respectively.
    """
    image_data = read_image(input_images)
    inputs = {
        "data": [image_data],
    }
    response = do_api_requests(
        url=API_URL_EXTRACT,
        **inputs,
    )
    # breakpoint()
    # print("Keypoints detected: {}".format(len(response[0]["keypoints"])))

    # draw matching, debug only
    if viz:
        from hloc.utils.viz import plot_keypoints
        from ui.viz import fig2im, plot_images

        kpts = np.array(response[0]["keypoints_orig"])
        if "image_orig" in response[0].keys():
            img_orig = np.array(["image_orig"])

            output_keypoints = plot_images([img_orig], titles="titles", dpi=300)
            plot_keypoints([kpts])
            output_keypoints = fig2im(output_keypoints)
            cv2.imwrite(
                "demo_match.jpg",
                output_keypoints[:, :, ::-1].copy(),  # RGB -> BGR
            )
    return response


def get_api_version():
    try:
        response = requests.get(API_VERSION).json()
        print("API VERSION: {}".format(response["version"]))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Send text to stable audio server and receive generated audio."
    )
    parser.add_argument(
        "--image0",
        required=False,
        help="Path for the file's melody",
        default=str(
            Path(__file__).parents[1]
            / "datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot45.jpg"
        ),
    )
    parser.add_argument(
        "--image1",
        required=False,
        help="Path for the file's melody",
        default=str(
            Path(__file__).parents[1]
            / "datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot90.jpg"
        ),
    )
    args = parser.parse_args()

    # get api version
    get_api_version()

    # request match
    # for i in range(10):
    #     t1 = time.time()
    #     preds = send_request_match(args.image0, args.image1)
    #     t2 = time.time()
    #     print(
    #         "Time cost1: {} seconds, matched: {}".format(
    #             (t2 - t1), len(preds["mmkeypoints0_orig"])
    #         )
    #     )

    # request extract
    for i in range(1000):
        t1 = time.time()
        preds = send_request_extract(args.image0)
        t2 = time.time()
        print(f"Time cost2: {(t2 - t1)} seconds")

    # dump preds
    with open("preds.pkl", "wb") as f:
        pickle.dump(preds, f)
