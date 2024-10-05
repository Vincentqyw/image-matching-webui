import os
import argparse
import pickle
import time
from typing import Dict
import numpy as np
import requests
from loguru import logger

URL = "http://127.0.0.1:8001"
if "REMOTE_URL_RAILWAY" in os.environ:
    URL = os.environ["REMOTE_URL_RAILWAY"]

logger.info(f"API URL: {URL}")

API_URL_MATCH = f"{URL}/v1/match"
API_URL_EXTRACT = f"{URL}/v1/extract"
API_URL_EXTRACT_V2 = f"{URL}/v2/extract"


def send_generate_request(path0: str, path1: str) -> Dict[str, np.ndarray]:
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
        response = requests.post(API_URL_MATCH, files=files)
        pred = {}
        if response.status_code == 200:
            pred = response.json()
            for key in list(pred.keys()):
                pred[key] = np.array(pred[key])
        else:
            print(
                f"Error: Response code {response.status_code} - {response.text}"
            )
    finally:
        files["image0"].close()
        files["image1"].close()
    return pred


def send_generate_request1(path0: str) -> Dict[str, np.ndarray]:
    """
    Send a request to the API to extract features from an image.

    Args:
        path0 (str): The path to the image.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the extracted features.
            The keys are "keypoints", "descriptors", and "scores", and the
            values are ndarrays of shape (N, 2), (N, 128), and (N,),
            respectively.
    """
    files = {"image": open(path0, "rb")}
    try:
        response = requests.post(API_URL_EXTRACT, files=files)
        pred: Dict[str, np.ndarray] = {}
        if response.status_code == 200:
            pred = response.json()
            for key in list(pred.keys()):
                pred[key] = np.array(pred[key])
        else:
            print(
                f"Error: Response code {response.status_code} - {response.text}"
            )
    finally:
        files["image"].close()
    return pred


def send_generate_request2(image_path: str) -> Dict[str, np.ndarray]:
    """
    Send a request to the API to extract features from an image.

    Args:
        image_path (str): The path to the image.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the extracted features.
            The keys are "keypoints", "descriptors", and "scores", and the
            values are ndarrays of shape (N, 2), (N, 128), and (N,), respectively.
    """
    data = {
        "image_path": image_path,
        "max_keypoints": 1024,
        "reference_points": [[0.0, 0.0], [1.0, 1.0]],
    }
    pred = {}
    try:
        response = requests.post(API_URL_EXTRACT_V2, json=data)
        pred: Dict[str, np.ndarray] = {}
        if response.status_code == 200:
            pred = response.json()
            for key in list(pred.keys()):
                pred[key] = np.array(pred[key])
        else:
            print(
                f"Error: Response code {response.status_code} - {response.text}"
            )
    except Exception as e:
        print(f"An error occurred: {e}")
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send text to stable audio server and receive generated audio."
    )
    parser.add_argument(
        "--image0",
        required=False,
        help="Path for the file's melody",
        default="datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot45.jpg",
    )
    parser.add_argument(
        "--image1",
        required=False,
        help="Path for the file's melody",
        default="datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot90.jpg",
    )
    args = parser.parse_args()
    for i in range(10):
        t1 = time.time()
        preds = send_generate_request(args.image0, args.image1)
        t2 = time.time()
        logger.info(f"Time cost1: {(t2 - t1)} seconds")

    for i in range(10):
        t1 = time.time()
        preds = send_generate_request1(args.image0)
        t2 = time.time()
        logger.info(f"Time cost2: {(t2 - t1)} seconds")

    for i in range(10):
        t1 = time.time()
        preds = send_generate_request2(args.image0)
        t2 = time.time()
        logger.info(f"Time cost2: {(t2 - t1)} seconds")

    with open("preds.pkl", "wb") as f:
        pickle.dump(preds, f)
