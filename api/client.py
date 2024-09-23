import argparse
import requests
import numpy as np
import cv2
import json
import pickle
import time
from loguru import logger

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:8001/v1/predict"


def send_generate_request(path0, path1):
    with open(path0, "rb") as f:
        file0 = f.read()

    with open(path1, "rb") as f:
        file1 = f.read()

    files = {
        "image0": ("image0", file0),
        "image1": ("image1", file1),
    }
    response = requests.post(API_URL, files=files)
    pred = {}
    if response.status_code == 200:
        response_json = response.json()
        pred = json.loads(response_json)
        for key in list(pred.keys()):
            pred[key] = np.array(pred[key])
    else:
        print(f"Error: Response code {response.status_code} - {response.text}")
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send text to stable audio server and receive generated audio."
    )
    parser.add_argument(
        "--image0",
        required=False,
        help="Path for the file's melody",
        default="../datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot45.jpg",
    )
    parser.add_argument(
        "--image1",
        required=False,
        help="Path for the file's melody",
        default="../datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot90.jpg",
    )
    args = parser.parse_args()
    for i in range(100):
        t1 = time.time()
        preds = send_generate_request(args.image0, args.image1)
        t2 = time.time()
        logger.info(f"Time cost: {(t2 - t1)} seconds")

    # with open("preds.pkl", "wb") as f:
    #     pickle.dump(preds, f)
