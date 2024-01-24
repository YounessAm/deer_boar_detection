import json
import os
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
from PIL import Image

HOME = os.getcwd()
# HOME = os.path.dirname(HOME)

last_train_id=13
YOLO_WEIGHTS_PATH = f"{HOME}/runs/detect/train{last_train_id}/weights/best.pt"

def count_occurrences(dict_labels, labels):
    """
    Count the occurrences of each class label in the detected labels.

    Parameters:
    - dict_labels (list): dictionary {"label" : label_index(int)}
    - labels (list): List of detected class labels.

    Returns:
    - dict: Dictionary mapping class labels to the number of occurrences.
    """
    occurrences = defaultdict(int)
    for label in labels:
        occurrences[int(label.item())] += 1
    
    dict_result={}
    for key, value in dict_labels.items():
        dict_result[key] = occurrences[value]
    
    return dict_result

def detect_animal(url, YOLO_WEIGHTS_PATH, confidence=0.6):
    """
    Detect animals in an image from a given URL using a pre-trained YOLO model.

    Parameters:
    - url (str): URL of the image to be analyzed.
    - YOLO_WEIGHTS_PATH (str): Path to the yolo(.bt) weight file.
    - confidence (float): Confidence threshold for detections.

    Returns:
    - list: List of dictionaries containing detection results.
    """
    # Define labels
    dict_labels = {'boar': 0, 'deer': 1}

    # Load a YOLO model
    model = YOLO(YOLO_WEIGHTS_PATH)

    # Run inference
    results = model.predict(source=url, conf=confidence, save=False, device = 'cpu')

    # Extract results
    vars_names = ['x1', 'y1', 'x2', 'y2', 'score', 'class_id']
    detection_results = []

    for result in results:
        detection_result = {
            'img_source': result.orig_img.tolist(),
            'img_annotated': result.plot().tolist(),
            'number_of_detections_by_class': count_occurrences(dict_labels, result.boxes.cls),
            'boxes': [{var: value for var, value in zip(vars_names, detection)} for detection in result.boxes.data.tolist()]
        }
        detection_results.append(detection_result)

    return detection_results


def plot_annotated_from_result(results):
    """
    Plot annotated images from the inference results.

    Parameters:
    - results (list): List of dictionaries containing detection results.
    """
    for result in results:
        annotated_img = result['img_annotated']

        im_array = np.array(annotated_img).astype(np.uint8)  # Convert to a BGR numpy array
        rgb_image = Image.fromarray(im_array[..., ::-1])  # Convert to an RGB PIL image
        rgb_image.show()  # Show the image

