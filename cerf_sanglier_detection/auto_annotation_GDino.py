from tqdm import tqdm
import os
import sys
HOME = os.getcwd()
HOME = os.path.dirname(HOME)
sys.path.append(f'{HOME}/GroundingDINO')

from groundingdino.util.inference import load_model, load_image, predict
from setup_grounding_dino import CONFIG_PATH, WEIGHTS_PATH
from download_images import IMG_FOLDER
from utils import get_image_names


def annotate(model, image_name, img_folder, class_name, box_threshold=0.35, text_threshold=0.25):
    """
    Annotate an image with bounding boxes and return the result.

    Parameters:
    - model: GroundingDino model.
    - image_name (str): Name of the image file.
    - img_folder (str): Path to the folder containing images.
    - class_name (str): Class name for annotation.
    - box_threshold (float): Bounding box threshold.
    - text_threshold (float): Text threshold.

    Returns:
    - tuple: Tuple containing bounding boxes and phrases.
    """
    image_path = os.path.join(img_folder, image_name)

    _, image = load_image(image_path)

    boxes, _, phrases = predict(
        model=model,
        image=image,
        caption=class_name,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes, phrases


def transform_to_yolo_annotation(phrases, boxes, dict_labels):
    """
    Transform GroundingDino annotation to YOLO format.

    Parameters:
    - phrases (list): List of predicted phrases.
    - boxes (list): List of predicted bounding boxes.
    - dict_labels (dict): Dictionary mapping class names to YOLO indices.

    Returns:
    - str: YOLO-formatted annotation.
    """
    annotations = []
    for i, classe in enumerate(phrases):
        annotation = f'{dict_labels[classe]} ' + ' '.join(map(str, boxes[i].tolist()))
        annotations.append(annotation)
    return '\n'.join(annotations)


def save_annotation(yolo_annotation, file_path):
    """
    Save YOLO-formatted annotation to a file.

    Parameters:
    - yolo_annotation (str): YOLO-formatted annotation.
    - file_path (str): Path to the annotation file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(yolo_annotation)


def yolo_annotation(img_folder, config_path, weights_path, box_threshold=0.35, text_threshold=0.25):
    """
    Perform YOLO annotation on a folder of images using GroundingDino model.

    Parameters:
    - img_folder (str): Path to the folder containing images.
    - config_path (str): Path to the GroundingDino configuration file.
    - weights_path (str): Path to the GroundingDino weights file.
    - box_threshold (float): Bounding box threshold.
    - text_threshold (float): Text threshold.

    Returns:
    - tuple: Tuple containing dictionary of labels and list of non-annotated images.
    """
    model = load_model(config_path, weights_path)

    image_names = get_image_names(img_folder)
    data_folder = os.path.dirname(img_folder)
    annotations_folder = os.path.join(data_folder, "labels")

    dict_labels = {'boar': 0, 'deer': 1}

    non_annotated_imgs = []
    for image_name in tqdm(image_names):
        class_name = image_name.split('_')[0]

        if class_name not in dict_labels:
            new_value = max(dict_labels.values()) + 1
            dict_labels[class_name] = new_value

        try:
            boxes, phrases = annotate(model, image_name, img_folder, class_name, box_threshold, text_threshold)
            yolo_annotation = transform_to_yolo_annotation(phrases, boxes, dict_labels)

            filename = os.path.splitext(image_name)[0] + '.txt'
            annotation_file_path = os.path.join(annotations_folder, filename)
            save_annotation(yolo_annotation, annotation_file_path)
        except Exception as e:
            print(f"Error processing image '{image_name}': {e}")
            non_annotated_imgs.append(image_name)

    success_rate = round((1 - len(non_annotated_imgs) / len(image_names)) * 100, 2)
    print(f'Success rate: {success_rate}%')
    return dict_labels, non_annotated_imgs


if __name__ == "__main__":
    yolo_annotation(IMG_FOLDER, CONFIG_PATH, WEIGHTS_PATH)
