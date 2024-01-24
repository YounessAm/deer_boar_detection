import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from utils import get_image_names

# Set the path to your data folder
HOME = os.getcwd()
HOME = os.path.dirname(HOME)
DATA_FOLDER = os.path.join(HOME,'data')

def list_file_byclass(image_folder, dict_labels, non_annotated_imgs):
    """
    List image files by class based on a dictionary of labels.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - dict_labels (dict): Dictionary mapping class names to labels.
    - non_annotated_imgs (list): List of non-annotated image names.

    Returns:
    - dict: Dictionary mapping class names to lists of image files.
    """
    final_dict = {}

    for key in dict_labels.keys():
        final_dict[key] = [image_file for image_file in get_image_names(image_folder)
                       if (image_file not in non_annotated_imgs) and (image_file.split('_')[0] == key)]
        
    return final_dict

def create_train_valid_test(DATA_FOLDER, non_annotated_imgs):
    """
    Create training, validation, and testing sets.

    Parameters:
    - DATA_FOLDER (str): Path to the data folder.
    - non_annotated_imgs (list): List of non-annotated image names.

    Returns:
    - tuple: Tuple containing lists of training, validation, and testing image files.
    """
    # List all image files in the 'images' folder 
    image_folder = os.path.join(DATA_FOLDER, 'images')
    image_files = [image_file for image_file in get_image_names(image_folder)
                       if (image_file not in non_annotated_imgs)]
    
    # Split the data into training, validation, and testing sets
    train_images, test_images = train_test_split(image_files, test_size=0.2)
    train_images, valid_images = train_test_split(train_images, test_size=0.2)

    return train_images, valid_images, test_images

def move_images(source_folder, destination_folder, image_list):
    """
    Move images from source folder to destination folder 
    and move labels of those image wtih the same mouvement as images

    Parameters:
    - source_folder (str): Path to the source image folder.
    - destination_folder (str): Path to the destination image folder.
    - image_list (list): List of image files to be moved.
    """
    for image in tqdm(image_list):
        img_source_path = os.path.join(source_folder, image)
        img_destination_path = os.path.join(destination_folder, image)

        label = image.split('.')[0]+'.txt'
        label_source_folder = os.path.join(DATA_FOLDER, 'labels')
        label_destination_folder = os.path.join(DATA_FOLDER, 'labels', destination_folder.split('/')[-1])

        label_source_path = os.path.join(label_source_folder, label)
        label_destination_path = os.path.join(label_destination_folder, label)

        shutil.copy(img_source_path , img_destination_path)
        shutil.copy(label_source_path , label_destination_path)

def create_train_valid_test_stratified(DATA_FOLDER, non_annotated_imgs):
    """
    Create stratified training, validation, and testing sets.

    Parameters:
    - DATA_FOLDER (str): Path to the data folder.
    - non_annotated_imgs (list): List of non-annotated image names.

    Returns:
    - tuple: Tuple containing lists of training, validation, and testing image files.
    """
    # List all image files in the 'images' folder 
    image_folder = os.path.join(DATA_FOLDER, 'images')
    image_files = [image_file for image_file in get_image_names(image_folder)
                       if (image_file not in non_annotated_imgs)]

    # Extract labels from file names
    labels = [f.split('_')[0] for f in image_files]

    # Split the data into training, validation, and testing sets with stratification
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_index, test_index = next(stratified_splitter.split(image_files, labels))

    train_images, test_images = [image_files[i] for i in train_index], [image_files[i] for i in test_index]
    train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]

    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_index, valid_index = next(stratified_splitter.split(train_images, train_labels))

    train_images, valid_images = [train_images[i] for i in train_index], [train_images[i] for i in valid_index]

    return train_images, valid_images, test_images

def create_train_valid_test_folders(DATA_FOLDER, stratified=True):
    """
    Create folders for training, validation, and testing sets. and copiying the data there

    Parameters:
    - DATA_FOLDER (str): Path to the data folder.
    - stratified (bool): Whether to use stratified sampling.

    Returns:
    - tuple: Tuple containing lists of training, validation, and testing image files.
    """
    # Define the paths for the new images folders
    train_folder = os.path.join(DATA_FOLDER,'images','train_data')
    valid_folder = os.path.join(DATA_FOLDER,'images','valid_data')
    test_folder = os.path.join(DATA_FOLDER,'images','test_data')

    # Create the new folders if they don't exist
    for img_folder in [train_folder, valid_folder, test_folder]:
        labels_folder = os.path.join(DATA_FOLDER, 'labels', img_folder.split('/')[-1])
        
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

    if stratified:
        train_images, valid_images, test_images = create_train_valid_test_stratified(DATA_FOLDER, non_annotated_imgs)
    else:
        train_images, valid_images, test_images = create_train_valid_test(DATA_FOLDER, non_annotated_imgs)

    image_folder = os.path.join(DATA_FOLDER, 'images')
    move_images(image_folder, train_folder, train_images)
    move_images(image_folder, valid_folder, valid_images)
    move_images(image_folder, test_folder, test_images)


if __name__ == "__main__":
    # Set the path to your data folder
    DATA_FOLDER = os.path.join(HOME,'data')
    create_train_valid_test_folders(DATA_FOLDER, stratified = True)