import os
from ultralytics import YOLO

def create_config_str(DATA_FOLDER, train_folder, valid_folder, dict_labels):
    """
    Create the content of a config.yaml file for training.

    Parameters:
    - DATA_FOLDER (str): Path to the data folder.
    - train_folder (str): Path to the training data folder.
    - valid_folder (str): Path to the validation data folder.
    - dict_labels (dict): Dictionary mapping class names to labels.

    Returns:
    - str: Content of the config.yaml file.
    """
    output = f'\npath: {DATA_FOLDER}\n'
    output += f'train: {train_folder}\n'
    output += f'val: {valid_folder}\n'
    output += '\n# Classes \nnames:\n'
    for class_name, class_label in dict_labels.items():
        output += f'  {class_label}: {class_name}\n'
    return output

def save_config_file(config_content, file_path):
    """
    Save the content to a file.

    Parameters:
    - config_content (str): Content to be saved.
    - file_path (str): Path to the file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Open the file in write mode ('w')
    with open(file_path, 'w') as file:
        # Write the content to the file
        file.write(config_content)

def create_config_file(DATA_FOLDER, train_folder, valid_folder, dict_labels, file_path):
    """
    Create a config.yaml file for training.

    Parameters:
    - DATA_FOLDER (str): Path to the data folder.
    - train_folder (str): Path to the training data folder.
    - valid_folder (str): Path to the validation data folder.
    - dict_labels (dict): Dictionary mapping class names to labels.
    - file_path (str): Path to the config.yaml file.
    """
    config_content = create_config_str(DATA_FOLDER, train_folder, valid_folder, dict_labels)
    save_config_file(config_content, file_path)



if __name__ == "__main__":
    HOME = os.getcwd()
    HOME = os.path.dirname(HOME)
    DATA_FOLDER = os.path.join(HOME, 'data')
    train_folder = os.path.join(DATA_FOLDER, 'images', 'train_data')
    valid_folder = os.path.join(DATA_FOLDER, 'images', 'valid_data')
    dict_labels = {'boar': 0, 'deer': 1}
    config_file_path = os.path.join(HOME, 'config.yaml')

    NB_EPOCHS = 300
    NB_BATCHES = 7

    create_config_file(DATA_FOLDER, train_folder, valid_folder, dict_labels, config_file_path)

    model = YOLO("yolov8n.pt") # load pretrained model
    
    results = model.train(data=config_file_path, epochs=NB_EPOCHS, batch = NB_BATCHES)  # train the model