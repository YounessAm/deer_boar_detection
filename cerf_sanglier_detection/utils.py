import os

def get_image_names(img_folder, extensions=None):
    """
    Get a list of image names in the specified folder with the given extensions.

    Parameters:
    - img_folder (str): Path to the folder containing images.
    - extensions (list): List of allowed image file extensions.

    Returns:
    - list: List of image names.
    """
    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg", ".gif", ".bmp"]  # Add more extensions if needed

    image_names = [file for file in os.listdir(img_folder) if any(file.lower().endswith(ext) for ext in extensions)]
    return image_names


def delete_file(file_path):
    """
    Delete a file at the specified path.

    Parameters:
    - file_path (str): Path to the file to be deleted.
    """
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting file '{file_path}': {e}")