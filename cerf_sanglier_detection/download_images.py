import os
import requests

HOME = os.getcwd()
HOME = os.path.dirname(HOME)

URL_FILES_PATH = os.path.join(HOME, "data", "urls")
IMG_FOLDER = os.path.join(HOME, "data", "images")

def download_images_from_file(file_path, output_folder):
    """
    Download images from URLs listed in a file.

    Parameters:
    - file_path (str): Path to the file containing image URLs.
    - output_folder (str): Path to the folder where downloaded images will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract class name from the file path
    class_name = os.path.splitext(os.path.basename(file_path))[0].split("_")[0]

    # Read URLs from the file
    with open(file_path, 'r') as file:
        urls = file.readlines()

    # Download images
    for i, url in enumerate(urls):
        url = url.strip()  # Remove leading/trailing whitespaces
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for errors

            # Save the image to the output folder
            image_name = f"{class_name}_{i + 1}.jpg"
            image_path = os.path.join(output_folder, image_name)
            with open(image_path, 'wb') as image_file:
                image_file.write(response.content)

            print(f"{class_name} {i + 1} downloaded successfully: {image_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {class_name} {i + 1}: {e}")

# Example usage:
# download_images_from_file("path/to/your/urls.txt", "path/to/your/output_folder")
            
if __name__ == "__main__":
    
    urls_files = os.listdir(URL_FILES_PATH)
    urls_files = [os.path.join(URL_FILES_PATH, urls_file) for urls_file in urls_files]

    # Download Images
    for urls_file in urls_files:
        download_images_from_file(urls_file, IMG_FOLDER)