from PIL import Image
import numpy as np
import os

def create_image_mosaic(input_folder, output_path, label, mosaic_size=(800, 800), max_img = 10, rows=2, columns=5):
    # Get a list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if (f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))) and (f.split('_')[0]==label)]
    image_files = image_files[:max_img]

    if not image_files:
        print("No image files found in the input folder.")
        return

    # Calculate the size of each tile in the mosaic
    tile_width = mosaic_size[0] // columns
    tile_height = mosaic_size[1] // rows

    # Create an empty mosaic image
    mosaic = Image.new('RGB', mosaic_size)  

    # Iterate through the images and paste them onto the mosaic
    for i, image_file in enumerate(image_files):
        row = i // columns
        col = i % columns
        image_path = os.path.join(input_folder, image_file)
        try :
            img = Image.open(image_path)
            img = img.resize((tile_width, tile_height), Image.LANCZOS)
            mosaic.paste(img, (col * tile_width, row * tile_height))
        except :
            pass

    # Save the resulting mosaic image
    mosaic.save(output_path)
    print(f"Mosaic created and saved to {output_path}")

if __name__ == "__main__":
    
    HOME = os.getcwd()
    IMG_FOLDER = os.path.join(HOME, "data", "images")
    

    # Specify the output path for the mosaic image
    output_path = "deer_mosaic.jpg"
    create_image_mosaic(IMG_FOLDER, output_path, label = 'deer')
    
    output_path = "boar_mosaic.jpg"
    create_image_mosaic(IMG_FOLDER, output_path, label = 'boar')



