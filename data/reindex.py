import os
from PIL import Image


# Set the path to the directory containing the images
directory_root = 'data/celeba/'
valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}


for directory_name in os.listdir(directory_root):
    directory_path = directory_root + directory_name

    # Get a list of files in the directory
    files = os.listdir(directory_path)
    
    # Sort the files to ensure they are processed in the correct order
    files.sort()
    
    # Rename each file sequentially
    filtered_files = []
    for index, file_name in enumerate(files):
        # Create the new file name
        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension not in valid_extensions:
            continue
        filtered_files.append(file_name)
        
    for index, file_name in enumerate(filtered_files):
        new_name = f"{index}.jpg"
        
        # Get the full paths for the old and new file names
        old_file_path = os.path.join(directory_path, file_name)
        new_file_path = os.path.join(directory_path, new_name)

        with Image.open(old_file_path) as img:
            # Save the image as a PNG
            img.save(new_file_path, "PNG")
        # Delete the original JPEG file
        os.remove(old_file_path)
    
    print("Files have been renamed successfully.")