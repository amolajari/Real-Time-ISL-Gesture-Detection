# import os
# import shutil
#
# source_folder = '/path/to/source/folder'
# destination_folder = '/path/to/destination/folder'
#
# # Get a list of all files in the source folder
# files = os.listdir(source_folder)
#
# # Move the first 100 files from the source folder to the destination folder
# for i in range(100):
#     source_file = os.path.join(source_folder, files[i])
#     destination_file = os.path.join(destination_folder, files[i])
#     shutil.move(source_file, destination_file)
#
import os
import random

# Path to the folder containing the dataset
dataset_path = "own-dataset/test"

# Iterate over all subdirectories of the dataset folder
for subfolder in os.listdir(dataset_path):
    # Construct the path to the subfolder
    subfolder_path = os.path.join(dataset_path, subfolder)

    # Check if the path is a directory
    if os.path.isdir(subfolder_path):
        # Get a list of all the image files in the subfolder
        image_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(".jpg")]

        # Check if there are at least 25 images in the subfolder
        if len(image_files) >= 40:
            # Choose 25 random images to delete
            images_to_delete = random.sample(image_files, 40)

            # Delete the chosen images
            for image_path in images_to_delete:
                os.remove(image_path)

            print(f"Deleted 40 images from folder {subfolder}")
        else:
            print(f"Skipping folder {subfolder} (less than 40 images)")
