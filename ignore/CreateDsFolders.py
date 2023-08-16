import os

# Create the folder if it does not exist
Bi_folder_path = 'C:\\Users\\amola\\PycharmProjects\\SignDetection\\own-dataset'
if not os.path.exists(Bi_folder_path):
    os.mkdir(Bi_folder_path)

# Create directories for digits
for i in range(10):
    os.mkdir(os.path.join(Bi_folder_path, str(i)))

# Create directories for letters
for letter in range(ord('A'), ord('Z')+1):
    os.mkdir(os.path.join(Bi_folder_path, chr(letter)))

#
# import os
# import random
# import shutil
#
# # set the path to the binary_images directory
# dir_path = "C:\\Users\\amola\\PycharmProjects\\SignDetection\\36KDS\\binary_images"
#
# # set the path to the training and testing directories
# train_dir = "C:\\Users\\amola\\PycharmProjects\\SignDetection\\36KDS\\binary_images1\\train"
# test_dir = "C:\\Users\\amola\\PycharmProjects\\SignDetection\\36KDS\\binary_images1\\test"
#
# # create the training and testing directories if they don't exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
#
# # loop over each label directory
# for label in os.listdir(dir_path):
#     # create the corresponding directories in the training and testing sets
#     os.makedirs(os.path.join(train_dir, label), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, label), exist_ok=True)
#
#     # get a list of all the image filenames in the current label directory
#     filenames = os.listdir(os.path.join(dir_path, label))
#
#     # shuffle the filenames
#     random.shuffle(filenames)
#
#     # move the first 750 filenames to the training set
#     for filename in filenames[:750]:
#         src = os.path.join(dir_path, label, filename)
#         dst = os.path.join(train_dir, label, filename)
#         shutil.copy(src, dst)
#
#     # move the remaining 250 filenames to the testing set
#     for filename in filenames[750:1000]:
#         src = os.path.join(dir_path, label, filename)
#         dst = os.path.join(test_dir, label, filename)
#         shutil.copy(src, dst)
