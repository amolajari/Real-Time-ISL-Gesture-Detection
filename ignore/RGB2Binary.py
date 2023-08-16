import os
import cv2
imgSize = 150
counter = 0
minValue = 70
offset = 20
sz = 150

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for letter in range(0,36):
    # Set the directory path containing the colored images
    print(labels[letter],"...")
    dir_path = f"C:\\Users\\amola\\PycharmProjects\\SignDetection\\36KDS\\original_images\\{labels[letter]}"

    # Set the directory path to save the binary images with edges
    save_dir_path = f"C:\\Users\\amola\\PycharmProjects\\SignDetection\\36KDS\\binary_images\\{labels[letter]}"

    # Loop through each file in the directory
    for filename in os.listdir(dir_path):
        print(labels[letter], filename, "\n")
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            img = cv2.imread(os.path.join(dir_path, filename))

            # Convert RGB to Gray
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(imgGray, (5, 5), 2)

            # Finding Edges
            th3 = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Save the binary image
            cv2.imwrite(os.path.join(save_dir_path, filename), res)

