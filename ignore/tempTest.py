from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2

model09AZ = load_model('model/model09AZ.h5')
img = cv2.imread("data-09AZ/Val/212.png")

resized_img = cv2.resize(img, (150, 150))
h, w, ch = resized_img.shape

labels = ["0", "1", "2","3","4","5","6","7","8","9","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

def getPredectedClassIndex(res):
    img2arr = img_to_array(res)
    imgexp = np.expand_dims(img2arr, axis=0)
    # Make the prediction
    predictions = model09AZ.predict(imgexp)
    # Get the predicted class index
    index = np.argmax(predictions)
    return index


#calling function to get Predicted index

path = 'data-09AZ/Val/212.png'

img1 = load_img(path, target_size=(w, h), color_mode="grayscale")

index = getPredectedClassIndex(img1)
print('The predicted Sign is:', labels[index])