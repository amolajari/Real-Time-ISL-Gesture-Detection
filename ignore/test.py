from cvzone.HandTrackingModule import HandDetector
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2
import math

modelbw = load_model('model/model09AZ.h5')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300
counter = 0
minValue = 70
offset = 20
sz = 300
labels = ["1", "2", "3"]


def getPredectedClassIndex(res):
    img2arr = img_to_array(res)
    imgexp = np.expand_dims(img2arr, axis=0)
    # Make the prediction
    predictions = modelbw.predict(imgexp)
    # Get the predicted class index
    index = np.argmax(predictions)
    return index


while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio >1:
            k = imgSize/h
            wCalc = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCalc) / 2)
            imgWhite[:, wGap:wCalc+wGap] = imgResize
        else:
            k = imgSize / w
            hCalc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalc))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[hGap:hCalc + hGap, :] = imgResize

        #Converting RGB to Gray
        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(imgGray, (5, 5), 2)
        #Finding Edges
        th3 = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imshow("ImageBW", res)
        #calling function to get Predicted index
        index = getPredectedClassIndex(res)
        print('The predicted Sign is:', labels[index])

    cv2.imshow("Image", img)
    cv2.waitKey(1)