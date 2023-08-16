import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.utils import load_img, img_to_array
from keras.models import load_model
import pyttsx3
import statistics


engine = pyttsx3.init()

model09AZ = load_model('model/model09AZ-OWNDS-95.h5',compile=False)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
imgSize = 300
counter = 0
minValue = 70
offset = 20
sz = 300
predCount = 0
predList = []

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def getPredectedClassIndex(res):
    img2arr = img_to_array(res)
    imgexp = np.expand_dims(img2arr, axis=0)
    # Make the prediction
    predictions = model09AZ.predict(imgexp)
    # Get the predicted class index
    index = np.argmax(predictions)
    return index

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        hand1 = hands[0]
        x, y, w, h = hand1['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        # imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        if len(hands) == 2:
            hand2 = hands[1]
            x1, y1, w1, h1 = hand2['bbox']
            xmin = min(x, x1)
            ymin = min(y, y1)
            xmax = max(x + w, x1 + w1)
            ymax = max(y + h, y1 + h1)
            imgCrop = img[ymin - offset:ymax+offset, xmin - offset:xmax+offset]

        aspectRatio = h / w
        if not imgCrop.size == 0:
            if aspectRatio > 1:
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

        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

        th3 = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imshow("Resultant Image", res)

        #calling function to get Predicted index
        index = getPredectedClassIndex(res)
        print('Predicted Sign is:', labels[index])

        predList.append(labels[index])
        predCount = predCount+1

        if predCount == 15:
            modeLabel = statistics.mode(predList)
            print('Mode:', modeLabel)
            engine.say(modeLabel)
            engine.runAndWait()

            # Reset counter and list
            predCount = 0
            predList = []


    cv2.imshow("Image", img)
    cv2.waitKey(1)