import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
minValue = 70

#Must change before data collection
trainFolder = "data2/train/1"
testFolder = "data2/train"
tryFolder = "data2/try"

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        aspectRatio = h / w

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
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 2)

        # Sobel Edge Detection
        # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

        th3 = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)
        cv2.imshow("Resultant Image", res)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        #must rename the folder name (i.e "A_")
        cv2.imwrite(f'{trainFolder}/0-{counter}.jpg', res)
        print(counter)
