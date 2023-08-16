import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

letter = 'A'
trainFolder = f"own-dataset/train/{letter}"
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0
minValue = 70

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

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Resultant Image", res)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        #must rename the folder name (i.e "A_")
        cv2.imwrite(f'{trainFolder}/{letter}-{counter}.jpg', res)
        print(counter)
