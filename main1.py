import cv2
import mediapipe as mp
import time
import TrakingHandMod as thm


cap = cv2.VideoCapture(0)
detector = thm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print("/n")
        print(lmList)



