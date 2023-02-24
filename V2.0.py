import cv2
import mediapipe as mp
import time
import numpy as np
import math


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # method for drawing hand vectors
prevTime = 0
ctTime = 0

def drawHands(img):

    if results.multi_hand_landmarks:  # check if there are multiple hands
        for handLms in results.multi_hand_landmarks:  # for each hand
            for id, lm in enumerate(
                    handLms.landmark):  # id, lm-landmark vector(x, y, z) <- those are a ratio of the image => they have to be processed
                h, w, c = img.shape  # h- height, w - width, c - center
                cx, cy = int(lm.x * w), int(lm.y * h)  # x and y position

            mpDraw.draw_landmarks(img, handLms,
                                  mpHands.HAND_CONNECTIONS)  # draw dots on the joints(handLms), and hand connections mpHands.HAND_CONNECTIONS
    cv2.imshow("Image", img)
    cv2.waitKey(1)
def findPosition(img, handNo = 0):
    landMarkList = []
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(hand.landmark):
            h, w, c = img.shape  # h- height, w - width, c - center
            cx, cy = int(lm.x * w), int(lm.y * h)  # x and y position
            landMarkList.append([id, cx, cy])

        return landMarkList

def findDistance(landMarkList, a, b):
    length = 0
    info = ()
    if results.multi_hand_landmarks:
        # distance = np.sqrt((landMarkList[a][1]-landMarkList[b][1])^2 + (landMarkList[a][2]-landMarkList[b][2])^2)
        x1 = landMarkList[a][1]
        y1 = landMarkList[a][2]
        x2 = landMarkList[b][1]
        y2 = landMarkList[b][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
    return length, info



# def findAngles(joint_list):
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for joint in joint_list:
#                 a = np.array([handLms.landmark[joint[0]].x, handLms.landmark[joint[0]].y])  # First coord
#                 b = np.array([handLms.landmark[joint[1]].x, handLms.landmark[joint[1]].y])  # Second coord
#                 c = np.array([handLms.landmark[joint[2]].x, handLms.landmark[joint[2]].y])  # Third coord
#
#                 radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#                 angle = np.abs(radians * 180.0 / np.pi)
#
#                 if angle > 180.0:
#                     angle = 360 - angle
#
#         return angle


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert img to RGB for hands
    results = hands.process(imgRGB)

    drawHands(img)
    landMarkList = findPosition(img)



    a= 8
    b= 5
    print(findDistance(landMarkList, a, b))




    # configure thee fps
    ctTime = time.time()
    FPS = 1 / (ctTime - prevTime)
    prevTime = ctTime

    # display FPS
    cv2.putText(img, str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                3)  # img, value , pos, font, scale, color, thickness

