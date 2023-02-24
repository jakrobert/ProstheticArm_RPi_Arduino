import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands = 2, complexity = 1, detectionConf=0.5, trackConf=0.5):  #Complexity param added!!!
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils  # method for drawing hand vectors

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert img to RGB for hands
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:  # check if there are multiple hands
            for handLms in self.results.multi_hand_landmarks:  # for each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                      self.mpHands.HAND_CONNECTIONS)  # draw dots on the joints(handLms), and hand connections mpHands.HAND_CONNECTIONS

        return img

    def findPosition(self, img, handNo=0, draw = True): #position for a specific hand

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(
                   myHand.landmark):  # id, lm-landmark vector(x, y, z) <- those are a ratio of the image => they have to be processed
                h, w, c = img.shape  # h- height, w - width, c - center
                cx, cy = int(lm.x * w), int(lm.y * h)  # x and y position
                lmList.append([id, cx, cy])

        return lmList



def main():
    prevTime = 0
    ctTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
               print(lmList)
        angle = detector.findAngles()


        # configure thee fps
    ctTime = time.time()
    FPS = 1 / (ctTime - prevTime)
    prevTime = ctTime

        # display FPS
    cv2.putText(img, str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                    3)  # img, value , pos, font, scale, color, thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()