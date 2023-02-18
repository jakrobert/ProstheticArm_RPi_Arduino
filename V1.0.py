import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # method for drawing hand vectors
prevTime = 0
ctTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert img to RGB for hands
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:  # check if there are multiple hands
        for handLms in results.multi_hand_landmarks:  # for each hand
            for id, lm in enumerate(
                    handLms.landmark):  # id, lm-landmark vector(x, y, z) <- those are a ratio of the image => they have to be processed
                h, w, c = img.shape  # h- height, w - width, c - center
                cx, cy = int(lm.x * w), int(lm.y * h)  # x and y position

            mpDraw.draw_landmarks(img, handLms,
                                  mpHands.HAND_CONNECTIONS)  # draw dots on the joints(handLms), and hand connections mpHands.HAND_CONNECTIONS

    # configure thee fps
    ctTime = time.time()
    FPS = 1 / (ctTime - prevTime)
    prevTime = ctTime

    # display FPS
    cv2.putText(img, str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                3)  # img, value , pos, font, scale, color, thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)
