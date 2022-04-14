import cv2 as cv
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

capture = cv.VideoCapture(0)

while True:
    success, img = capture.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('Webcam', img)
    cv.waitKey(1)