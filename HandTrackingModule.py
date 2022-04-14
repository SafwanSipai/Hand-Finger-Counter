import cv2 as cv
import mediapipe as mp
import time

class HandDetector():

    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackingConfidence=0.5):
         self.mode = mode
         self.maxHands = maxHands
         self.modelComplexity = modelComplexity
         self.detectionConfidence = detectionConfidence
         self.trackingConfidence = trackingConfidence
         self.mpHands = mp.solutions.hands
         self.mpDraw = mp.solutions.drawing_utils
         self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackingConfidence)


    def findHands(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)
        return lmList

def main():
    # pTime = 0
    # cTime = 0
    detector = HandDetector()
    capture = cv.VideoCapture(0)
    while True:
        success, img = capture.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv.putText(img, str(int(fps)), (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Webcam', img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()