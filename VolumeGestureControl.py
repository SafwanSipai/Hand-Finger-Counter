import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#################################
wCam = hCam = 640, 480
#################################


capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

detector = htm.HandDetector(detectionConfidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]


while True:
    success, img = capture.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv.circle(img, (x1, y1), 10, (0, 255, 0), thickness=cv.FILLED)
        cv.circle(img, (x2, y2), 10, (0, 255, 0), thickness=cv.FILLED)
        cv.circle(img, (cx, cy), 10, (0, 255, 0), thickness=cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hand length range -> 50 - 300
        # Volume range -> -65 - 0

        vol = np.interp(length, [50, 300], [minVolume, maxVolume])
        volume.SetMasterVolumeLevel(vol, None)

        length = math.hypot(x2-x1, y2-y1)
        if length < 50:
            cv.circle(img, (cx, cy), 10, (255, 255, 0), thickness=cv.FILLED)

    cv.imshow('Webcam', img)
    cv.waitKey(1)