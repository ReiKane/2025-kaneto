import cv2
import numpy as np

cap = cv2.VideoCapture("../piano_test_5.mp4")
bg = cv2.imread("background_2.jpg")
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (3,3), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (3,3), 0)
        diff = cv2.absdiff(frame, bg)
        _, diff_bin = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        cv2.imshow('背景差分', diff_bin)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    else:
        break
