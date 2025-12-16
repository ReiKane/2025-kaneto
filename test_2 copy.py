import cv2
import numpy as np

cap = cv2.VideoCapture("../piano_test_5.mp4")
bg = cv2.imread("background_2.jpg")
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (3,3), 0)
key_boundaries = [29, 66, 103, 141, 179, 216, 254, 292, 330, 368, 405, 443, 480, 517, 554, 592, 629, 665, 702, 739, 775, 812, 849, 885, 922, 958, 995, 1032, 1069, 1105, 1142, 1178, 1215, 1251, 1288, 1325, 1362, 1398, 1435, 1472, 1509, 1546, 1583, 1621, 1658, 1695, 1733, 1770, 1807, 1844, 1880, 1903]
press_log = []
threshold = 100

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        diff = cv2.absdiff(gray, bg)
        _, diff_bin = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        roi = diff_bin[987:1054, :]
        press_states = []
        for i in range(len(key_boundaries) - 1):
            x1, x2 = key_boundaries[i], key_boundaries[i + 1]
            key_roi = roi[:, x1:x1+1]
            mean_val = np.mean(key_roi)
            pressed = mean_val > threshold
            press_states.append(pressed)

            # 押されている鍵盤を赤枠で表示
            color = (0, 0, 255) if pressed else (255, 255, 255)
            cv2.rectangle(frame, (x1, 987), (x2, 1054), color, 2)

        press_log.append(press_states)
        frame = cv2.resize(frame, [960, 540])
        cv2.imshow('背景差分', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    else:
        break
