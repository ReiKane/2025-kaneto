import cv2
import numpy as np

# 1. 画像を読み込み
img = cv2.imread('background.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 1)

# 2. Sobelフィルタでエッジを検出
# dx=1, dy=0 は水平方向、dx=0, dy=1 は垂直方向の勾配
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# 3. 勾配強度（エッジ強度）を計算
abssobelx = cv2.convertScaleAbs(sobelx)

# 4. エッジ画像を二値化（閾値は画像に応じて調整）
_, edges = cv2.threshold(abssobelx, 60, 255, cv2.THRESH_BINARY)

#kernel = np.ones((3, 3), np.uint8)
#edges = cv2.dilate(edges, kernel, iterations=1)


cv2.imshow('Detected Lines', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Hough変換で直線を検出
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                        minLineLength=20, maxLineGap=10)
print("lines: " + str(len(lines)))

# 6. 検出した直線を元画像に描画
drawed = 0
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(angle) > 80:
            drawed += 1
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 7. 結果を表示
print("drawed: " + str(drawed))
cv2.imshow('Detected Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()