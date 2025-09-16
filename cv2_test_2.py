import cv2
import numpy as np

pic = cv2.imread("background.jpg")
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
pic_median = cv2.medianBlur(pic, 5)
sobel_x = cv2.Sobel(pic_median,cv2.CV_32F,1,0,ksize = 5)
sobel_y = cv2.Sobel(pic_median,cv2.CV_32F,0,1,ksize = 5)
sobel_x = cv2.convertScaleAbs(sobel_x, alpha = 0.5)
sobel_y = cv2.convertScaleAbs(sobel_y, alpha = 0.5)
sobel_xy = cv2.add(sobel_x, sobel_y)
_, sobel_xy = cv2.threshold(sobel_xy, 160, 255, cv2.THRESH_BINARY)

length = 10
gap = 10
lines = cv2.HoughLinesP(sobel_xy.astype(np.uint8), 1, np.pi/180, 100, length, gap)
sobel_xy = cv2.cvtColor(sobel_xy, cv2.COLOR_GRAY2BGR)
print("lines: " + str(len(lines)))
for x1,y1,x2,y2 in lines.squeeze():
    if(abs(x1-x2) <= 3):
        cv2.line(sobel_xy,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("Sobel", sobel_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()