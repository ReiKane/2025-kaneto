import cv2
import numpy as np

pic = cv2.imread("../goban_test.jpg")
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
pic = cv2.GaussianBlur(pic, (3,3), 1)
cv2.imshow("pic", pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
#sobel_x = cv2.Sobel(pic,cv2.CV_32F,1,0,ksize = 5)
#sobel_y = cv2.Sobel(pic,cv2.CV_32F,0,1,ksize = 5)
#sobel_x = cv2.convertScaleAbs(sobel_x, alpha = 0.5)
#sobel_y = cv2.convertScaleAbs(sobel_y, alpha = 0.5)
#sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sobel_xy = cv2.Canny(pic, 50, 150, apertureSize=3)

length = 100
gap = 5
#lines = cv2.HoughLinesP(sobel_xy.astype(np.uint8), 1, np.pi/360, 100, length, gap)
lines = cv2.HoughLinesP(sobel_xy, 1, np.pi/180, 100, length, gap)
sobel_xy = cv2.cvtColor(sobel_xy, cv2.COLOR_GRAY2BGR)
cv2.imshow("hough", sobel_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("lines: " + str(len(lines)))
for x1,y1,x2,y2 in lines.squeeze():
    if(abs(x1-x2) <= 3):
        cv2.line(sobel_xy,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("hough", sobel_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()