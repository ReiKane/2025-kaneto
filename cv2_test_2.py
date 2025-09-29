import cv2
import numpy as np

pic = cv2.imread("../goban_test.jpg")
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow("pic", pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
pic_blur = cv2.GaussianBlur(pic, (3,3), 0)
pic_edges = cv2.Canny(pic_blur, threshold1=50, threshold2=150, apertureSize=3)

length = 100
gap = 5
lines = cv2.HoughLinesP(pic_edges.astype(np.uint8), 1, np.pi/180, 100, length, gap)
pic_edges = cv2.cvtColor(pic_edges, cv2.COLOR_GRAY2BGR)
print("lines: " + str(len(lines)))
for x1,y1,x2,y2 in lines.squeeze():
    if(abs(x1-x2) <= 3):
        cv2.line(pic_edges,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("Canny", pic_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()