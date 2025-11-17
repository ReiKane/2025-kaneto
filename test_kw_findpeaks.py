# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 画像を読み込み
img = cv2.imread('background.jpg')

# 半分にして鍵盤の部分を取り出す
w_orig, h_orig = img.shape[1], img.shape[0]
img_bottom = img[int(h_orig/2):h_orig, 0:w_orig]

gray = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5,5), 1)  # [HK]Sobelでもぼかすので外す

# [HK] 水平方向に平均を取って白い鍵盤のみの部分を取り出す
h_proj = np.mean(gray, axis=1)
plt.plot(h_proj)
plt.show()

# 黒鍵＋白鍵と、白鍵のみに分けるために閾値を設定
thresh_wkey = 150
# 閾値を越えるy座標の範囲を見つける
wkeys = np.where(h_proj > thresh_wkey)[0]
wkey_range = (wkeys[0], wkeys[-1])
print("White key range (y-coordinates):", wkey_range)

# 白鍵の部分だけを切り出す
margin = 5  # 少し削る
wkey_range = (wkey_range[0] + margin, wkey_range[1] - margin)
gray_wkey = gray[wkey_range[0]:wkey_range[1], :]

cv2.imshow('White keys', gray_wkey)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 垂直方向に射影
v_proj_wkey = gray_wkey.mean(axis=0)
plt.plot(v_proj_wkey)
plt.title('Vertical Projection of White Keys')
plt.show()

# %%
# 下向きのピーク検出で暗い部分を取り出す
# ピーク検出 find_peaks は上向きのみなので、値を反転させて使用
from scipy.signal import find_peaks
inverted_v_proj = -v_proj_wkey
print("Mean of vertical projection:", np.mean(v_proj_wkey))
min_dist_between_keys = 8  # 鍵盤同士の最小ピクセル距離（★ここは画像に合わせて要調整）
peaks, properties = find_peaks(inverted_v_proj, height= -np.mean(v_proj_wkey), distance=min_dist_between_keys)
print(peaks)
# ピークのプロット
plt.plot(inverted_v_proj)
plt.plot(peaks, inverted_v_proj[peaks], "x")
plt.title('Detected Valleys in Vertical Projection')
plt.show()

# ピークの高さの標準偏差を求め、異常に浅いピークを除外
peak_heights = properties['peak_heights']
std_height = np.std(peak_heights)
mean_height = np.mean(peak_heights)
filtered_peaks = [p for p, h in zip(peaks, peak_heights) if h < mean_height + 3 * std_height]
print("Filtered peaks:", filtered_peaks)
peaks = np.array(filtered_peaks)
plt.plot(inverted_v_proj)
plt.plot(peaks, inverted_v_proj[peaks], "x")
plt.title('Filtered Valleys in Vertical Projection')
plt.show()

# ピークの周期を計算
if len(peaks) > 1:
    peak_distances = np.diff(peaks)
    avg_distance = np.mean(peak_distances)
    print("Average distance between white keys (pixels):", avg_distance)

# ピークの抜けや誤検出を、周期性を利用して補正
corrected_peaks = []
expected_pos = peaks[0]
tolerance = avg_distance * 0.3  # 許容範囲
for peak in peaks:
    if abs(peak - expected_pos) <= tolerance:
        corrected_peaks.append(peak)
        expected_pos = peak + avg_distance
    else:
        # 予想位置に基づいて補正
        while expected_pos < peak - tolerance:
            corrected_peaks.append(int(expected_pos))
            expected_pos += avg_distance
        corrected_peaks.append(peak)
        expected_pos = peak + avg_distance
print("Corrected peaks:", corrected_peaks)

# 補正前と補正後のピークをプロットして比較
plt.figure(figsize=(10, 5))
plt.plot(inverted_v_proj)
plt.plot(peaks, inverted_v_proj[peaks], "x", label='Original Peaks')
plt.plot(corrected_peaks, inverted_v_proj[corrected_peaks], "o", label='Corrected Peaks')
plt.title('Corrected Valleys in Vertical Projection')
plt.legend()
plt.show()

# %%
# 元の画像に検出した白鍵の境界線を描画 (垂直線)
img_with_lines = img_bottom.copy()
for x in corrected_peaks:
    cv2.line(img_with_lines, (x, 0), (x, img_with_lines.shape[0]), (0, 0, 255), 1)

cv2.imshow('Detected White Key Boundaries', img_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

from ultralytics import YOLO
model = YOLO("runs/detect/train4/weights/best.pt")

# 動画読込
cap = cv2.VideoCapture("../piano_test_3.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

