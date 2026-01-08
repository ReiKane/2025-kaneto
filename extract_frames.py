# %%
import cv2
import numpy as np
import os

print("Importing done.")

#動画と背景読み込み
bg = cv2.imread("background_2.jpg")
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (3,3), 0)
#エッジの座標(別のコードで取得)
key_boundaries = [29, 66, 103, 141, 179, 216, 254, 292, 330, 368, 405, 443, 480, 517, 554, 592, 629, 665, 702, 739, 775, 812, 849, 885, 922, 958, 995, 1032, 1069, 1105, 1142, 1178, 1215, 1251, 1288, 1325, 1362, 1398, 1435, 1472, 1509, 1546, 1583, 1621, 1658, 1695, 1733, 1770, 1807, 1844, 1880, 1903]
press_log = []
threshold = 254

RAWFRAME_ODIR = "raw_frames/"
SAVE_RAWFRAME_ONLY = True
GEN_BG_ONLY = False
# FRAME_TO_SAVE = 800

os.makedirs(RAWFRAME_ODIR, exist_ok=True)

img_array = []

tlim = [2980, 3734]

print("Starting processing...")
# t = -1

cap = cv2.VideoCapture("piano_test_5.mp4")
if cap.isOpened() == False:
    print("Error opening video file")
else:
    print(f"Video file opened successfully: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames found.")

cap.set(cv2.CAP_PROP_POS_FRAMES, tlim[0])

while cap.isOpened() and (cap.get(cv2.CAP_PROP_POS_FRAMES) < tlim[1]):
    t = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        print("ret is False")
        break
 
    # t += 1
    print(f"Processing frame {t}")
    if GEN_BG_ONLY:
        img_array.append(frame)
        # if t >= FRAME_TO_SAVE:
        #     break
        continue

    if SAVE_RAWFRAME_ONLY:
        cv2.imwrite(f"{RAWFRAME_ODIR}/frame_{t:04d}.jpg", frame)
        # if t >= FRAME_TO_SAVE:
        #     break
        continue

    #前処理・差分取得
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    diff = cv2.absdiff(gray, bg)
    _, diff_bin = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    roi = diff_bin[987:1054, :]#検出に使う範囲
    press_states = []
    for i in range(len(key_boundaries) - 1):
        x1, x2 = key_boundaries[i], key_boundaries[i + 1]
        key_roi_1 = roi[:, x1:x1+1]#左端
        key_roi_2 = roi[:, x2:x2+1]#右端
        mean_val = np.mean(key_roi_1 + key_roi_2)
        pressed = mean_val > threshold
        press_states.append(pressed)

        # 押されている鍵盤を赤枠で表示
        color = (0, 0, 255) if pressed else (255, 255, 255)
        cv2.rectangle(frame, (x1, 987), (x2, 1054), color, 2)


    #ログ記録(今後に向けて出せるようにしています)
    press_log.append(press_states)
    
    frame = cv2.resize(frame, [960, 540])
    cv2.imshow('背景差分', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# %%
cap.release()
cv2.destroyAllWindows()

# %%
if GEN_BG_ONLY:
    img_array = np.array(img_array)
    print(img_array.shape)
    bg_new = np.median(img_array, axis=0).astype(np.uint8)
    cv2.imwrite("background_all.jpg", bg_new)
    print("Background image generated and saved as background_all.jpg")
