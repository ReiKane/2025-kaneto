# %%
# 静止画数枚に対して鍵盤の押下検出のアルゴリズムを検討するコード
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

RAWFRAME_IDIR = "raw_frames/"

#エッジの座標(別のコードで取得)
key_boundaries = [29, 66, 103, 141, 179, 216, 254, 292, 330, 368, 405, 443, 480, 517, 554, 592, 629, 665, 702, 739, 775, 812, 849, 885, 922, 958, 995, 1032, 1069, 1105, 1142, 1178, 1215, 1251, 1288, 1325, 1362, 1398, 1435, 1472, 1509, 1546, 1583, 1621, 1658, 1695, 1733, 1770, 1807, 1844, 1880, 1903]

# %%
bg_orig = cv2.imread("background_2.jpg")
print("Original background shape:", bg_orig.shape)

# frame_D4off = 114
# frame_D4on = 139
frame_D4off = 3142
frame_D4on = 3162
img_on_orig = cv2.imread(f"{RAWFRAME_IDIR}/frame_{frame_D4on:04d}.jpg")
img_off_orig = cv2.imread(f"{RAWFRAME_IDIR}/frame_{frame_D4off:04d}.jpg")

xlim = [750, 1250]

# ylim = [830, 1060]  # 830 + 230 = 1060
# figsize = (8, 4)
ylim = [987, 1060]  # 987 + 73 = 1060
figsize = (12, 3)

# xオフセットを保存（座標変換用）
x_offset = xlim[0]

bg = bg_orig[ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.float32)
img_on = img_on_orig[ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.float32)
img_off = img_off_orig[ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.float32)
print(bg.shape)

# %%
# 画像の確認（RとBを入れ替えて）

fig = plt.figure(figsize=figsize)
plt.imshow(bg[:, :, ::-1].astype(np.uint8))
plt.title("Background")
plt.show()
fig = plt.figure(figsize=figsize)
plt.imshow(img_off[:, :, ::-1].astype(np.uint8))
plt.title(f"Frame {frame_D4off} (D4 off)")
plt.show()
fig = plt.figure(figsize=figsize)
plt.imshow(img_on[:, :, ::-1].astype(np.uint8))
plt.title(f"Frame {frame_D4on} (D4 on)")
plt.show()


# %% 
# --- エッジマスク方式 -------------------------------
# エッジ部分だけ取り出して差分を確認
# まずエッジの確認
fig = plt.figure(figsize=figsize)
bg_sblx = cv2.Sobel(cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=5)
plt.imshow(bg_sblx, cmap='gray')
plt.colorbar()
plt.title("Sobel X of Background")
plt.show()

# エッジ部分の絶対値
fig = plt.figure(figsize=figsize)
plt.imshow(np.abs(bg_sblx), cmap='gray')
plt.colorbar()
plt.title("Sobel X of Background")
plt.show()

# エッジ部分の2値化
edge_thresh = 2000
bg_sblx_bin = np.abs(bg_sblx) > edge_thresh
bg_sblx_bin = bg_sblx_bin.astype(np.uint8) * 255
fig = plt.figure(figsize=figsize)
plt.imshow(bg_sblx_bin, cmap='gray')
plt.colorbar()
plt.title("Binary Sobel X of Background")
plt.show()

# エッジ部分の2値化後に膨張処理
kernel = np.ones((3, 3), np.uint8)
bg_sblx_bin_dil = cv2.dilate(bg_sblx_bin, kernel, iterations=2)
fig = plt.figure(figsize=figsize)
plt.imshow(bg_sblx_bin_dil, cmap='gray')
plt.title("Dilated Binary Sobel X of Background")
plt.show()

# この2値画像をマスクとする
mask = bg_sblx_bin_dil.astype(bool)

# 膨張後の画像に key_boundaries をカラーで重ねて確認
fig = plt.figure(figsize=figsize)
plt.imshow(bg_sblx_bin_dil, cmap='gray')
for x in key_boundaries:
    xd = x - x_offset
    if 0 <= xd < (xlim[1] - xlim[0]):
        plt.axvline(x=xd, color='red')
plt.title("Key Boundaries on Dilated Binary Sobel X of Background")
plt.show()

# %%
# float にしたうえで引き算
# y軸は画像表示用に反転
USE_RED_BGSUB = True  # Rチャネルを使う...指が255に近くなる (なおcv2でjpeg readはBGR順)
# USE_RED_BGSUB = False  # もしくは gray変換 
ch = 2
fig = plt.figure(figsize=figsize)
if USE_RED_BGSUB:
    diffR_on_bg = (img_on - bg)[:, :, ch]
else:
    diffR_on_bg = cv2.cvtColor(img_on.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
plt.imshow(diffR_on_bg, 'gray')
plt.colorbar()
plt.title("img_on - bg")
plt.show()

fig = plt.figure(figsize=figsize)
if USE_RED_BGSUB:
    diffR_off_bg = (img_off - bg)[:, :, ch]
else:
    diffR_off_bg = cv2.cvtColor(img_off.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
plt.imshow(diffR_off_bg, 'gray')
plt.colorbar()
plt.title("img_off - bg")
plt.show()

fig = plt.figure(figsize=figsize)
if USE_RED_BGSUB:
    diffR_on_off = (img_on - img_off)[:, :, ch]
else:
    diffR_on_off = cv2.cvtColor(img_on.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(img_off.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
plt.imshow(diffR_on_off, 'gray')
plt.colorbar()
plt.title("img_on - img_off")
plt.show()

# %%
# diffに対してマスク白部分に対応するピクセルを赤色alphaブレンド表示
CHECK_ON_DIFF = False
if CHECK_ON_DIFF:
    diff_copy = diffR_on_bg.copy()  # bgとの差分を試す
    title_str = "img_on - bg"
else:
    diff_copy = diffR_off_bg.copy()  # bgとの差分を試す
    title_str = "img_off - bg"
# diff_copy = diffR_on_off.copy()  # on-off 差分でも試す
# [-255, 255] -> [0, 255]
diff_copy = (diff_copy + 255) / 2
# カラーへ変換
diff_color = cv2.cvtColor(diff_copy.astype(np.uint8), cv2.COLOR_GRAY2BGR)
# マスク部分を赤くする
mask_color = np.zeros_like(diff_color, dtype=np.uint8)
mask_color[mask] = [255, 0, 0]

# alphaブレンド
alpha = 0.3
diff_masked = cv2.addWeighted(diff_color, 1 - alpha, mask_color, alpha, 0)


fig = plt.figure(figsize=figsize)
plt.imshow(diff_masked)
for x in key_boundaries:
    xd = x - x_offset
    if 0 <= xd < (xlim[1] - xlim[0]):
        plt.axvline(x=xd, color='yellow')
plt.title(f"Masked diff ({title_str})")
plt.show()

# %%
# --- 垂直方向平均投影プロファイル -------------------------------
# 背景差分にて鍵盤ROI内で垂直方向に平均を取ってみる
for label, diff_img in [('off', diffR_off_bg), ('on', diffR_on_bg), ]:
    fig = plt.figure(figsize=figsize)
    diff_mean_vert = np.mean(diff_img, axis=0)
    plt.plot(diff_mean_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Mean of diff (img_{label} - bg)")
    plt.ylim(-255, 255)
    plt.show()

# %%
# オリジナル画像で垂直方向に平均を取ってみる
for label, img in [('off', img_off), ('on', img_on), ]:
    fig = plt.figure(figsize=figsize)
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_mean_vert = np.mean(img_gray, axis=0)
    plt.plot(img_mean_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Mean of img_{label} Gray")
    plt.ylim(0, 255)
    plt.show()


# %%
# --- 垂直方向最小値投影プロファイル -------------------------------
# 背景差分にて鍵盤ROI内で垂直方向に最小値を取ってみる
for label, diff_img in [('off', diffR_off_bg), ('on', diffR_on_bg), ]:
    fig = plt.figure(figsize=figsize)
    diff_min_vert = np.min(diff_img, axis=0)
    plt.plot(diff_min_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Min of diff (img_{label} - bg)")
    plt.ylim(-255, 255)
    plt.show()

# %%
# オリジナル画像で垂直方向に最小値を取ってみる
for label, img in [('off', img_off), ('on', img_on), ]:
    fig = plt.figure(figsize=figsize)
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # img_gray = img[:, :, 0].astype(np.float32)
    img_min_vert = np.min(img_gray, axis=0)
    plt.plot(img_min_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Min of img_{label} Gray")
    # plt.xlim(125, 150)
    plt.ylim(0, 255)
    plt.show()


# %%
# --- さらに非線形変換 ---
# 背景差分にて垂直方向に最小値を取り非線形変換（暗さ強調）
for label, diff_img in [('off', diffR_off_bg), ('on', diffR_on_bg), ]:
    fig = plt.figure(figsize=figsize)
    diff_min_vert = np.min( np.clip(diff_img, -255, 0), axis=0)  # 暗くなった方奥だけで見る
    diff_min_vert_nl = (diff_min_vert / 255) ** 4  # 非線形変換で暗さを強調
    plt.plot(diff_min_vert_nl)
    # plt.plot(diff_min_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Min of diff (img_{label} - bg) with Nonlinear Transform")
    plt.ylim(0, 0.5)
    plt.show()


# %%
# オリジナル画像で垂直方向に最小値を取り非線形変換（暗さ強調）
for label, img in [('off', img_off), ('on', img_on), ]:
    fig = plt.figure(figsize=figsize)
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # img_gray = img[:, :, 0].astype(np.float32)
    img_min_vert = (1 - np.min(img_gray, axis=0)/255) ** 4  # 非線形変換で暗さを強調
    plt.plot(img_min_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Min of img_{label} Gray with Nonlinear Transform")
    # plt.xlim(125, 150)
    plt.ylim(0, 1)
    plt.show()


# --- 彩度の高い領域を先に白にしてから垂直方向最小値投影プロファイル -------------------------------
# %%
# saturation の画像としての可視化
img_on_hsv = cv2.cvtColor(img_on.astype(np.uint8), cv2.COLOR_BGR2HSV)
saturation = img_on_hsv[:, :, 1]
# saturation = img_on[:, :, 1]
fig = plt.figure(figsize=figsize)
plt.imshow(saturation, clim=(0, 100), cmap='gray')
plt.colorbar()
plt.title("Saturation of img_on")
plt.show()

# saturationをヒストグラムで確認
fig = plt.figure(figsize=(6,4))
plt.hist(saturation.ravel(), bins=256, range=(0, 256))
plt.title("Histogram of Saturation")
plt.show()

# %%
# 彩度が高い部分を背景にする
saturation_thresh = 30
saturation_mask = saturation > saturation_thresh
img_on_saturationmasked = img_on.copy()
img_on_saturationmasked[saturation_mask] = bg[saturation_mask]  # 手の部分を背景で置換
fig = plt.figure(figsize=figsize)
plt.imshow(img_on_saturationmasked.astype(np.uint8))
plt.title("img_on with High Saturation Areas Masked to White")
plt.show()

# %%
# 彩度マスク後の背景差分画像で垂直方向に最小値を取ってみる
for label, img in [('off', img_off), ('on', img_on), ]:
    fig = plt.figure(figsize=figsize)
    img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1]
    saturation_mask = saturation > saturation_thresh
    diff_saturationmasked = (img - bg)[:, :, 2].copy()
    diff_saturationmasked[saturation_mask] = 0  # 手の部分を0で置換
    diff_saturationmasked_min_vert = np.min(diff_saturationmasked, axis=0)
    plt.plot(diff_saturationmasked_min_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Min of img_on with High Saturation Areas Masked")
    plt.ylim(-255, 255)
    plt.show()

# %%
# 彩度マスク後の画像で垂直方向に最小値を取ってみる
for label, img in [('off', img_off), ('on', img_on), ]:
    fig = plt.figure(figsize=figsize)
    img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1]
    saturation_mask = saturation > saturation_thresh
    img_saturationmasked = img.copy()
    img_saturationmasked[saturation_mask] = bg[saturation_mask]  # 手の部分を背景で置換
    img_saturationmasked_gray = cv2.cvtColor(img_saturationmasked.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_saturationmasked_min_vert = np.min(img_saturationmasked_gray, axis=0)
    plt.plot(img_saturationmasked_min_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Min of img_on with High Saturation Areas Masked")
    plt.ylim(0, 255)
    plt.show()

# %%
# 彩度では指のみの判定は難しい・・・
# 水平Sobelの垂直方向投影でプロファイルの重みづけできるか？
for label, img in [('off', img_off), ('on', img_on), ]:
    fig = plt.figure(figsize=figsize)
    sblx_img = cv2.Sobel(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=5)
    img_mean_vert = np.mean(sblx_img, axis=0)
    plt.plot(img_mean_vert)
    for x in key_boundaries:
        xd = x - x_offset
        if 0 <= xd < (xlim[1] - xlim[0]):
            plt.axvline(x=xd, color='red', linestyle='--', linewidth=1.0)
    plt.title(f"Vertical Mean of img_{label} Sobel-x")
    plt.show()
