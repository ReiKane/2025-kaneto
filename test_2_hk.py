# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from enum import Enum

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# 背景読み込み
bg_orig = cv2.imread("background_2.jpg")
H, W, _ = bg_orig.shape

# GRAY or red_channel
# COLOR_CH = -1  # -1: GRAY, 2: R channel
COLOR_CH = 2  # -1: GRAY, 2: R channel
if COLOR_CH < 0:
    bg = cv2.cvtColor(bg_orig, cv2.COLOR_BGR2GRAY)
else:
    bg = bg_orig[:, :, COLOR_CH]

bg = cv2.GaussianBlur(bg, (3,3), 0)

USE_TSNE = False

#エッジの座標(別のコードで取得)
key_boundaries = [29, 66, 103, 141, 179, 216, 254, 292, 330, 368, 405, 443, 480, 517, 554, 592, 629, 665, 702, 739, 775, 812, 849, 885, 922, 958, 995, 1032, 1069, 1105, 1142, 1178, 1215, 1251, 1288, 1325, 1362, 1398, 1435, 1472, 1509, 1546, 1583, 1621, 1658, 1695, 1733, 1770, 1807, 1844, 1880, 1903]

n_keys = len(key_boundaries) - 1
print(f'Number of keys detected: {n_keys}')

# 鍵盤のy座標範囲を指定
# key_ylim = [830, 1060]  # 白鍵と黒鍵両方
key_ylim = [987, 1060]  # 白鍵のみ

# モード定義
class Mode(Enum):
    THRESHOLDING = 0  # 閾値のみで判定
    TRAINING = 1  # GMM学習(+PCA) (1次元なら自動で閾値決めているようなもの)
    CLASSIFICATION = 2 # GMM利用判定(+PCA)
    CHECK_ONLY = 3  # 特徴量確認のみ


# ===================================================================
# 各種設定
mode = Mode.THRESHOLDING
# mode = Mode.TRAINING
# mode = Mode.CLASSIFICATION  # 事前に mode = Mode.TRAINING で学習しておく必要あり
# mode = Mode.CHECK_ONLY

# --- for THRESHOLDING ---
threshold = 0.02  # 閾値判定用

# --- for TRAINING / CLASSIFICATION ---
# どちらの特徴量を使うか
f_use = 1  # 0 or 1

# PCA次元
pca_components = 1  # 用いるPCA次元数

# GMMコンポーネント数（混合数）
gmm_components = 2  # 押下・非押下の2クラス想定（3-6でも良いかも）

# SATURATION_THRES = 20  # 彩度閾値（使用しない）

# 処理するフレーム範囲
# tlim = [0, 800]  # 前半避けた方がいい
# tlim = [0, 300]  # 前半避けた方がいい
tlim = [2980, 3734]

# ===================================================================

# %%
if mode == Mode.CLASSIFICATION:
    # 既存モデル読み込み
    model_fname = f'models_f{f_use}_pca{pca_components}_gmm{gmm_components}_frame{tlim[0]}-{tlim[1]}.pkl'
    # model_fname = f'models_frame0-800.pkl'
    with open(model_fname, 'rb') as f:
        models = pickle.load(f)

# 動画読み込み
cap = cv2.VideoCapture("../piano_test_5.mp4")
if cap.isOpened() == False:
    print("Error opening video file")
else:
    print(f"Video file opened successfully: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames found.")

cap.set(cv2.CAP_PROP_POS_FRAMES, tlim[0])
H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f'Video frame size: {W} x {H}')

# n_keys x n_frames x d_k(マスク内のピクセル数)
# 2通りの特徴量を保存できるようにしておく（いろいろ試すため）
feature0_log = [ [] for _ in range(n_keys) ]  # n_keys個のリストを作成
feature1_log = [ [] for _ in range(n_keys) ]
feature_log = [ feature0_log, feature1_log ]

press_log = []

while cap.isOpened() and (cap.get(cv2.CAP_PROP_POS_FRAMES) < tlim[1]):
    print(f'Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}')
    ret, frame = cap.read()
    if ret:
        #前処理・差分取得
        if COLOR_CH < 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame[:, :, COLOR_CH]
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        # diff = cv2.absdiff(gray, bg)
        # _, diff_bin = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        # roi = diff_bin[987:1054, :]#検出に使う範囲

        # [HK] 差分をsignedのままま取得
        diff_float = gray.astype(np.float32) - bg.astype(np.float32)
        # saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        # saturation_mask = saturation > SATURATION_THRES
        # diff_satmasked = diff_float.copy()
        # diff_satmasked[saturation_mask] = 0  # 彩度の高い部分（手を想定）を0で置換
        # img_satmasked = frame.copy()
        # img_satmasked[saturation_mask] = bg_orig[saturation_mask]  # 彩度の高い部分を背景で置換

        press_states = []
        for k in range(len(key_boundaries) - 1):
            x1, x2 = key_boundaries[k], key_boundaries[k + 1]
            # key_roi_1 = roi[:, x1:x1+1]  #左端
            # key_roi_2 = roi[:, x2:x2+1]  #右端
            # mean_val = np.mean(key_roi_1 + key_roi_2)
            # pressed = mean_val > threshold
            
            # [HK] 垂直方向最小値投影
            # feature0 = np.min( diff_satmasked[ key_ylim[0]:key_ylim[1], x1:x2 ], axis=0 )  # 彩度マスク後の垂直方向最小値投影            
            # feature1 = np.min( img_satmasked[ key_ylim[0]:key_ylim[1], x1:x2 , 2], axis=0 )  # 彩度マスク後の垂直方向最小値投影（Rチャネル）
            # feature0 = np.min( diff_float[ key_ylim[0]:key_ylim[1], x1:x2 ], axis=0 )  # 垂直方向最小値投影
            # feature1 = (1 - np.min( frame[ key_ylim[0]:key_ylim[1], x1:x2 , 2], axis=0 )/255) ** 4  # 垂直方向最小値投影（Rチャネル）
            diff_clip_vmin = np.min(np.clip( diff_float[ key_ylim[0]:key_ylim[1], x1:x2 ], -255, 0), axis=0 )
            diff_clip_vmin_nl = (diff_clip_vmin / 255) ** 4  # 非線形変換
            feature0 = diff_clip_vmin_nl  # x方向全て
            feature1 = np.concatenate([ diff_clip_vmin_nl[:5], diff_clip_vmin_nl[-5:] ])  # x方向両端5ピクセルずつ

            feature_log[0][k].append(feature0)
            feature_log[1][k].append(feature1)

            
            feature = feature_log[f_use][k][-1]  # どの特徴量を使うか
            
            if mode == Mode.CLASSIFICATION:
                pca, gmm = models[k]    
                feature_array = np.array(feature).reshape(1, -1)
                feature_pca = pca.transform(feature_array)
                label = gmm.predict(feature_pca)[0]
                # print(f'Key {k}: GMM label = {label}')
                pressed = (label != 0)  # ラベル0をOFF，それ以外をONと仮定
            elif mode == Mode.THRESHOLDING:
                pressed = np.mean(feature) > threshold  # 閾値判定
            
            # 押されている鍵盤を赤枠で表示
            if mode == Mode.CLASSIFICATION or mode == Mode.THRESHOLDING:
                press_states.append(pressed)
                color = (0, 0, 255) if pressed else (255, 255, 255)
                cv2.rectangle(frame, (x1, key_ylim[0]), (x2, key_ylim[1]), color, 2)

        #ログ記録(今後に向けて出せるようにしています)
        if mode == Mode.CLASSIFICATION or mode == Mode.THRESHOLDING:
            press_log.append(press_states)
            #表示
            frame = cv2.resize(frame, (W//2, H//2))
            cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    else:
        break

# %%
cap.release()
cv2.destroyAllWindows()

# %%
# feature_logの各キーごとの配列をnumpy配列に変換
for f in feature_log:
    for k in range(n_keys):
        f[k] = np.array(f[k])  # n_frames x d_k
        print(f'Key {k}: feature shape = {f[k].shape}')

# GMMで教師なし学習（クラスタリング）
if mode == Mode.TRAINING:
    models = []
    for k in range(n_keys):
        key_features = feature_log[f_use][k]
        pca = PCA(n_components=pca_components)
        key_features_pca = pca.fit_transform(key_features)
        gmm = GaussianMixture(n_components=gmm_components, random_state=42)
        gmm.fit(key_features_pca)
        # frame=0 が含まれるラベルを押下していないラベル(0)に設定
        first_frame_feature = key_features[0].reshape(1, -1)
        first_frame_pca = pca.transform(first_frame_feature)
        first_frame_label = gmm.predict(first_frame_pca)[0]
        print(f'Key {k}: First frame label = {first_frame_label}')
        if first_frame_label != 0:
            # ラベル入れ替え
            print(f'Key {k}: Swapping GMM labels to set first frame as label 0')
            gmm.means_[[0, first_frame_label]] = gmm.means_[[first_frame_label, 0]]
            gmm.covariances_[[0, first_frame_label]] = gmm.covariances_[[first_frame_label, 0]]
            gmm.weights_[[0, first_frame_label]] = gmm.weights_[[first_frame_label, 0]]
            if hasattr(gmm, 'precisions_'):
                gmm.precisions_[[0, first_frame_label]] = gmm.precisions_[[first_frame_label, 0]]
        models.append((pca, gmm))
    # モデル保存
    model_fname = f'models_f{f_use}_pca{pca_components}_gmm{gmm_components}_frame{tlim[0]}-{tlim[1]}.pkl'
    with open(model_fname, 'wb') as f:
        pickle.dump(models, f)

# %%
if mode == Mode.CHECK_ONLY or mode == Mode.TRAINING:
    # test one key
    # key_x = 885
    # key_id = key_boundaries.index(key_x)
    # key_id = 23  # 23:D4
    key_id = 21  # 20:A3
    print(f'Key ID: {key_id}, boundaries = ({key_boundaries[key_id]}, {key_boundaries[key_id+1]})')


    # どの特徴量を使うか
    key_features = feature_log[f_use][key_id]  # n_frames x d_k
    print('Feature log shape:', key_features.shape)

    # imshow
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(key_features, aspect='auto', cmap='jet')
    # y軸: frame番号は tlim[0]始まりで100frameごと
    y_ticks = np.arange(0, key_features.shape[0], 100)
    y_ticklabels = [str(i + tlim[0]) for i in y_ticks]
    plt.yticks(y_ticks, y_ticklabels)
    plt.colorbar(label='Feature Value')
    plt.title(f'Feature Map for Key {key_id}')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Frame Index')
    plt.show()


    # まずは平均値のヒストグラムを確認
    mean_values = np.mean(key_features, axis=1)  # n_frames
    plt.figure(figsize=(10, 4))
    plt.hist(mean_values, bins=30, color='blue', alpha=0.7)
    plt.title(f'Histogram of Mean Values for Key {key_id}')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    plt.show()

    if USE_TSNE:
        # まずは t-SNE で2次元プロット (frame番号も表示)
        tsne = TSNE(n_components=2, random_state=42)
        # tsne = TSNE(n_components=2)
        key_features_tsne = tsne.fit_transform(key_features)
        plt.figure(figsize=(8, 8))
        plt.scatter(key_features_tsne[:, 0], key_features_tsne[:, 1], c='blue', alpha=0.6)
        for i in range(key_features_tsne.shape[0]):
            plt.text(key_features_tsne[i, 0], key_features_tsne[i, 1], str(i + tlim[0]), fontsize=10, alpha=0.7)
        plt.title(f't-SNE Projection of Features for Key {key_id}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    # PCAで主成分分析
    pca = PCA(n_components=3)  # TRAININGの設定とは独立に試すので注意！
    key_features_pca = pca.fit_transform(key_features)
    # 寄与率確認
    print(f'Total explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
    # 
    plt.figure(figsize=(8, 8))
    plt.scatter(key_features_pca[:, 0], key_features_pca[:, 1], c='green', alpha=0.6)
    for i in range(key_features_pca.shape[0]):
        plt.text(key_features_pca[i, 0], key_features_pca[i, 1], str(i + tlim[0]), fontsize=10, alpha=0.7)
    plt.title(f'PCA Projection of Features for Key {key_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


    # GMM (k=2) でクラスタリング
    gmm = GaussianMixture(n_components=2, random_state=42)  # TRAININGの設定とは独立に試すので注意！
    gmm.fit(key_features)
    labels = gmm.predict(key_features)
    plt.figure(figsize=(8, 8))
    plt.scatter(key_features_pca[:, 0], key_features_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    for i in range(key_features_pca.shape[0]):
        plt.text(key_features_pca[i, 0], key_features_pca[i, 1], str(i + tlim[0]), fontsize=10, alpha=0.7)
    plt.title(f'GMM Clustering of Features for Key {key_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    if USE_TSNE:
        # クラスタリング結果を t-SNEでも表示
        plt.figure(figsize=(8, 8))
        plt.scatter(key_features_tsne[:, 0], key_features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
        for i in range(key_features_tsne.shape[0]):
            plt.text(key_features_tsne[i, 0], key_features_tsne[i, 1], str(i + tlim[0]), fontsize=10, alpha=0.7)
        plt.title(f'GMM Clustering on t-SNE Projection for Key {key_id}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()



# %%
