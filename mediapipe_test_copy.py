import cv2
import mediapipe as mp

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# カメラからの映像をキャプチャ
cap = cv2.VideoCapture("../test_kamui.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像をRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手のランドマークの検出
    results = hands.process(image)

    # 検出結果がある場合、ランドマークを描画し、右手の位置を取得
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('result', frame)


# リソースの解放
cap.release()
cv2.destroyAllWindows()

