# 最もシンプルなカメラテスト - 問題を特定するため

import cv2
import numpy as np

print("=== 最小限のカメラテスト ===\n")

# カメラ1をDirectShowで開く
print("カメラ1をDirectShowで開いています...")
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

if not cap.isOpened():
    print("DirectShow失敗、通常の方法で試します...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("カメラを開けません")
    exit()

print("カメラを開きました\n")

# 設定前の状態を表示
print("現在の設定:")
print(f"  幅: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"  高さ: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"  バックエンド: {cap.getBackendName()}")

# 最初の10フレームの状態を確認
print("\n最初の10フレームをチェック:")
for i in range(10):
    ret, frame = cap.read()
    if ret and frame is not None:
        mean = np.mean(frame)
        std = np.std(frame)
        min_val = np.min(frame)
        max_val = np.max(frame)
        print(f"フレーム{i}: 平均={mean:.1f}, 標準偏差={std:.1f}, 最小={min_val}, 最大={max_val}")
        
        # 1枚保存
        if i == 5:
            cv2.imwrite(f"test_frame_{i}.jpg", frame)
            print(f"  → test_frame_{i}.jpg に保存")
    else:
        print(f"フレーム{i}: 取得失敗")

# 簡単な表示
print("\n映像を表示します（何か映っていますか？）")
print("ESCで終了")

cv2.namedWindow('Test', cv2.WINDOW_NORMAL)

for i in range(300):  # 10秒間（30fps想定）
    ret, frame = cap.read()
    if ret:
        # 情報追加
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 中央に十字線
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2-50, h//2), (w//2+50, h//2), (0, 255, 0), 2)
        cv2.line(frame, (w//2, h//2-50), (w//2, h//2+50), (0, 255, 0), 2)
        
        cv2.imshow('Test', frame)
        
        if cv2.waitKey(33) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

print("\nテスト終了")