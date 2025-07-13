# 最小限のカメラ動作確認コード

import cv2
import numpy as np

# カメラを開く（0は通常内蔵カメラ、1は外付けカメラ）
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"エラー: カメラ {camera_index} を開けません")
    print("\n試してみてください:")
    print("1. 別のカメラインデックス（1, 2など）を試す")
    print("2. Webカメラが他のアプリで使用されていないか確認")
    print("3. カメラのドライバーが正しくインストールされているか確認")
    exit()

print("カメラが正常に開きました！")
print(f"解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print("\nESCキーで終了します")

frame_count = 0

while True:
    # フレームを取得
    ret, frame = cap.read()
    
    if ret:
        frame_count += 1
        
        # フレーム番号を表示
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 画面中央に十字線を描画
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)
        
        # 表示
        cv2.imshow('Camera Check', frame)
        
        # 's'キーでスクリーンショット保存
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            cv2.imwrite(f'camera_test_{frame_count}.png', frame)
            print(f"スクリーンショット保存: camera_test_{frame_count}.png")
    else:
        print("警告: フレームを取得できませんでした")

# 後片付け
cap.release()
cv2.destroyAllWindows()

print(f"\n合計 {frame_count} フレームを取得しました")