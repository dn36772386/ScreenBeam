# カメラ動作確認・デバッグツール
# Windows/Mac/Linux対応

import cv2
import numpy as np
import time
from datetime import datetime

def list_available_cameras():
    """利用可能なカメラをリストアップ"""
    print("=== 利用可能なカメラを検索中 ===")
    available_cameras = []
    
    # 最大10個のカメラインデックスをチェック
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # カメラ情報取得
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available_cameras.append({
                'index': i,
                'width': width,
                'height': height,
                'fps': fps
            })
            
            print(f"カメラ {i}: {int(width)}x{int(height)} @ {fps}fps")
            cap.release()
    
    if not available_cameras:
        print("カメラが見つかりません！")
    
    return available_cameras

def test_camera_basic(camera_index=0):
    """基本的なカメラ動作テスト"""
    print(f"\n=== カメラ {camera_index} の基本テスト ===")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"エラー: カメラ {camera_index} を開けません")
        return False
    
    # カメラプロパティ表示
    print("\nカメラプロパティ:")
    props = {
        'Width': cv2.CAP_PROP_FRAME_WIDTH,
        'Height': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'Brightness': cv2.CAP_PROP_BRIGHTNESS,
        'Contrast': cv2.CAP_PROP_CONTRAST,
        'Saturation': cv2.CAP_PROP_SATURATION,
        'Exposure': cv2.CAP_PROP_EXPOSURE,
        'Gain': cv2.CAP_PROP_GAIN,
    }
    
    for name, prop in props.items():
        value = cap.get(prop)
        if value != -1:
            print(f"  {name}: {value}")
    
    # フレーム取得テスト
    print("\nフレーム取得テスト中... (ESCで終了)")
    
    frame_count = 0
    start_time = time.time()
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("警告: フレームを取得できません")
            continue
        
        frame_count += 1
        
        # FPS計算
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            fps_list.append(current_fps)
            fps_text = f"FPS: {current_fps:.1f}"
        else:
            fps_text = f"FPS: {fps_list[-1]:.1f}" if fps_list else "FPS: 計測中"
        
        # フレーム情報オーバーレイ
        info_text = [
            f"Camera: {camera_index}",
            f"Size: {frame.shape[1]}x{frame.shape[0]}",
            fps_text,
            f"Frame: {frame_count}",
            "Press ESC to exit"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # ヒストグラム表示（明るさ分布）
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], 
                           [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.max() * 100  # 正規化
        
        # ヒストグラムを画面右側に描画
        hist_width = 256
        hist_height = 100
        hist_img = np.zeros((hist_height, hist_width), dtype=np.uint8)
        
        for i in range(256):
            cv2.line(hist_img, (i, hist_height), 
                    (i, hist_height - int(hist[i])), 255, 1)
        
        # ヒストグラムを本画像に重ねる
        frame[10:10+hist_height, frame.shape[1]-hist_width-10:frame.shape[1]-10, 1] = hist_img
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # スクリーンショット
            filename = f"camera_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            print(f"スクリーンショット保存: {filename}")
    
    # 統計情報
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\n平均FPS: {avg_fps:.1f}")
        print(f"最小FPS: {min(fps_list):.1f}")
        print(f"最大FPS: {max(fps_list):.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_camera_advanced(camera_index=0):
    """高度なカメラテスト（60fps、露出制御など）"""
    print(f"\n=== カメラ {camera_index} の高度なテスト ===")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"エラー: カメラ {camera_index} を開けません")
        return False
    
    # 60fps設定を試行
    print("\n60fps設定を試行中...")
    cap.set(cv2.CAP_PROP_FPS, 60)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"要求FPS: 60, 実際のFPS: {actual_fps}")
    
    # 解像度設定
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (640, 480)
    ]
    
    print("\n解像度テスト:")
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"  {width}x{height} -> {int(actual_width)}x{int(actual_height)}")
    
    # 自動露出OFF設定（カメラによってはサポートされない）
    print("\n露出制御テスト:")
    
    # 自動露出OFF
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual, 0.75 = auto
    print(f"自動露出設定: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    
    # 手動露出値設定
    exposure_values = [-7, -5, -3, -1]
    for exp in exposure_values:
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"  露出値 {exp} -> {actual_exp}")
    
    cap.release()
    return True

def test_optical_receiver_camera(camera_index=0):
    """光学通信受信用のカメラ設定テスト"""
    print(f"\n=== 光学通信用カメラ設定テスト ===")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("エラー: カメラを開けません")
        return False
    
    # 光学通信用の推奨設定
    print("\n光学通信用設定を適用中...")
    
    # バッファサイズを最小に（レイテンシ削減）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 60fps設定
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 解像度設定（1280x720推奨）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 自動露出OFF
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    # 露出時間を短く
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    
    # ゲイン調整
    cap.set(cv2.CAP_PROP_GAIN, 0)
    
    print("\n適用された設定:")
    print(f"  解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  バッファサイズ: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    print(f"  露出: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"  ゲイン: {cap.get(cv2.CAP_PROP_GAIN)}")
    
    # リアルタイムプレビュー
    print("\nプレビュー表示中... (ESCで終了)")
    print("'a' - 自動露出切替")
    print("'+'/'-' - 露出値調整")
    print("'g'/'h' - ゲイン調整")
    
    auto_exposure = False
    exposure_value = -5
    gain_value = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # グレースケール変換（光学通信では輝度のみ使用）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 10x10セルグリッドを描画
        cell_size = 10
        for x in range(0, frame.shape[1], cell_size):
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 0), 1)
        for y in range(0, frame.shape[0], cell_size):
            cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 1)
        
        # 設定情報表示
        info = [
            f"Auto Exposure: {auto_exposure}",
            f"Exposure: {exposure_value}",
            f"Gain: {gain_value}",
            f"Mean Brightness: {np.mean(gray):.1f}",
            f"Std Dev: {np.std(gray):.1f}"
        ]
        
        y_offset = 30
        for text in info:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        cv2.imshow('Optical Receiver Camera Test', frame)
        cv2.imshow('Grayscale', gray)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('a'):  # 自動露出切替
            auto_exposure = not auto_exposure
            if auto_exposure:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        elif key == ord('+'):  # 露出値増加
            exposure_value = min(exposure_value + 1, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        elif key == ord('-'):  # 露出値減少
            exposure_value = max(exposure_value - 1, -10)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        elif key == ord('g'):  # ゲイン増加
            gain_value = min(gain_value + 10, 100)
            cap.set(cv2.CAP_PROP_GAIN, gain_value)
        elif key == ord('h'):  # ゲイン減少
            gain_value = max(gain_value - 10, 0)
            cap.set(cv2.CAP_PROP_GAIN, gain_value)
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    """メインメニュー"""
    while True:
        print("\n=== カメラデバッグツール ===")
        print("1. 利用可能なカメラをリスト表示")
        print("2. 基本的なカメラテスト")
        print("3. 高度なカメラテスト（60fps、露出制御）")
        print("4. 光学通信用カメラ設定テスト")
        print("5. 終了")
        
        choice = input("\n選択してください (1-5): ")
        
        if choice == '1':
            list_available_cameras()
        elif choice == '2':
            camera_index = int(input("カメラインデックス (通常は0): "))
            test_camera_basic(camera_index)
        elif choice == '3':
            camera_index = int(input("カメラインデックス (通常は0): "))
            test_camera_advanced(camera_index)
        elif choice == '4':
            camera_index = int(input("カメラインデックス (通常は0): "))
            test_optical_receiver_camera(camera_index)
        elif choice == '5':
            print("終了します")
            break
        else:
            print("無効な選択です")

if __name__ == "__main__":
    main()