# AIRCODE風 光学通信受信側実装
# Windows 11 Python 3.12+

import numpy as np
import cv2
import pyaudio
import struct
import time
from reedsolo import RSCodec, ReedSolomonError
import zlib
from threading import Thread, Lock
import queue
from collections import deque
from datetime import datetime

class OpticalReceiver:
    def __init__(self):
        # カメラ設定
        self.camera_fps = 60
        self.camera_index = 0
        
        # セル設定（送信側と同一）
        self.cell_size = 10
        self.base_brightness = 128
        self.modulation_depth = 5
        
        # パケット設定
        self.packet_size = 1024
        self.header_size = 8
        
        # 誤り訂正
        self.rs = RSCodec(10)
        
        # 音声同期設定
        self.sync_freq = 19000
        self.sync_threshold = 0.3
        
        # バッファ
        self.frame_buffer = deque(maxlen=10)
        self.bit_buffer = []
        self.packet_buffer = []
        self.received_packets = {}
        
        # 状態管理
        self.lock = Lock()
        self.receiving = False
        self.file_size = None
        
        # プレビュー設定
        self.show_preview = True
        self.preview_scale = 1.0
        
    def init_camera(self):
        """カメラ初期化（シンプル版）"""
        print("カメラを初期化中...")
        
        # シンプルにDirectShowで開く（Windowsで最も安定）
        print(f"カメラ {self.camera_index} をDirectShowで開きます...")
        self.cap = cv2.VideoCapture(self.camera_index + cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print("DirectShow失敗、標準方法で試します...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            print("✗ カメラを開けません")
            return False
            
        print("✓ カメラを開きました")
        
        # 現在の設定を表示
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        backend = self.cap.getBackendName()
        
        print(f"  解像度: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  バックエンド: {backend}")
        
        # 最初の数フレームを確認
        print("  映像確認中...")
        for i in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                mean = np.mean(frame)
                std = np.std(frame)
                print(f"    フレーム{i}: 平均={mean:.1f}, 標準偏差={std:.1f}")
                if std > 5:  # 映像がある
                    print("✓ 正常な映像を確認")
                    return True
        
        print("✗ 正常な映像が取得できません")
        return False
        
    def init_audio(self):
        """音声入力初期化"""
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        
    def detect_sync_pulse(self, audio_data):
        """同期パルス検出"""
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # FFTで周波数成分解析
        fft = np.fft.rfft(audio_array)
        freqs = np.fft.rfftfreq(len(audio_array), 1/44100)
        
        # 同期周波数付近のパワーチェック
        sync_idx = np.argmin(np.abs(freqs - self.sync_freq))
        sync_power = np.abs(fft[sync_idx])
        
        return sync_power > self.sync_threshold
        
    def detect_screen(self, frame):
        """スクリーン領域検出と透視変換"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # 最大の輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 輪郭を矩形に近似
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            return None, None
            
        # 頂点を並び替え（左上、右上、右下、左下）
        points = approx.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        
        # 透視変換行列計算
        dst_width = 1920
        dst_height = 1080
        dst = np.array([[0, 0], [dst_width, 0], 
                       [dst_width, dst_height], [0, dst_height]], 
                      dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, matrix, (dst_width, dst_height))
        
        return warped, rect
        
    def extract_bits_from_frame(self, frame, prev_frame=None):
        """フレームからビット抽出"""
        if frame is None:
            return []
            
        bits = []
        
        # グリッド分割
        grid_height = frame.shape[0] // self.cell_size
        grid_width = frame.shape[1] // self.cell_size
        
        # デバッグ: 最初の抽出時に情報表示
        if not hasattr(self, '_first_extraction_done'):
            print(f"ビット抽出開始: グリッドサイズ {grid_width}x{grid_height}")
            print(f"  セルサイズ: {self.cell_size}x{self.cell_size}")
            print(f"  合計セル数: {grid_width * grid_height}")
            self._first_extraction_done = True
        
        for y in range(grid_height):
            for x in range(grid_width):
                # セル領域切り出し
                y_start = y * self.cell_size
                y_end = (y + 1) * self.cell_size
                x_start = x * self.cell_size
                x_end = (x + 1) * self.cell_size
                
                cell = frame[y_start:y_end, x_start:x_end]
                
                # セルの平均輝度
                avg_brightness = np.mean(cell)
                
                # 変調検出（閾値を調整）
                if avg_brightness > self.base_brightness + 2:  # より緩い閾値
                    bits.append(1)
                else:
                    bits.append(0)
        
        # デバッグ: 定期的にビットパターンを表示
        if hasattr(self, '_extraction_count'):
            self._extraction_count += 1
        else:
            self._extraction_count = 1
            
        if self._extraction_count % 100 == 0:
            # 最初の100ビットのパターンを表示
            bit_sample = bits[:100] if len(bits) >= 100 else bits
            bit_string = ''.join(map(str, bit_sample[:50]))  # 最初の50ビット
            ones = sum(bit_sample)
            print(f"ビット抽出 #{self._extraction_count}: {ones}/{len(bit_sample)} が1")
            print(f"  サンプル: {bit_string}...")
                    
        return bits
        
    def bits_to_bytes(self, bits):
        """ビット列をバイト列に変換"""
        if len(bits) % 8 != 0:
            # 8の倍数に調整
            bits.extend([0] * (8 - len(bits) % 8))
            
        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_data.append(byte)
            
        return bytes(bytes_data)
        
    def decode_packet(self, data):
        """パケットデコード"""
        try:
            # Reed-Solomon復号
            decoded = self.rs.decode(data)[0]
            
            # CRCチェック
            payload = decoded[:-4]
            crc_received = struct.unpack('<I', decoded[-4:])[0]
            crc_calculated = zlib.crc32(payload) & 0xffffffff
            
            if crc_received != crc_calculated:
                return None, None
                
            # シーケンス番号取得
            seq_num = struct.unpack('<I', payload[:4])[0]
            data = payload[4:]
            
            return seq_num, data
            
        except (ReedSolomonError, struct.error):
            return None, None
            
    def process_packets(self):
        """受信パケット処理とファイル復元（簡易版）"""
        with self.lock:
            if len(self.received_packets) == 0:
                return None
                
            # デバッグ用: 受信したデータをそのまま結合
            print(f"パケット処理: {len(self.received_packets)} 個のパケット")
            
            # 簡易的にデータを結合
            file_data = bytearray()
            for i in range(len(self.received_packets)):
                if i in self.received_packets:
                    file_data.extend(self.received_packets[i])
                    
            # 最低限のサイズチェック
            if len(file_data) > 10:
                print(f"ファイルデータ準備完了: {len(file_data)} bytes")
                return bytes(file_data)
                
            return None
            
    def camera_worker(self):
        """カメラキャプチャワーカー（シンプル版）"""
        if not self.init_camera():
            print("カメラワーカー: 初期化失敗")
            self.receiving = False
            return
            
        print("カメラワーカー: 開始")
        
        # プレビュー用の統計情報
        frame_count = 0
        start_time = time.time()
        fps_list = []
        last_fps_time = time.time()
        
        # ビット抽出の視覚化用
        bit_visualization = None
        
        while self.receiving:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
                
            frame_count += 1
            
            # デバッグ: 最初の数フレームの状態を表示
            if frame_count <= 5:
                mean = np.mean(frame)
                std = np.std(frame)
                print(f"表示フレーム {frame_count}: 平均={mean:.1f}, 標準偏差={std:.1f}")
            
            display_frame = frame.copy()
            
            # FPS計算
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                fps_list.append(fps)
                last_fps_time = current_time
            
            # スクリーン検出と歪み補正
            warped, corners = self.detect_screen(frame)
            
            if warped is not None:
                # ビット抽出
                bits = self.extract_bits_from_frame(warped, None)
                
                with self.lock:
                    self.bit_buffer.extend(bits)
                    
                # スクリーン検出結果を描画
                if corners is not None:
                    # 検出された四角形を緑で描画
                    cv2.drawContours(display_frame, [corners.astype(int)], -1, (0, 255, 0), 3)
                    
                    # 各頂点に番号を表示
                    for i, point in enumerate(corners):
                        cv2.circle(display_frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, str(i+1), tuple(point.astype(int) + 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # ビット抽出の視覚化（小さいサムネイル）
                bit_vis_size = 200
                bit_visualization = self.create_bit_visualization(warped, bits, bit_vis_size)
            
            # 情報オーバーレイ
            with self.lock:
                bit_buffer_size = len(self.bit_buffer)
            self.draw_info_overlay(display_frame, frame_count, fps_list, 
                                 len(self.received_packets), bit_visualization, bit_buffer_size)
            
            # メインプレビュー表示（重要: 必ず表示する）
            if self.show_preview:
                cv2.imshow('Optical Receiver - Camera Preview', display_frame)
                
                # 歪み補正後の画像も表示
                if warped is not None:
                    warped_resized = cv2.resize(warped, (640, 360))
                    cv2.imshow('Warped Screen', warped_resized)
            
            # キー操作（必須: waitKeyがないと表示されない）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.receiving = False
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'receiver_screenshot_{timestamp}.png', display_frame)
                print(f"スクリーンショット保存: receiver_screenshot_{timestamp}.png")
            elif key == ord('p'):
                cv2.waitKey(0)
                
        print("カメラワーカー: 終了")
        self.cap.release()
        cv2.destroyAllWindows()
        
    def create_bit_visualization(self, warped_frame, bits, size):
        """ビット抽出結果の視覚化画像を作成"""
        if warped_frame is None or not bits:
            return None
            
        # グリッドサイズ計算
        grid_height = warped_frame.shape[0] // self.cell_size
        grid_width = warped_frame.shape[1] // self.cell_size
        
        # ビット値を画像化（白=1、黒=0）
        bit_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        bit_index = 0
        for y in range(grid_height):
            for x in range(grid_width):
                if bit_index < len(bits):
                    bit_image[y, x] = 255 if bits[bit_index] == 1 else 0
                    bit_index += 1
                    
        # リサイズして返す
        return cv2.resize(bit_image, (size, size), interpolation=cv2.INTER_NEAREST)
        
    def draw_info_overlay(self, frame, frame_count, fps_list, packet_count, bit_vis, bit_buffer_size):
        """情報オーバーレイを描画"""
        # 半透明の背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # テキスト情報
        current_fps = fps_list[-1] if fps_list else 0
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        
        info_lines = [
            f"Frame: {frame_count}",
            f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})",
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
            f"Packets received: {packet_count}",
            f"Bit buffer size: {bit_buffer_size}",
            "",
            "Controls:",
            "  'q' - Quit",
            "  's' - Screenshot",
            "  'p' - Pause"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25
            
        # ビット視覚化表示
        if bit_vis is not None:
            # 右上に表示
            vis_y = 10
            vis_x = frame.shape[1] - bit_vis.shape[1] - 10
            
            # カラー画像に変換（グレースケール→BGR）
            bit_vis_color = cv2.cvtColor(bit_vis, cv2.COLOR_GRAY2BGR)
            
            # 枠を追加
            cv2.rectangle(bit_vis_color, (0, 0), 
                        (bit_vis.shape[1]-1, bit_vis.shape[0]-1), (0, 255, 0), 2)
            
            # フレームに配置
            frame[vis_y:vis_y+bit_vis.shape[0], vis_x:vis_x+bit_vis.shape[1]] = bit_vis_color
            
            # ラベル追加
            cv2.putText(frame, "Extracted Bits", (vis_x, vis_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # ステータスバー
        status_text = "RECEIVING" if self.receiving else "STOPPED"
        status_color = (0, 255, 0) if self.receiving else (0, 0, 255)
        
        cv2.rectangle(frame, (0, frame.shape[0]-30), 
                     (frame.shape[1], frame.shape[0]), status_color, -1)
        cv2.putText(frame, f"Status: {status_text} | File size: {self.file_size or 'Unknown'}", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    def audio_worker(self):
        """音声同期ワーカー"""
        print("音声ワーカー: 開始")
        
        # 音声を使わず、タイマーベースで処理（デバッグ用）
        packet_timer = time.time()
        
        while self.receiving:
            try:
                # 0.5秒ごとにビットバッファを処理
                if time.time() - packet_timer > 0.5:
                    with self.lock:
                        if len(self.bit_buffer) > 100:  # 最小ビット数
                            # デバッグ出力
                            print(f"ビットバッファ処理: {len(self.bit_buffer)} bits")
                            
                            # バイト列に変換
                            bytes_data = self.bits_to_bytes(self.bit_buffer)
                            self.bit_buffer.clear()
                            
                            # パケットとして処理（簡易版）
                            if len(bytes_data) > self.header_size:
                                # 仮のパケット番号
                                seq_num = len(self.received_packets)
                                self.received_packets[seq_num] = bytes_data[:100]
                                print(f"パケット受信: #{seq_num} ({len(bytes_data)} bytes)")
                                
                    packet_timer = time.time()
                    
                time.sleep(0.1)
                
            except Exception as e:
                print(f"音声ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
                
        print("音声ワーカー: 終了")
        
    def receive_file(self, output_path, show_preview=True):
        """ファイル受信メイン（修正版）"""
        self.receiving = True
        self.show_preview = show_preview
        
        # カメラを先に初期化
        if not self.init_camera():
            print("カメラ初期化に失敗しました")
            return
            
        # プレビューウィンドウの初期設定
        if self.show_preview:
            cv2.namedWindow('Optical Receiver - Camera Preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Optical Receiver - Camera Preview', 1024, 768)
            print("プレビューウィンドウを作成しました")
        
        # 音声スレッドのみ開始（カメラはメインスレッドで処理）
        audio_thread = Thread(target=self.audio_worker)
        audio_thread.start()
        
        print("\n受信待機中...")
        print("操作方法:")
        print("  'q' - 終了")
        print("  's' - スクリーンショット保存")
        
        # メインループでカメラ処理とプレビュー
        frame_count = 0
        start_time = time.time()
        fps_list = []
        last_fps_time = time.time()
        last_check_time = time.time()
        
        while self.receiving:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                frame_count += 1
                
                # デバッグ: 最初の数フレーム
                if frame_count <= 5:
                    mean = np.mean(frame)
                    std = np.std(frame)
                    print(f"フレーム {frame_count}: 平均={mean:.1f}, 標準偏差={std:.1f}")
                
                display_frame = frame.copy()
                
                # FPS計算
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - start_time)
                    fps_list.append(fps)
                    last_fps_time = current_time
                
                # スクリーン検出
                warped, corners = self.detect_screen(frame)
                
                if warped is not None and corners is not None:
                    # 検出結果を描画
                    cv2.drawContours(display_frame, [corners.astype(int)], -1, (0, 255, 0), 3)
                    
                    # ビット抽出
                    bits = self.extract_bits_from_frame(warped, None)
                    with self.lock:
                        self.bit_buffer.extend(bits)
                
                # 簡単な情報表示
                info = [
                    f"Frame: {frame_count}",
                    f"FPS: {fps_list[-1]:.1f}" if fps_list else "FPS: --",
                    f"Packets: {len(self.received_packets)}",
                    f"Bit buffer: {len(self.bit_buffer) if hasattr(self, 'bit_buffer') else 0}",
                    f"Screen: {'DETECTED' if warped is not None else 'NOT FOUND'}",
                    "Press 'q' to quit, 's' for screenshot"
                ]
                
                y = 30
                for text in info:
                    cv2.putText(display_frame, text, (10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 25
                    
                # ビット抽出状況のデバッグ（100フレームごと）
                if frame_count % 100 == 0 and warped is not None:
                    print(f"フレーム {frame_count}: ビットバッファ={len(self.bit_buffer)}, パケット={len(self.received_packets)}")
                
                # プレビュー表示（メインスレッドで実行）
                if self.show_preview:
                    cv2.imshow('Optical Receiver - Camera Preview', display_frame)
                
                # キー操作
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.receiving = False
                elif key == ord('s'):
                    cv2.imwrite(f'screenshot_{frame_count}.png', display_frame)
                    print(f"保存: screenshot_{frame_count}.png")
            
            # 定期的にファイル復元をチェック
            if current_time - last_check_time > 0.5:
                file_data = self.process_packets()
                if file_data is not None:
                    # ファイル保存
                    with open(output_path, 'wb') as f:
                        f.write(file_data)
                    print(f"\nファイル受信完了: {output_path} ({len(file_data)} bytes)")
                    self.receiving = False
                last_check_time = current_time
                
        # クリーンアップ
        print("受信を終了します...")
        audio_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        
    def calibrate(self):
        """キャリブレーションモード"""
        if not self.init_camera():
            print("カメラ初期化に失敗しました")
            return
        
        print("=== キャリブレーションモード ===")
        print("送信画面を映してください")
        print("\n操作方法:")
        print("  's' - スクリーン検出結果を保存")
        print("  'c' - セルグリッド表示切替")
        print("  'e' - エッジ検出表示切替")
        print("  'q' - 終了")
        
        show_grid = True
        show_edges = False
        detection_count = 0
        
        # ウィンドウ作成
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration', 800, 600)
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
                
            # フレームの正常性チェック
            mean_val = np.mean(frame)
            std_val = np.std(frame)
            if std_val < 1.0 and 127 < mean_val < 129:
                continue
                
            # スクリーン検出
            warped, corners = self.detect_screen(frame)
            
            if warped is not None and corners is not None:
                detection_count += 1
                
                # 検出結果を描画
                cv2.drawContours(frame, [corners.astype(int)], -1, (0, 255, 0), 3)
                
                # 頂点番号表示
                for i, point in enumerate(corners):
                    cv2.circle(frame, tuple(point.astype(int)), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"{i+1}", tuple(point.astype(int) + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # セルグリッド表示
                if show_grid:
                    grid_overlay = warped.copy()
                    for i in range(0, warped.shape[0], self.cell_size):
                        cv2.line(grid_overlay, (0, i), (warped.shape[1], i), (0, 255, 0), 1)
                    for i in range(0, warped.shape[1], self.cell_size):
                        cv2.line(grid_overlay, (i, 0), (i, warped.shape[0]), (0, 255, 0), 1)
                    
                    cv2.imshow('Warped with Grid', grid_overlay)
                else:
                    cv2.imshow('Warped', warped)
                
                # エッジ検出表示
                if show_edges:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    cv2.imshow('Edge Detection', edges)
                
                # 統計情報
                cv2.putText(frame, f"Detection: {detection_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Grid cells: {warped.shape[1]//self.cell_size} x {warped.shape[0]//self.cell_size}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Screen not detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and warped is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'calibration_frame_{timestamp}.png', frame)
                cv2.imwrite(f'calibration_warped_{timestamp}.png', warped)
                print(f"キャリブレーション結果を保存しました")
            elif key == ord('c'):
                show_grid = not show_grid
                print(f"グリッド表示: {'ON' if show_grid else 'OFF'}")
            elif key == ord('e'):
                show_edges = not show_edges
                if not show_edges:
                    cv2.destroyWindow('Edge Detection')
                
        self.cap.release()
        cv2.destroyAllWindows()

# 使用例
if __name__ == "__main__":
    print("AIRCODE風 光学通信受信プログラム")
    print("================================")
    
    # カメラインデックスの確認
    print("\n利用可能なカメラを確認中...")
    camera_list = []
    camera_info = {}
    
    for i in range(3):
        # DirectShowで詳細確認
        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
            
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                # 映像の特徴を確認
                mean_val = np.mean(frame)
                std_val = np.std(frame)
                
                # カメラタイプを推測
                if mean_val < 50 and width == 640:
                    camera_type = "内蔵カメラ（暗い）"
                elif std_val < 5:
                    camera_type = "仮想カメラ？"
                else:
                    camera_type = "Webカメラ"
                
                print(f"カメラ {i}: {width}x{height} @{fps}fps [{backend}] - {camera_type}")
                camera_list.append(i)
                camera_info[i] = {
                    'type': camera_type,
                    'resolution': f"{width}x{height}",
                    'mean': mean_val,
                    'std': std_val
                }
                
                # プレビュー画像保存
                preview_file = f"camera_{i}_preview.jpg"
                cv2.imwrite(preview_file, frame)
                print(f"  → プレビュー保存: {preview_file}")
            
            cap.release()
    
    if not camera_list:
        print("\n✗ 利用可能なカメラが見つかりません")
        exit(1)
    
    # デフォルトカメラの推奨
    recommended = None
    for idx in camera_list:
        if "Webカメラ" in camera_info[idx]['type']:
            recommended = idx
            break
    if recommended is None:
        recommended = camera_list[0]
    
    print(f"\n推奨: カメラ {recommended} ({camera_info[recommended]['type']})")
    
    camera_idx = input(f"使用するカメラインデックス (Enter で推奨カメラ{recommended}): ")
    if camera_idx == "":
        camera_idx = recommended
    else:
        camera_idx = int(camera_idx)
    
    receiver = OpticalReceiver()
    receiver.camera_index = camera_idx
    
    print("\n1. キャリブレーション")
    print("2. ファイル受信（プレビューあり）")
    print("3. ファイル受信（プレビューなし）")
    print("4. カメラテストのみ")
    
    choice = input("\n選択してください (1-4): ")
    
    output_filename = "received_file.txt"
    if choice in ['2', '3']:
        custom = input("\n保存ファイル名 (Enterでreceived_file.txt): ")
        if custom:
            output_filename = custom
    
    if choice == '1':
        # キャリブレーション
        receiver.calibrate()
    elif choice == '2':
        # ファイル受信（プレビューあり）
        print(f"\n受信ファイルは '{output_filename}' に保存されます")
        receiver.receive_file(output_filename, show_preview=True)
    elif choice == '3':
        # ファイル受信（プレビューなし - 高速）
        print(f"\n受信ファイルは '{output_filename}' に保存されます")
        receiver.receive_file(output_filename, show_preview=False)
    elif choice == '4':
        # カメラテストのみ
        print("\nカメラテストモード")
        if receiver.init_camera():
            print("ESCで終了、's'でスクリーンショット")
            frame_count = 0
            while True:
                ret, frame = receiver.cap.read()
                if ret and frame is not None:
                    # 情報表示
                    cv2.putText(frame, f"Camera {camera_idx} - Frame {frame_count}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Mean: {np.mean(frame):.1f}, Std: {np.std(frame):.1f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera Test', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('s'):
                        filename = f"test_capture_{frame_count}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"保存: {filename}")
                        
                    frame_count += 1
                    
            receiver.cap.release()
            cv2.destroyAllWindows()
        else:
            print("カメラ初期化失敗")
    else:
        print("無効な選択です")