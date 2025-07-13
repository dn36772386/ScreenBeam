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
        """カメラ初期化"""
        print("カメラを初期化中...")
        
        # 複数の方法でカメラ接続を試行
        methods = [
            (self.camera_index, "標準"),
            (self.camera_index + cv2.CAP_DSHOW, "DirectShow"),
            (self.camera_index + cv2.CAP_MSMF, "Media Foundation"),
        ]
        
        for idx, method_name in methods:
            print(f"{method_name}で接続試行...")
            self.cap = cv2.VideoCapture(idx)
            
            if self.cap.isOpened():
                # 設定を適用
                self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # テストフレーム
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"✓ {method_name}で接続成功")
                    print(f"  解像度: {frame.shape[1]}x{frame.shape[0]}")
                    print(f"  FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                    return True
                else:
                    self.cap.release()
                    
        print("✗ カメラ接続に失敗しました")
        print("\n確認事項:")
        print("- 他のアプリがカメラを使用していないか")
        print("- カメラドライバーが正しくインストールされているか")
        print("- 別のカメラインデックス（1, 2など）を試してみてください")
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
                
                # 変調検出
                if avg_brightness > self.base_brightness:
                    bits.append(1)
                else:
                    bits.append(0)
                    
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
        """受信パケット処理とファイル復元"""
        with self.lock:
            # ヘッダーパケット確認
            if 0 in self.received_packets:
                header_data = self.received_packets[0]
                self.file_size = struct.unpack('<Q', header_data[:8])[0]
                
            if self.file_size is None:
                return None
                
            # 必要パケット数計算
            data_packets_needed = (self.file_size + self.packet_size - self.header_size - 1) // (self.packet_size - self.header_size)
            total_packets_needed = data_packets_needed + 1  # +1 for header
            
            # 全パケット受信確認
            if len(self.received_packets) < total_packets_needed:
                return None
                
            # ファイル復元
            file_data = bytearray()
            for i in range(1, total_packets_needed):
                if i in self.received_packets:
                    file_data.extend(self.received_packets[i])
                    
            # 実際のファイルサイズに切り詰め
            file_data = file_data[:self.file_size]
            
            return bytes(file_data)
            
    def camera_worker(self):
        """カメラキャプチャワーカー"""
        if not self.init_camera():
            print("カメラワーカー: 初期化失敗")
            self.receiving = False
            return
            
        prev_frame = None
        
        # プレビュー用の統計情報
        frame_count = 0
        start_time = time.time()
        fps_list = []
        last_fps_time = time.time()
        
        # ビット抽出の視覚化用
        bit_visualization = None
        
        print("カメラワーカー: 開始")
        
        while self.receiving:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame_count += 1
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
                bits = self.extract_bits_from_frame(warped, prev_frame)
                
                with self.lock:
                    self.bit_buffer.extend(bits)
                    
                prev_frame = warped
                
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
                
                # 歪み補正後の画像も表示
                cv2.imshow('Warped Screen', cv2.resize(warped, (640, 360)))
            
            # 情報オーバーレイ
            with self.lock:
                bit_buffer_size = len(self.bit_buffer)
            self.draw_info_overlay(display_frame, frame_count, fps_list, 
                                 len(self.received_packets), bit_visualization, bit_buffer_size)
            
            # メインプレビュー表示
            cv2.imshow('Optical Receiver - Camera Preview', display_frame)
            
            # キー操作
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.receiving = False
            elif key == ord('s'):  # スクリーンショット
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'receiver_screenshot_{timestamp}.png', frame)
                print(f"スクリーンショット保存: receiver_screenshot_{timestamp}.png")
            elif key == ord('p'):  # 一時停止/再開
                cv2.waitKey(0)  # 任意のキーで再開
                
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
        self.init_audio()
        
        while self.receiving:
            try:
                audio_data = self.audio_stream.read(1024, exception_on_overflow=False)
                
                if self.detect_sync_pulse(audio_data):
                    # 同期パルス検出時、ビットバッファを処理
                    with self.lock:
                        if len(self.bit_buffer) > 0:
                            bytes_data = self.bits_to_bytes(self.bit_buffer)
                            self.bit_buffer.clear()
                            
                            # パケットデコード試行
                            seq_num, data = self.decode_packet(bytes_data)
                            
                            if seq_num is not None:
                                self.received_packets[seq_num] = data
                                print(f"パケット受信: #{seq_num} ({len(data)} bytes)")
                                
            except Exception as e:
                print(f"音声エラー: {e}")
                
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()
        
    def receive_file(self, output_path, show_preview=True):
        """ファイル受信メイン"""
        self.receiving = True
        self.show_preview = show_preview
        
        # プレビューウィンドウの初期設定
        if self.show_preview:
            cv2.namedWindow('Optical Receiver - Camera Preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Optical Receiver - Camera Preview', 1280, 720)
        
        # ワーカースレッド開始
        camera_thread = Thread(target=self.camera_worker)
        audio_thread = Thread(target=self.audio_worker)
        
        camera_thread.start()
        audio_thread.start()
        
        print("受信待機中...")
        print("操作方法:")
        print("  'q' - 終了")
        print("  's' - スクリーンショット保存")
        print("  'p' - 一時停止/再開")
        print("  '+/-' - プレビューサイズ変更")
        
        # 受信監視ループ
        while self.receiving:
            time.sleep(1)
            
            # ファイル復元試行
            file_data = self.process_packets()
            
            if file_data is not None:
                # ファイル保存
                with open(output_path, 'wb') as f:
                    f.write(file_data)
                    
                print(f"ファイル受信完了: {output_path} ({len(file_data)} bytes)")
                self.receiving = False
                
        # スレッド終了待機
        camera_thread.join()
        audio_thread.join()
        
    def calibrate(self):
        """キャリブレーションモード"""
        self.init_camera()
        
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
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
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
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ カメラ {i}: 利用可能 ({frame.shape[1]}x{frame.shape[0]})")
            cap.release()
    
    camera_idx = input("\n使用するカメラインデックス (通常は0): ")
    if camera_idx == "":
        camera_idx = 0
    else:
        camera_idx = int(camera_idx)
    
    receiver = OpticalReceiver()
    receiver.camera_index = camera_idx
    
    print("\n1. キャリブレーション")
    print("2. ファイル受信（プレビューあり）")
    print("3. ファイル受信（プレビューなし）")
    
    choice = input("\n選択してください (1-3): ")
    
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
    else:
        print("無効な選択です")