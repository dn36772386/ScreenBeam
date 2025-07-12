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
        
    def init_camera(self):
        """カメラ初期化"""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
        self.init_camera()
        prev_frame = None
        
        while self.receiving:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # スクリーン検出と歪み補正
            warped, corners = self.detect_screen(frame)
            
            if warped is not None:
                # ビット抽出
                bits = self.extract_bits_from_frame(warped, prev_frame)
                
                with self.lock:
                    self.bit_buffer.extend(bits)
                    
                prev_frame = warped
                
                # デバッグ表示
                if corners is not None:
                    cv2.drawContours(frame, [corners.astype(int)], -1, (0, 255, 0), 2)
                    
            cv2.imshow('Receiver', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.receiving = False
                
        self.cap.release()
        cv2.destroyAllWindows()
        
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
        
    def receive_file(self, output_path):
        """ファイル受信メイン"""
        self.receiving = True
        
        # ワーカースレッド開始
        camera_thread = Thread(target=self.camera_worker)
        audio_thread = Thread(target=self.audio_worker)
        
        camera_thread.start()
        audio_thread.start()
        
        print("受信待機中... 'q'で終了")
        
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
        
        print("キャリブレーション中... 送信画面を映してください")
        print("'s'でスクリーン検出結果を保存、'q'で終了")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # スクリーン検出
            warped, corners = self.detect_screen(frame)
            
            if warped is not None and corners is not None:
                # 検出結果を描画
                cv2.drawContours(frame, [corners.astype(int)], -1, (0, 255, 0), 2)
                
                # セルグリッド表示
                for i in range(0, warped.shape[0], self.cell_size):
                    cv2.line(warped, (0, i), (warped.shape[1], i), (128, 128, 128), 1)
                for i in range(0, warped.shape[1], self.cell_size):
                    cv2.line(warped, (i, 0), (i, warped.shape[0]), (128, 128, 128), 1)
                    
                cv2.imshow('Warped', warped)
                
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and warped is not None:
                cv2.imwrite('calibration_result.png', warped)
                print("キャリブレーション結果を保存しました")
                
        self.cap.release()
        cv2.destroyAllWindows()

# 使用例
if __name__ == "__main__":
    receiver = OpticalReceiver()
    
    # キャリブレーション
    # receiver.calibrate()
    
    # ファイル受信
    receiver.receive_file("received_file.txt")