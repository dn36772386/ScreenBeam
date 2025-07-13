# AIRCODE風 光学通信送信側実装
# Windows 11 Python 3.12+

import numpy as np
import cv2
import pygame
import struct
import time
import os
from reedsolo import RSCodec
import zlib
from threading import Thread
import queue

class OpticalTransmitter:
    def __init__(self):
        # 画面設定
        self.screen_width = 1920
        self.screen_height = 1080
        self.fps = 60
        
        # セル設定
        self.cell_size = 10  # 10x10 pixels per cell
        self.grid_width = self.screen_width // self.cell_size
        self.grid_height = self.screen_height // self.cell_size
        self.bits_per_frame = self.grid_width * self.grid_height
        
        # 変調設定
        self.base_brightness = 128  # 中間グレー
        self.modulation_depth = 5   # ±2%相当（255の2%）
        
        # パケット設定
        self.packet_size = 1024  # bytes
        self.header_size = 8     # seq(4) + crc(4)
        
        # 誤り訂正設定
        self.rs = RSCodec(10)    # Reed-Solomon 10バイト冗長
        
        # キュー
        self.packet_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=5)
        
    def init_display(self):
        """Pygame画面初期化"""
        pygame.init()
        pygame.font.init()  # フォント初期化
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.FULLSCREEN | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("AIRCODE Transmitter")
        self.clock = pygame.time.Clock()
        
    def init_audio(self):
        """音声出力初期化"""
        try:
            self.audio = pyaudio.PyAudio()
            # 設定を調整
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                frames_per_buffer=512  # バッファサイズを小さく
            )
            print("音声出力を初期化しました")
        except Exception as e:
            print(f"音声初期化エラー: {e}")
            print("音声同期は無効になります")
            self.audio = None
            self.audio_stream = None
        
    def generate_sync_pulse(self):
        """同期用音声パルス生成"""
        samples = int(44100 * self.sync_duration)
        t = np.linspace(0, self.sync_duration, samples)
        # 音量を下げて、より短いパルスに
        pulse = 0.1 * np.sin(2 * np.pi * self.sync_freq * t)
        # フェードイン・フェードアウト
        fade_samples = int(samples * 0.1)
        for i in range(fade_samples):
            pulse[i] *= (i / fade_samples)
            pulse[-(i+1)] *= (i / fade_samples)
        return pulse.astype(np.float32).tobytes()
        
    def file_to_packets(self, filepath):
        """ファイルをパケットに分割"""
        with open(filepath, 'rb') as f:
            data = f.read()
            
        # ファイルサイズ情報を最初のパケットに含める
        file_size = len(data)
        packets = []
        
        # ヘッダーパケット
        header = struct.pack('<Q', file_size)  # 8 bytes
        packets.append(self.create_packet(0, header))
        
        # データパケット
        for i in range(0, len(data), self.packet_size - self.header_size):
            chunk = data[i:i + self.packet_size - self.header_size]
            seq_num = (i // (self.packet_size - self.header_size)) + 1
            packets.append(self.create_packet(seq_num, chunk))
            
        return packets
        
    def create_packet(self, seq_num, data):
        """パケット作成（シーケンス番号 + データ + CRC）"""
        header = struct.pack('<I', seq_num)
        payload = header + data
        crc = zlib.crc32(payload) & 0xffffffff
        packet = payload + struct.pack('<I', crc)
        
        # Reed-Solomon符号化
        encoded = self.rs.encode(packet)
        return encoded
        
    def packet_to_bits(self, packet):
        """パケットをビット列に変換"""
        bits = []
        for byte in packet:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        return bits
        
    def bits_to_frame(self, bits):
        """ビット列を画面フレームに変換"""
        frame = np.full((self.screen_height, self.screen_width), 
                       self.base_brightness, dtype=np.uint8)
        
        bit_index = 0
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if bit_index < len(bits):
                    # ビット値に応じて輝度変調
                    if bits[bit_index] == 1:
                        brightness = self.base_brightness + self.modulation_depth
                    else:
                        brightness = self.base_brightness - self.modulation_depth
                        
                    # セルを塗りつぶし
                    y_start = y * self.cell_size
                    y_end = (y + 1) * self.cell_size
                    x_start = x * self.cell_size
                    x_end = (x + 1) * self.cell_size
                    frame[y_start:y_end, x_start:x_end] = brightness
                    
                    bit_index += 1
                    
        return frame
        
    def encode_worker(self, filepath):
        """エンコードワーカースレッド"""
        packets = self.file_to_packets(filepath)
        
        for packet in packets:
            bits = self.packet_to_bits(packet)
            
            # ビットを複数フレームに分割
            frames_needed = (len(bits) + self.bits_per_frame - 1) // self.bits_per_frame
            
            for i in range(frames_needed):
                start = i * self.bits_per_frame
                end = min(start + self.bits_per_frame, len(bits))
                frame_bits = bits[start:end]
                
                # 不足分は0で埋める
                if len(frame_bits) < self.bits_per_frame:
                    frame_bits.extend([0] * (self.bits_per_frame - len(frame_bits)))
                    
                frame = self.bits_to_frame(frame_bits)
                self.frame_queue.put(frame)
                
        # 終了マーカー
        self.frame_queue.put(None)
        
    def transmit_file(self, filepath, repeat=False, repeat_delay=5.0):
        """ファイル送信メイン"""
        self.init_display()
        # 音声は使用しない
        # self.init_audio()
        
        running = True
        
        while running:
            print(f"\n送信開始: {filepath}")
            
            # エンコードスレッド開始
            encode_thread = Thread(target=self.encode_worker, args=(filepath,))
            encode_thread.start()
            
            frame_count = 0
            packet_count = 0
            
            while True:
                # イベント処理
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                            
                if not running:
                    break
                    
                try:
                    # フレーム取得
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    if frame is None:
                        # 送信完了
                        print(f"送信完了 - {frame_count} フレーム送信")
                        print(f"待機中... (ESCで終了、SPACEでスキップ)")
                        encode_thread.join()
                        
                        if repeat:
                            print(f"\n{repeat_delay}秒後に再送信...")
                            
                            # 待機中も画面を黒く保つ
                            black_frame = np.full((self.screen_height, self.screen_width), 
                                                0, dtype=np.uint8)
                            surf = pygame.surfarray.make_surface(black_frame.T)
                            self.screen.blit(surf, (0, 0))
                            
                            # カウントダウンは表示しない（データ領域を妨げないため）
                            pygame.display.flip()
                            
                            wait_start = time.time()
                            while time.time() - wait_start < repeat_delay:
                                # イベント処理のみ
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        running = False
                                        break
                                    elif event.type == pygame.KEYDOWN:
                                        if event.key == pygame.K_ESCAPE:
                                            running = False
                                            break
                                        elif event.key == pygame.K_SPACE:
                                            # スペースキーでスキップ
                                            break
                                            
                                if not running:
                                    break
                                    
                                self.clock.tick(10)
                                    
                            # フレームキューをクリア
                            while not self.frame_queue.empty():
                                self.frame_queue.get()
                                
                            break  # 内側のループを抜けて再送信
                        else:
                            running = False
                            break
                    
                    # 画面に表示（ステータス表示なし - データ領域を妨げないため）
                    surf = pygame.surfarray.make_surface(frame.T)
                    self.screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    
                    # FPS制御
                    self.clock.tick(self.fps)
                    frame_count += 1
                    packet_count += 1
                    
                except queue.Empty:
                    continue
                    
            if not running:
                break
                
        # クリーンアップ
        pygame.quit()
        
    def calibrate(self):
        """キャリブレーション画面表示"""
        self.init_display()
        
        print("キャリブレーション画面を表示中...")
        print("ESCで終了、Spaceで白黒反転")
        print("データ領域には何も表示しません")
        
        # 変調パターンのテスト
        pattern_mode = 0
        running = True
        frame_count = 0
        
        while running:
            frame_count += 1
            
            # パターン生成
            if pattern_mode == 0:
                # チェッカーボード
                pattern = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
                for y in range(0, self.grid_height, 2):
                    for x in range(0, self.grid_width, 2):
                        y_start = y * self.cell_size
                        y_end = (y + 1) * self.cell_size
                        x_start = x * self.cell_size
                        x_end = (x + 1) * self.cell_size
                        pattern[y_start:y_end, x_start:x_end] = 255
                        
                for y in range(1, self.grid_height, 2):
                    for x in range(1, self.grid_width, 2):
                        y_start = y * self.cell_size
                        y_end = (y + 1) * self.cell_size
                        x_start = x * self.cell_size
                        x_end = (x + 1) * self.cell_size
                        pattern[y_start:y_end, x_start:x_end] = 255
                        
            elif pattern_mode == 1:
                # 全白
                pattern = np.full((self.screen_height, self.screen_width), 255, dtype=np.uint8)
                
            elif pattern_mode == 2:
                # 全黒
                pattern = np.full((self.screen_height, self.screen_width), 0, dtype=np.uint8)
                
            elif pattern_mode == 3:
                # 動的パターン（点滅）
                if frame_count % 10 < 5:
                    pattern = np.full((self.screen_height, self.screen_width), 
                                    self.base_brightness + self.modulation_depth, dtype=np.uint8)
                else:
                    pattern = np.full((self.screen_height, self.screen_width), 
                                    self.base_brightness - self.modulation_depth, dtype=np.uint8)
            
            # 表示（情報は表示しない）
            surf = pygame.surfarray.make_surface(pattern.T)
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)
            
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        pattern_mode = (pattern_mode + 1) % 4
                        mode_names = ["チェッカーボード", "全白", "全黒", "点滅テスト"]
                        print(f"モード変更: {mode_names[pattern_mode]}")
                        
        pygame.quit()

# 使用例
if __name__ == "__main__":
    transmitter = OpticalTransmitter()
    
    print("AIRCODE風 光学通信送信プログラム")
    print("================================")
    
    # ファイル選択
    print("\n送信するファイルを選択してください:")
    print("1. sample.txt（デフォルト）")
    print("2. 別のファイルを指定")
    
    file_choice = input("\n選択 (1-2): ")
    
    if file_choice == '2':
        filename = input("ファイル名を入力: ")
        if not os.path.exists(filename):
            print(f"エラー: {filename} が見つかりません")
            exit(1)
    else:
        filename = "sample.txt"
        # サンプルファイルの確認・作成
        if not os.path.exists(filename):
            print("\nサンプルファイル 'sample.txt' を作成します")
            with open(filename, "w", encoding="utf-8") as f:
                f.write("AIRCODE Test Message\n")
                f.write("=" * 50 + "\n")
                f.write("This is a test file for optical communication.\n")
                f.write("送信テスト用のファイルです。\n")
                f.write("File size: ~200 bytes\n")
                f.write("Transfer rate: ~1 Mbps\n")
                f.write("=" * 50 + "\n")
            print("サンプルファイルを作成しました")
    
    # ファイル情報表示
    file_size = os.path.getsize(filename)
    print(f"\n送信ファイル: {filename}")
    print(f"ファイルサイズ: {file_size} bytes")
    
    print("\n送信モードを選択:")
    print("1. 単発送信")
    print("2. リピート送信（5秒間隔）")
    print("3. 連続送信（1秒間隔）")
    print("4. キャリブレーション")
    
    choice = input("\n選択してください (1-4): ")
    
    if choice == '1':
        # 単発送信
        transmitter.transmit_file(filename, repeat=False)
    elif choice == '2':
        # リピート送信（5秒間隔）
        print("\nリピート送信モード（ESCで終了）")
        transmitter.transmit_file(filename, repeat=True, repeat_delay=5.0)
    elif choice == '3':
        # 連続送信
        print("\n連続送信モード（ESCで終了）")
        transmitter.transmit_file(filename, repeat=True, repeat_delay=1.0)
    elif choice == '4':
        # キャリブレーション
        transmitter.calibrate()
    else:
        print("無効な選択です")