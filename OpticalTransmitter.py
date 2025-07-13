#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善版 AIRCODE 風 光学通信 送信プログラム（同期パターン付き）

依存:
  pip install pygame reedsolo crcmod
"""

import pygame, sys, time, argparse, math, hashlib, os, struct
from reedsolo import RSCodec
import crcmod

# ------- 送信設定 -------
FPS               = 10  # さらに下げて確実性を高める
GRID_W, GRID_H    = 96, 54  # 解像度を半分に（640x480カメラ対応）
CELL_SIZE         = 10
SCREEN_W          = GRID_W * CELL_SIZE
SCREEN_H          = GRID_H * CELL_SIZE

# 同期パターン（最初の行に配置）- 半分白、半分黒
SYNC_PATTERN      = [1] * 48 + [0] * 48  # 半分白、半分黒

# 適応的エラー訂正（デフォルトは中程度）
RS_LEVELS = {
    'light': RSCodec(5),     # 2.5バイトまでのエラー訂正
    'medium': RSCodec(10),   # 5バイトまでのエラー訂正
    'strong': RSCodec(20)    # 10バイトまでのエラー訂正
}

CRC32 = crcmod.mkCrcFun(0x104C11DB7, initCrc=0, xorOut=0xFFFFFFFF)
HEADER_SIZE       = 128
PKT_DATA_SIZE     = 1024

# ------- パケット生成 -------
def chunkify(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i:i+size]

def build_packets(path: str, rs_level: str = 'medium'):
    RS = RS_LEVELS[rs_level]
    raw = open(path, "rb").read()
    sha256 = hashlib.sha256(raw).digest()[:16]
    total_pkts = math.ceil(len(raw) / PKT_DATA_SIZE)
    
    # ヘッダーにRS強度情報を追加
    header = struct.pack(">QII16sB", len(raw), total_pkts, PKT_DATA_SIZE, sha256, 
                        {'light': 0, 'medium': 1, 'strong': 2}[rs_level])
    header += b"\x00" * (HEADER_SIZE - len(header))
    
    packets = [header]
    for seq, chunk in enumerate(chunkify(raw, PKT_DATA_SIZE), 1):
        pad = chunk + b"\x00" * (PKT_DATA_SIZE - len(chunk))
        crc = struct.pack(">I", CRC32(pad))
        body = struct.pack(">I", seq) + pad + crc
        packets.append(RS.encode(body))
    
    return packets

# ------- ビット→画面 -------
def bytes_to_bits(buf: bytes):
    for b in buf:
        for i in range(8):
            yield (b >> (7-i)) & 1

def draw_frame_with_sync(surf, packet_bits, frame_number):
    surf.fill((0,0,0))
    
    # 最初の行に同期パターンを描画
    for idx, bit in enumerate(SYNC_PATTERN):
        if bit:
            x = idx * CELL_SIZE
            y = 0
            pygame.draw.rect(surf, (255,255,255), (x, y, CELL_SIZE, CELL_SIZE))
    
    # フレーム番号を2行目に埋め込む（16ビット）
    frame_bits = [(frame_number >> (15-i)) & 1 for i in range(16)]
    for idx, bit in enumerate(frame_bits):
        if bit:
            x = idx * CELL_SIZE
            y = CELL_SIZE
            pygame.draw.rect(surf, (255,255,255), (x, y, CELL_SIZE, CELL_SIZE))
    
    # 残りの領域にデータを描画（3行目から）
    data_start_idx = 2 * GRID_W  # 2行分スキップ
    bit_idx = 0
    
    for idx, bit in enumerate(packet_bits):
        if bit_idx + data_start_idx >= GRID_W * GRID_H:
            break
        if bit:
            grid_idx = bit_idx + data_start_idx
            x = (grid_idx % GRID_W) * CELL_SIZE
            y = (grid_idx // GRID_W) * CELL_SIZE
            pygame.draw.rect(surf, (255,255,255), (x, y, CELL_SIZE, CELL_SIZE))
        bit_idx += 1
    
    pygame.display.flip()

# ------- メイン -------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="送信するファイル")
    parser.add_argument("--interval", type=float, default=5.0, help="再送間隔[s]")
    parser.add_argument("--rs-level", choices=['light', 'medium', 'strong'], 
                       default='medium', help="エラー訂正強度")
    parser.add_argument("--show-stats", action="store_true", help="統計情報を表示")
    args = parser.parse_args()

    pkts = build_packets(args.file, args.rs_level)
    file_size = os.path.getsize(args.file)
    
    # 実効データレート計算（同期パターン分を考慮）
    data_bits_per_frame = GRID_W * (GRID_H - 2)  # 2行は同期用
    data_rate = data_bits_per_frame * FPS / 8 / 1024  # KB/秒
    
    print(f"=== 送信設定 ===")
    print(f"ファイル       : {args.file}")
    print(f"ファイルサイズ : {file_size:,} bytes")
    print(f"総パケット数   : {len(pkts)-1}")
    print(f"エラー訂正     : {args.rs_level}")
    print(f"フレームレート : {FPS} FPS")
    print(f"グリッドサイズ : {GRID_W}×{GRID_H}")
    print(f"理論転送速度   : {data_rate:.1f} KB/秒")
    print(f"推定転送時間   : {file_size/1024/data_rate:.1f} 秒")
    print(f"再送間隔       : {args.interval:.1f} 秒")
    print()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"光学通信送信 - {os.path.basename(args.file)}")
    clock = pygame.time.Clock()
    
    frame_counter = 0
    start_time = time.time()
    
    while True:
        for idx, pkt in enumerate(pkts):
            bits = list(bytes_to_bits(pkt))
            draw_frame_with_sync(screen, bits, frame_counter)
            frame_counter = (frame_counter + 1) % 65536  # 16ビットでループ
            
            # 統計情報表示
            if args.show_stats and frame_counter % 60 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_counter / elapsed
                print(f"\rFPS: {actual_fps:.1f} | フレーム: {frame_counter} | "
                      f"パケット: {idx}/{len(pkts)-1}", end="")
            
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            
            clock.tick(FPS)
        
        # 再送間隔
        print(f"\n→ {args.interval}秒待機...")
        wait_start = time.time()
        while time.time() - wait_start < args.interval:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            time.sleep(0.05)

if __name__ == "__main__":
    main()