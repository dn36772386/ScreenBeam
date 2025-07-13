#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIRCODE 風 光学通信 送信プログラム（リピート送信＋ヘッダーフレーム付き）

依存:
  pip install pygame reedsolo crcmod
"""

import pygame, sys, time, argparse, math, hashlib, os, struct
from reedsolo import RSCodec
import crcmod

# ------- 送信設定 -------
FPS               = 60
GRID_W, GRID_H    = 192, 108          # 16:9 フル HD 向け
CELL_SIZE         = 10                # ピクセル
SCREEN_W          = GRID_W * CELL_SIZE
SCREEN_H          = GRID_H * CELL_SIZE
RS                = RSCodec(10)       # RS(255,245)
CRC32             = crcmod.mkCrcFun(0x104C11DB7, initCrc=0, xorOut=0xFFFFFFFF)
HEADER_SIZE       = 128               # バイト
PKT_DATA_SIZE     = 1024              # RSを加えて 1 058 byte
PKT_FULL_SIZE     = 1058
BITS_PER_FRAME    = GRID_W * GRID_H   # 20 736bit
PKT_BITS          = PKT_FULL_SIZE * 8 # 8 464bit

# ------- パケット生成 -------
def chunkify(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i:i+size]

def build_packets(path: str):
    raw = open(path, "rb").read()
    sha256 = hashlib.sha256(raw).digest()[:16]  # 16byte だけ送る
    total_pkts = math.ceil(len(raw) / PKT_DATA_SIZE)
    header = struct.pack(">QII16s", len(raw), total_pkts, PKT_DATA_SIZE, sha256)
    header += b"\x00" * (HEADER_SIZE - len(header))
    packets = [header]  # packet 0 はヘッダー用フレーム（RS/CRCなし）
    for seq, chunk in enumerate(chunkify(raw, PKT_DATA_SIZE), 1):
        pad = chunk + b"\x00" * (PKT_DATA_SIZE - len(chunk))
        crc = struct.pack(">I", CRC32(pad))
        body = struct.pack(">I", seq) + pad + crc          # 4 + 1024 + 4
        packets.append(RS.encode(body))                    # 1 058byte
    return packets

# ------- ビット→画面 -------
def bytes_to_bits(buf: bytes):
    for b in buf:
        for i in range(8):
            yield (b >> (7-i)) & 1

def draw_frame(surf, bits):
    surf.fill((0,0,0))
    for idx, bit in enumerate(bits):
        if idx >= GRID_W*GRID_H: break
        if bit:
            x = (idx % GRID_W) * CELL_SIZE
            y = (idx // GRID_W) * CELL_SIZE
            pygame.draw.rect(surf, (255,255,255), (x, y, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()

# ------- メイン -------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="送信するファイル")
    parser.add_argument("--interval", type=float, default=5.0, help="再送間隔[s]")
    args = parser.parse_args()

    pkts = build_packets(args.file)
    print(f"Total packets  : {len(pkts)-1}")
    print(f"File size      : {os.path.getsize(args.file)} byte")
    print(f"Interval       : {args.interval:.1f} s")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    while True:  # 無限リピート
        for idx, pkt in enumerate(pkts):
            bits = bytes_to_bits(pkt)
            draw_frame(screen, bits)
            # ESC キーで強制終了
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
            clock.tick(FPS)

        # 休止
        t0 = time.time()
        while time.time() - t0 < args.interval:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
            time.sleep(0.05)

if __name__ == "__main__":
    main()
