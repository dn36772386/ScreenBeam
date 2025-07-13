#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIRCODE 風 光学通信 受信プログラム（ヘッダー検出・欠損パケット補完）

依存:
  pip install opencv-python reedsolo crcmod numpy
"""

import cv2, numpy as np, argparse, collections, time, struct, hashlib
from reedsolo import RSCodec
import crcmod

def draw_guides(img):
    """グリッドと中心線を半透明で描画"""
    h, w = img.shape[:2]
    overlay = img.copy()
    # 外枠
    cv2.rectangle(overlay, (0,0), (w-1,h-1), (0,255,0), 2)
    # 中心十字
    cv2.line(overlay, (w//2, 0), (w//2, h), (0,255,0), 1)
    cv2.line(overlay, (0, h//2), (w, h//2), (0,255,0), 1)
    # 3×3 グリッド
    for i in range(1,3):
        cv2.line(overlay, (w*i//3, 0), (w*i//3, h), (0,255,0), 1)
        cv2.line(overlay, (0, h*i//3), (w, h*i//3), (0,255,0), 1)
    # 透明度 30%
    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

def preview_loop(cap):
    """位置合わせ用のプレビュー。SPACE で確定、ESC で終了"""
    print("プレビュー開始（SPACE で受信開始、ESC で終了）")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        guide = draw_guides(frame)
        cv2.putText(guide, "SPACE: start  ESC: quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("preview", guide)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:      # SPACE
            cv2.destroyWindow("preview")
            print("→ 受信開始")
            return True
        elif k == 27:    # ESC
            cv2.destroyWindow("preview")
            print("キャンセルされました")
            return False

GRID_W, GRID_H   = 192, 108
CELL_SIZE        = 10
PKT_DATA_SIZE    = 1024
PKT_FULL_SIZE    = 1058
PKT_BITS         = PKT_FULL_SIZE * 8
HEADER_SIZE      = 128
RS               = RSCodec(10)
CRC32            = crcmod.mkCrcFun(0x104C11DB7, initCrc=0, xorOut=0xFFFFFFFF)

def bits_from_frame(frame_gray):
    h, w = frame_gray.shape
    cell_w = w // GRID_W
    cell_h = h // GRID_H
    thresh = np.median(frame_gray)
    bits = []
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            block = frame_gray[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            avg = block.mean()
            bits.append(1 if avg > thresh else 0)
    return bits

def bits_to_bytes(bits):
    res = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        res.append(byte)
    return bytes(res)

def decode_packet(buf):
    try:
        body = RS.decode(buf)
    except ReedSolomonError:
        return None, None
    seq = struct.unpack(">I", body[:4])[0]
    data = body[4:-4]
    crc = struct.unpack(">I", body[-4:])[0]
    if CRC32(data) != crc:
        return None, None
    return seq, data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--outfile", default="received_file.txt")
    ap.add_argument("--nopreview", action="store_true",
                    help="位置合わせプレビューをスキップ")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("カメラを開けません"); return
    if not args.nopreview:
        if not preview_loop(cap):
            cap.release()
            return

    bit_buf = []
    header_parsed = False
    total_pkts = None
    file_size  = None
    sha16      = None
    packets    = {}
    last_report = time.time()

    print("受信待機...")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = bits_from_frame(gray)
        bit_buf.extend(bits)

        # ヘッダーフレーム確認
        if not header_parsed and len(bit_buf) >= HEADER_SIZE*8:
            header_bits = bit_buf[:HEADER_SIZE*8]
            bit_buf = bit_buf[HEADER_SIZE*8:]
            hdr = bits_to_bytes(header_bits)
            file_size, total_pkts, _, sha16 = struct.unpack(">QII16s", hdr[:32])
            header_parsed = True
            print(f"Header: size={file_size}, pkts={total_pkts}")
            packets[0] = hdr  # 0 番はヘッダーとして保存
            continue

        # データパケット処理
        while header_parsed and len(bit_buf) >= PKT_BITS:
            pkt_bits = bit_buf[:PKT_BITS]
            bit_buf = bit_buf[PKT_BITS:]
            seq, data = decode_packet(bits_to_bytes(pkt_bits))
            if seq is not None and seq not in packets:
                packets[seq] = data
                print(f"✔ Packet {seq}/{total_pkts} OK")

        # 進捗レポート
        if header_parsed and time.time() - last_report > 2:
            print(f"Progress: {len(packets)-1}/{total_pkts}")
            last_report = time.time()

        # 完了判定
        if header_parsed and (len(packets)-1) == total_pkts:
            ordered = [packets[i] for i in range(1, total_pkts+1)]
            payload = b"".join(ordered)[:file_size]
            if hashlib.sha256(payload).digest()[:16] != sha16:
                print("⚠ ハッシュ不一致。再送を待機します")
                header_parsed = False; packets.clear(); bit_buf.clear()
                continue
            with open(args.outfile, "wb") as f:
                f.write(payload)
            print(f"受信完了: {args.outfile} ({len(payload)} byte)")
            break

        # ESC キーで強制終了
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()

if __name__ == "__main__":
    main()
