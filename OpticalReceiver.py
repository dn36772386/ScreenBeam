#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善版 AIRCODE 風 光学通信 受信プログラム（同期検出・カメラ最適化）

依存:
  pip install opencv-python reedsolo crcmod numpy
"""

import cv2, numpy as np, argparse, collections, time, struct, hashlib
from reedsolo import RSCodec, ReedSolomonError
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
    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

def optimize_camera(cap):
    """カメラ設定を最適化"""
    print("カメラ設定を最適化中...")
    
    # 利用可能な設定を確認
    props = {
        'FPS': cv2.CAP_PROP_FPS,
        'Width': cv2.CAP_PROP_FRAME_WIDTH,
        'Height': cv2.CAP_PROP_FRAME_HEIGHT,
        'Exposure': cv2.CAP_PROP_EXPOSURE,
        'Gain': cv2.CAP_PROP_GAIN,
        'Brightness': cv2.CAP_PROP_BRIGHTNESS,
        'Contrast': cv2.CAP_PROP_CONTRAST
    }
    
    # 現在の設定を表示
    for name, prop in props.items():
        val = cap.get(prop)
        if val != -1:
            print(f"  {name}: {val}")
    
    # 最適化設定
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファ最小化
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 露出設定（環境に応じて調整）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手動モード
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 短い露出時間
    
    print("→ 最適化完了")

def preview_loop(cap):
    """位置合わせ用のプレビュー。SPACE で確定、ESC で終了"""
    print("プレビュー開始（SPACE で受信開始、ESC で終了）")
    
    # FPS計測用
    fps_counter = collections.deque(maxlen=30)
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # FPS計算
        current_time = time.time()
        fps_counter.append(1.0 / (current_time - last_time))
        last_time = current_time
        current_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
        
        guide = draw_guides(frame)
        
        # 情報表示
        cv2.putText(guide, f"FPS: {current_fps:.1f}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
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

# 定数定義
GRID_W, GRID_H   = 192, 108
SYNC_PATTERN     = [1,0,1,0,1,0,1,0] * 24  # 192ビット
PKT_DATA_SIZE    = 1024
HEADER_SIZE      = 128

# エラー訂正レベル
RS_LEVELS = {
    0: RSCodec(5),   # light
    1: RSCodec(10),  # medium
    2: RSCodec(20)   # strong
}

CRC32 = crcmod.mkCrcFun(0x104C11DB7, initCrc=0, xorOut=0xFFFFFFFF)

def bits_from_frame(frame_gray, adaptive_thresh=True):
    """フレームからビット列を抽出（適応的閾値対応）"""
    h, w = frame_gray.shape
    cell_w = w // GRID_W
    cell_h = h // GRID_H
    
    if adaptive_thresh:
        # 適応的閾値処理
        thresh_img = cv2.adaptiveThreshold(frame_gray, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    else:
        # 固定閾値
        thresh = np.median(frame_gray)
        _, thresh_img = cv2.threshold(frame_gray, thresh, 255, cv2.THRESH_BINARY)
    
    bits = []
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            block = thresh_img[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            avg = block.mean()
            bits.append(1 if avg > 127 else 0)
    
    return bits

def find_sync_pattern(bits):
    """同期パターンを検出してフレームの開始位置を特定"""
    if len(bits) < len(SYNC_PATTERN):
        return -1
    
    # 最初の行（192ビット）で同期パターンを確認
    first_row = bits[:GRID_W]
    if first_row == SYNC_PATTERN:
        return 0
    
    return -1

def extract_frame_number(bits):
    """フレーム番号を抽出（2行目の16ビット）"""
    if len(bits) < GRID_W * 2:
        return None
    
    frame_bits = bits[GRID_W:GRID_W + 16]
    frame_num = 0
    for bit in frame_bits:
        frame_num = (frame_num << 1) | bit
    
    return frame_num

def bits_to_bytes(bits):
    res = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        res.append(byte)
    return bytes(res)

def decode_packet(buf, rs_codec):
    try:
        body = rs_codec.decode(buf)
    except ReedSolomonError:
        return None, None
    
    if len(body) < 8:
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
    ap.add_argument("--adaptive", action="store_true",
                    help="適応的閾値処理を使用")
    ap.add_argument("--debug", action="store_true",
                    help="デバッグ情報を表示")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("カメラを開けません")
        return
    
    # カメラ最適化
    optimize_camera(cap)
    
    if not args.nopreview:
        if not preview_loop(cap):
            cap.release()
            return

    # 受信状態変数
    header_parsed = False
    total_pkts = None
    file_size = None
    sha16 = None
    rs_level = 1  # デフォルトはmedium
    rs_codec = RS_LEVELS[rs_level]
    
    packets = {}
    last_report = time.time()
    frame_count = 0
    sync_count = 0
    error_count = 0
    last_frame_num = -1
    
    print("受信待機中...")
    print("同期パターンを検索中...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = bits_from_frame(gray, args.adaptive)
        
        # 同期パターン検出
        sync_pos = find_sync_pattern(bits)
        if sync_pos != 0:
            continue
        
        sync_count += 1
        
        # フレーム番号抽出
        frame_num = extract_frame_number(bits)
        if frame_num is not None:
            if args.debug and frame_num != (last_frame_num + 1) % 65536:
                print(f"\n⚠ フレームスキップ検出: {last_frame_num} → {frame_num}")
            last_frame_num = frame_num
        
        # データ部分を抽出（3行目以降）
        data_bits = bits[GRID_W * 2:]
        
        # ヘッダー処理
        if not header_parsed and len(data_bits) >= HEADER_SIZE * 8:
            header_bits = data_bits[:HEADER_SIZE * 8]
            hdr = bits_to_bytes(header_bits)
            
            try:
                file_size, total_pkts, _, sha16, rs_level_byte = struct.unpack(">QII16sB", hdr[:33])
                rs_level = rs_level_byte
                rs_codec = RS_LEVELS.get(rs_level, RS_LEVELS[1])
                header_parsed = True
                
                print(f"\n=== ヘッダー情報 ===")
                print(f"ファイルサイズ : {file_size:,} bytes")
                print(f"総パケット数   : {total_pkts}")
                print(f"エラー訂正     : {['light', 'medium', 'strong'][rs_level]}")
                print()
                
                packets[0] = hdr
                continue
            except:
                error_count += 1
                continue
        
        # データパケット処理
        if header_parsed:
            # パケットサイズを計算
            pkt_size = rs_codec.encode(b'x' * (PKT_DATA_SIZE + 8))
            pkt_bits = len(pkt_size) * 8
            
            if len(data_bits) >= pkt_bits:
                pkt_bits_data = data_bits[:pkt_bits]
                pkt_bytes = bits_to_bytes(pkt_bits_data)
                
                seq, data = decode_packet(pkt_bytes, rs_codec)
                if seq is not None and seq not in packets:
                    packets[seq] = data
                    print(f"✔ パケット {seq}/{total_pkts} 受信")
                elif seq is None:
                    error_count += 1
        
        # 統計情報表示
        if args.debug and time.time() - last_report > 2:
            print(f"\n--- 統計情報 ---")
            print(f"フレーム数     : {frame_count}")
            print(f"同期成功数     : {sync_count}")
            print(f"同期成功率     : {sync_count/frame_count*100:.1f}%")
            print(f"エラー数       : {error_count}")
            print(f"受信パケット   : {len(packets)-1 if header_parsed else 0}/{total_pkts or '?'}")
            last_report = time.time()
        
        # 完了判定
        if header_parsed and total_pkts and (len(packets) - 1) == total_pkts:
            print("\n全パケット受信完了！")
            
            # データ結合と検証
            ordered = [packets[i] for i in range(1, total_pkts + 1)]
            payload = b"".join(ordered)[:file_size]
            
            computed_hash = hashlib.sha256(payload).digest()[:16]
            if computed_hash != sha16:
                print("⚠ ハッシュ不一致！再送を待機します...")
                header_parsed = False
                packets.clear()
                error_count = 0
                continue
            
            # ファイル保存
            with open(args.outfile, "wb") as f:
                f.write(payload)
            
            print(f"\n✅ 受信完了: {args.outfile} ({len(payload):,} bytes)")
            print(f"同期成功率: {sync_count/frame_count*100:.1f}%")
            print(f"エラー率  : {error_count/frame_count*100:.1f}%")
            break
        
        # ESCキーで終了
        if cv2.waitKey(1) == 27:
            print("\n受信を中断しました")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()