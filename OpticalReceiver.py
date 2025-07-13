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
    
    # HD解像度に設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 実際に設定された値を確認
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if actual_width < 1920 or actual_height < 1080:
        print(f"  ⚠ HD解像度が利用できません: {int(actual_width)}x{int(actual_height)}")
        # 中間解像度を試す
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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
    cap.set(cv2.CAP_PROP_FPS, 10)  # 10FPSに設定（送信側と同じ）
    
    # 明るさとコントラストを調整（画面が暗い問題に対処）
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)  # 明るさを上げる
    cap.set(cv2.CAP_PROP_CONTRAST, 50)     # コントラストを上げる
    
    # 露出設定（環境に応じて調整）
    # 注：一部のカメラではこれらの設定が効かない場合があります
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手動モード
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # 露出を少し明るめに
    except:
        print("  ⚠ 露出設定をスキップしました")
    
    print("→ 最適化完了")

def preview_loop(cap):
    """位置合わせ用のプレビュー。SPACE で確定、ESC で終了"""
    print("プレビュー開始（SPACE で受信開始、ESC で終了）")
    print("ヒント: モニターの輝度を最大にしてください")
    
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
        
        # グレースケール変換して明るさ情報を取得
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        min_brightness = np.min(gray)
        max_brightness = np.max(gray)
        
        # 情報表示
        cv2.putText(guide, f"FPS: {current_fps:.1f}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(guide, f"Brightness: {mean_brightness:.0f} ({min_brightness}-{max_brightness})", 
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # 明るさが低い場合の警告
        if mean_brightness < 50:
            cv2.putText(guide, "WARNING: Too dark! Increase monitor brightness", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(guide, "SPACE: start  ESC: quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow("preview", guide)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:      # SPACE
            cv2.destroyWindow("preview")
            print(f"→ 受信開始 (明るさ: {mean_brightness:.0f})")
            if mean_brightness < 50:
                print("⚠ 警告: 画面が暗すぎます。モニターの輝度を上げてください。")
            return True
        elif k == 27:    # ESC
            cv2.destroyWindow("preview")
            print("キャンセルされました")
            return False

# 定数定義
GRID_W, GRID_H   = 96, 54  # 送信側と同じに変更
SYNC_PATTERN     = [1] * 48 + [0] * 48  # 半分白、半分黒
PKT_DATA_SIZE    = 1024
PKT_FULL_SIZE    = 1058  # RSコード追加後のサイズ（デフォルト）
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
    
    # 画面中央の適切な領域を切り出し（画面端のノイズを避ける）
    margin_x = w // 10
    margin_y = h // 10
    roi = frame_gray[margin_y:h-margin_y, margin_x:w-margin_x]
    roi_h, roi_w = roi.shape
    
    cell_w = roi_w // GRID_W
    cell_h = roi_h // GRID_H
    
    # 画面全体の明るさを確認
    overall_mean = roi.mean()
    
    if adaptive_thresh:
        # 適応的閾値処理（ブロックサイズを調整）
        block_size = max(3, min(cell_w, cell_h) // 2)
        if block_size % 2 == 0:
            block_size += 1
        thresh_img = cv2.adaptiveThreshold(roi, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block_size, 10)
    else:
        # Otsuの閾値処理を使用（より安定）
        _, thresh_img = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    bits = []
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            y1 = gy * cell_h
            y2 = min((gy + 1) * cell_h, roi_h)
            x1 = gx * cell_w
            x2 = min((gx + 1) * cell_w, roi_w)
            
            block = thresh_img[y1:y2, x1:x2]
            if block.size > 0:
                # 中央部分の平均を取る（エッジのノイズを避ける）
                center_y = block.shape[0] // 4
                center_x = block.shape[1] // 4
                if center_y > 0 and center_x > 0:
                    center_block = block[center_y:-center_y, center_x:-center_x]
                    if center_block.size > 0:
                        avg = center_block.mean()
                    else:
                        avg = block.mean()
                else:
                    avg = block.mean()
                
                # 画面が暗い場合は閾値を調整
                if overall_mean < 50:
                    bits.append(1 if avg > 100 else 0)
                else:
                    bits.append(1 if avg > 127 else 0)
            else:
                bits.append(0)
    
    return bits

def find_sync_pattern(bits, debug=False):
    """同期パターンを検出してフレームの開始位置を特定"""
    if len(bits) < len(SYNC_PATTERN):
        return -1
    
    # 最初の行（96ビット）で同期パターンを確認
    first_row = bits[:GRID_W]
    
    # 同期パターンとの一致度を計算
    matches = sum(1 for i in range(len(SYNC_PATTERN)) if first_row[i] == SYNC_PATTERN[i])
    match_rate = matches / len(SYNC_PATTERN) * 100
    
    if debug and match_rate > 50:  # 50%以上一致したら表示
        print(f"\n同期パターン一致度: {match_rate:.1f}%")
        # 詳細な比較
        expected_white = sum(SYNC_PATTERN[:48])
        actual_white = sum(first_row[:48])
        expected_black = 48 - sum(SYNC_PATTERN[48:])
        actual_black = 48 - sum(first_row[48:])
        print(f"前半（白）: {actual_white}/{expected_white}, 後半（黒）: {actual_black}/{expected_black}")
    
    # 80%以上一致したら同期成功とする（完全一致は難しい）
    if match_rate >= 80:
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
    except (ReedSolomonError, Exception) as e:
        return None, None
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
    ap.add_argument("--no-adaptive", action="store_true",
                    help="適応的閾値処理を無効化")
    ap.add_argument("--debug", action="store_true",
                    help="デバッグ情報を表示")
    args = ap.parse_args()
    
    # 適応的閾値のフラグ設定
    use_adaptive = not args.no_adaptive

    cap = cv2.VideoCapture(args.cam)  # DirectShowを使わない
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
    print("\n💡 トラブルシューティング:")
    print("  1. モニターの輝度を最大に")
    print("  2. カメラをモニターに正対させる")
    print("  3. グリッド全体が画面に収まるように調整")
    print("  4. 部屋を暗くしてみる")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = bits_from_frame(gray, use_adaptive)
        
        if args.debug and frame_count % 60 == 0:
            # デバッグ：最初の行のビットパターンを表示
            first_48_bits = bits[:48] if len(bits) >= 48 else bits
            last_48_bits = bits[48:96] if len(bits) >= 96 else []
            
            first_pattern = ''.join(str(b) for b in first_48_bits)
            last_pattern = ''.join(str(b) for b in last_48_bits)
            
            print(f"\n=== フレーム {frame_count} ===")
            print(f"前半48ビット: {first_pattern[:16]}... (白: {sum(first_48_bits)}/48)")
            print(f"後半48ビット: {last_pattern[:16]}... (黒: {48-sum(last_48_bits)}/48)")
            
            # 画面の明暗情報
            bright_count = sum(bits[:GRID_W]) if len(bits) >= GRID_W else 0
            print(f"最初の行の明るいセル: {bright_count}/{GRID_W}")
            
            # 画面全体の統計
            if len(bits) >= GRID_W * GRID_H:
                total_bright = sum(bits)
                total_cells = GRID_W * GRID_H
                print(f"画面全体の明るいセル: {total_bright}/{total_cells} ({total_bright/total_cells*100:.1f}%)")
        sync_pos = find_sync_pattern(bits, args.debug)
        if sync_pos != 0:
            if args.debug and frame_count % 30 == 0:
                print(f"\r同期待機中... フレーム: {frame_count}", end="")
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
                if len(hdr) >= 33:  # 最小サイズチェック
                    file_size, total_pkts, _, sha16, rs_level_byte = struct.unpack(">QII16sB", hdr[:33])
                    
                    # 妥当性チェック
                    if file_size > 0 and file_size < 1000000000 and total_pkts > 0 and total_pkts < 10000:
                        rs_level = rs_level_byte
                        rs_codec = RS_LEVELS.get(rs_level, RS_LEVELS[1])
                        header_parsed = True
                        
                        print(f"\n=== ヘッダー情報 ===")
                        print(f"ファイルサイズ : {file_size:,} bytes")
                        print(f"総パケット数   : {total_pkts}")
                        print(f"エラー訂正     : {['light', 'medium', 'strong'][rs_level] if rs_level < 3 else 'unknown'}")
                        print()
                        
                        packets[0] = hdr
                        continue
                    else:
                        if args.debug:
                            print(f"⚠ 不正なヘッダー値: size={file_size}, pkts={total_pkts}")
            except Exception as e:
                if args.debug:
                    print(f"⚠ ヘッダー解析エラー: {e}")
                error_count += 1
                continue
        
        # データパケット処理
        if header_parsed:
            # パケットサイズを計算
            try:
                test_data = b'x' * (PKT_DATA_SIZE + 8)
                pkt_size = len(rs_codec.encode(test_data))
                pkt_bits = pkt_size * 8
            except:
                # デフォルトサイズ
                pkt_bits = PKT_FULL_SIZE * 8
            
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
            if frame_count > 0:
                print(f"同期成功率     : {sync_count/frame_count*100:.1f}%")
            print(f"エラー数       : {error_count}")
            print(f"受信パケット   : {len(packets)-1 if header_parsed else 0}/{total_pkts or '?'}")
            
            # ヒント表示
            if sync_count == 0 and frame_count > 100:
                print("\n💡 ヒント:")
                print("1. モニターの輝度を最大にしてください")
                print("2. カメラとモニターの距離・角度を調整してください")
                print("3. 部屋の照明を暗くしてみてください")
                print("4. --no-adaptive オプションを試してください")
            
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