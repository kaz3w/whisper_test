#!/usr/bin/env python3
import subprocess
import argparse
import os
import whisper
from faster_whisper import WhisperModel
from janome.tokenizer import Tokenizer
import torch
import time

start_time = time.time()

# パーサーを作成
parser = argparse.ArgumentParser(description="動画から音声を抽出するスクリプト")

# 引数の定義
parser.add_argument("input", help="入力ファイル（例: abc.mp4）")
parser.add_argument("--rate", type=int, default=16000, help="サンプリングレート（例: 16000）")
parser.add_argument('--faster', action='store_true', help='高速モードを有効にします')
parser.add_argument('--cpu', action='store_true', help='CPUモードを有効にします')
parser.add_argument('--compute-type', choices=['float16', 'int8_float16', 'float32', 'int8'], default='float16', help='変数タイプ(例: float16, int8_float16, [float32], int8)')
parser.add_argument('--model', choices=['base', 'medium', 'large', 'large-v3'], default='medium', help='モデル（例: base, [medium], large）')
parser.add_argument('--dryrun', action='store_true', help='Dry-runモード（実行しない）')

# 引数をパース
args = parser.parse_args()
print("入力:", args.input)
print(f'{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f}GB, sm:{torch.cuda.get_device_capability(0)}')

stt_method = "faster-whisper" if args.faster else "whisper"
device_mode = "cpu" if args.cpu else "cuda"
print(f"method: {stt_method}, device: {device_mode}, model: {args.model}, compute_type: {args.compute_type}")
# Dry-runモードの場合は実行しない
if args.dryrun:
    exit(0)

# 入力動画と出力音声ファイル
# ベースネーム取得（拡張子なし）
base_name = os.path.splitext(os.path.basename( args.input))[0]

# 出力ファイル名
output_audio = f"{base_name}.wav"
output_text = f"{base_name}.txt"
output_tokenized = f"{base_name}_tokenized.txt"

if os.path.exists(output_audio):
    print("音声[抽出済]")
else:
    cmd = [
        "ffmpeg",
        "-i", args.input,
        "-vn",             # 映像を無視（video none）
        "-acodec", "pcm_s16le",  # WAV用コーデック
        "-ar", "16000",     # サンプリングレート（Whisper向けに16kHz）
        "-ac", "1",         # モノラル
        output_audio
    ]

    # 実行
    subprocess.run(cmd, check=True)
    print(f"音声抽出完了: {output_audio}")

if args.faster:
    # int8_float16 "large-v3"
    model = WhisperModel(args.model, device=device_mode, compute_type=args.compute_type)
    segments, info = model.transcribe(output_audio, beam_size=5)
    print(f"Detected language: {info.language} ({info.language_probability:.2f})")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
else:
    model = whisper.load_model(args.model)
    result = model.transcribe(output_audio, language="ja", verbose=True)
    words = result["text"]

tokenizer = Tokenizer()
words_tokenized = [token.surface for token in tokenizer.tokenize(words)]

with open(output_text, "w", encoding="utf-8") as f:
    f.write(words)

with open(output_tokenized, "w", encoding="utf-8") as f:
    f.write(" ".join(words_tokenized))

# 経過時間を表示
elapsed = time.time() - start_time
print(f" 経過時間: {elapsed:.2f}秒")

print(f"テキストファイルが保存されました: {output_text}")