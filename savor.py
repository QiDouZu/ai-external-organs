# -*- coding: utf-8 -*-
"""savor.py —— 耳朵之后的鉴赏器：逐句猜感情，带着前文再猜，最后回溯

阿余方法（2026-10-28）：
  「看完一句猜感情，再看一句和前面的一起再猜」——艺术鉴赏不是通读后
  一次性下结论，而是：①每读一句先猜此刻的感情；②下一句要带上前一句
  一起重猜；③全部读完，回头否定/修正前面的判断（回溯性重构）。

本工具只做「逐句呈现 + 时间对齐 + 曲调段映射」，不产出感情标签——
感情由读的人（模型或人）猜。它给出的是：
  lines:   每句：序号、起止时间、文本
  contexts: 每句所在曲调段（能量/质心方向，来自 ear.py 的 tune_profile 口径）
  prompt:   给鉴赏者的逐句猜法（猜 → 带前文再猜 → 回溯否定）

用法：
  python savor.py <字幕json> [--audio <m4a>]
输出：逐句表 + 曲调段映射 + 鉴赏提示
"""
import argparse
import json
import os
import subprocess
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_lines(path):
    d = json.load(open(path, encoding="utf-8-sig"))
    items = d["data"]["items"]
    return [{"i": i, "from": it["from"], "to": it["to"], "text": it["content"]}
            for i, it in enumerate(items)]


def tune_segments(audio_path, win_s=5.0):
    wav = audio_path + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
                    "-f", "wav", wav], capture_output=True, timeout=300)
    with wave.open(wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(wav)
    win = int(16000 * win_s)
    n = len(pcm) // win
    rows = []
    for i in range(n):
        seg = pcm[i * win:(i + 1) * win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / 16000)
        cent = float(np.sum(freqs * sp) / (np.sum(sp) + 1e-9))
        rows.append({"t": i * win_s, "rms": rms, "cent": cent})
    segs = []
    for i, r in enumerate(rows):
        kind = "静音" if r["rms"] < 0.004 else ("强" if r["rms"] >= 0.05 else "中")
        segs.append({"t0": r["t"], "t1": r["t"] + win_s, "kind": kind})
    return segs


def map_context(lines, segs):
    out = []
    for ln in lines:
        mid = (ln["from"] + ln["to"]) / 2
        kind = "无音轨"
        for s in segs:
            if s["t0"] <= mid < s["t1"]:
                kind = s["kind"]
                break
        out.append({**ln, "energy": kind})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subtitle_json")
    ap.add_argument("--audio")
    args = ap.parse_args()

    lines = load_lines(args.subtitle_json)
    segs = tune_segments(args.audio) if args.audio else []
    ctx = map_context(lines, segs)
    print("逐句表（%d 句）：" % len(ctx))
    for c in ctx:
        print("[%02d] %6.1f-%6.1f 能量=%s  %s" % (c["i"], c["from"], c["to"],
                                                  c["energy"], c["text"]))
    print()
    print("鉴赏方法：")
    print(" 1) 读第 1 句，先猜此刻感情（一句话）")
    print(" 2) 读第 2 句，连第 1 句一起，重新猜（允许改判）")
    print(" 3) 每句都带上此前的句子猜，标记「改了什么、为什么」")
    print(" 4) 全部读完，回溯：哪句判断错了、后来哪句推翻了它（回溯性重构）")
    print(" 5) 数据只作佐证，不算感情——感情靠猜，猜要被否定逼着改")

    json.dump({"lines": ctx, "method": "逐句猜→带前文重猜→回溯否定"},
              open(os.path.join(HERE, "_savor_out.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
