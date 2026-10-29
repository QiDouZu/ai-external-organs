# -*- coding: utf-8 -*-
"""voice_timeline.py —— 逐 5 秒窗的人声背景化特征时间序列（后台批量跑）。

输出：质心、500-2k 占比、200-500 占比、静音比、RMS —— 供鉴赏看「包裹感」怎么铺开。
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
EA = os.path.join(HERE, "_ea_cache")
WIN = 5.0


def load_audio(path):
    tmp = path + ".vt.wav"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp],
                   capture_output=True, check=True)
    with wave.open(tmp, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(tmp)
    return x, sr


def window_feat(seg, sr):
    n = len(seg)
    if n < 256:
        return None
    f = np.fft.rfftfreq(n, 1 / sr)
    mag = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    tot = mag.sum() + 1e-9
    def band(lo, hi):
        return round(float(mag[(f >= lo) & (f < hi)].sum() / tot) * 100, 1)
    rms = float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
    return {"centroid": round(float((f * mag).sum() / tot)),
            "mid": band(500, 2000), "low_mid": band(200, 500),
            "low": band(0, 200), "rms": round(rms, 1)}


def run(path, label):
    x, sr = load_audio(path)
    step = int(WIN * sr)
    rows = []
    for i in range(0, len(x) - step, step):
        feat = window_feat(x[i:i + step], sr)
        if feat:
            rows.append({"t": round(i / sr, 1), **feat})
    out_path = os.path.join(HERE, "_timeline_%s.json" % label)
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    mids = [r["mid"] for r in rows]
    cent = [r["centroid"] for r in rows]
    print("%s: %d 窗 | 500-2k 均值 %.1f%% 波动 %d-%d | 质心均值 %d 波动 %d-%d"
          % (label, len(rows), np.mean(mids), min(mids), max(mids),
             np.mean(cent), min(cent), max(cent)))
    return out_path


if __name__ == "__main__":
    for path, label in [("BV12pp16eEfh.m4a", "单人翻唱"), ("BV14GrDYAEzL.m4a", "群唱颂歌")]:
        p = os.path.join(EA, path)
        if os.path.exists(p):
            run(p, label)
