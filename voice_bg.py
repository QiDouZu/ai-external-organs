# -*- coding: utf-8 -*-
"""voice_bg.py —— 人声「背景化」检测：人声失真/群唱到什么程度就从内容变背景。

判据（对比清晰人声）：
- 频谱质心：宽带上移（失真/噪声化）
- 谐波密度：多声部饱和时 200-1500Hz 能量占比升高、峰变宽
- 可辨度：whisper 可转写性（是否有清晰句）
- 静音比：群唱持续不停，静音段少

对比：BV12pp16eEfh（单人人声，清晰） vs BV14GrDYAEzL（群唱颂歌，失真）
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
EA = os.path.join(HERE, "_ea_cache")


def load_audio(path):
    import wave
    tmp = path + ".vb.wav"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp],
                   capture_output=True, check=True)
    with wave.open(tmp, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(tmp)
    return x, sr


def centroid(x, sr, win=0.05):
    """逐窗频谱质心均值（Hz）。"""
    n = int(win * sr)
    step = n // 2
    vals = []
    f = np.fft.rfftfreq(n, 1 / sr)
    for i in range(0, len(x) - n, step):
        seg = x[i:i + n] * np.hanning(n)
        mag = np.abs(np.fft.rfft(seg))
        if mag.sum() < 1e-9:
            continue
        vals.append(float((f * mag).sum() / mag.sum()))
    return float(np.mean(vals)) if vals else 0.0


def band_profile(x, sr):
    """能量分布：低(0-200)/中低(200-500)/中(500-2k)/中高(2k-6k)/高(6k+)。"""
    seg = x[:min(len(x), 60 * sr)]
    n = min(len(seg), 2 ** 16)
    if n < len(seg):
        seg = seg[:n]
    f = np.fft.rfftfreq(n, 1 / sr)
    mag = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    tot = mag.sum() + 1e-9
    bands = {"低0-200": (0, 200), "中低200-500": (200, 500),
             "中500-2k": (500, 2000), "中高2k-6k": (2000, 6000),
             "高6k+": (6000, sr / 2)}
    return {k: round(float(mag[(f >= lo) & (f < hi)].sum() / tot) * 100, 1)
            for k, (lo, hi) in bands.items()}


def rms_db(x, sr):
    win = int(0.1 * sr)
    vals = []
    for i in range(0, len(x) - win, win):
        seg = x[i:i + win]
        vals.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
    vals = np.array(vals)
    rows = []
    quiet = 0
    for j in range(0, len(vals), 10):
        block = vals[j:j + 10]
        med = float(np.median(block))
        if med < -45:
            quiet += 1
        rows.append(med)
    return {"blocks": len(rows), "quiet_ratio_pct": round(quiet / max(len(rows), 1) * 100, 1),
            "median_db": round(float(np.median(vals)), 1)}


def analyze(path, label):
    print("== " + label + " ==")
    x, sr = load_audio(path)
    dur = len(x) / sr
    print("dur %.1fs sr %d" % (dur, sr))
    print("centroid %.0f Hz" % centroid(x, sr))
    print("bands", json.dumps(band_profile(x, sr), ensure_ascii=False))
    print("rms", json.dumps(rms_db(x, sr), ensure_ascii=False))
    return {"label": label, "dur": round(dur, 1), "centroid": round(centroid(x, sr)),
            "bands": band_profile(x, sr), "rms": rms_db(x, sr)}


def main():
    out = []
    for path, label in [("BV12pp16eEfh.m4a", "单人人声·清晰"), ("BV14GrDYAEzL.m4a", "群唱颂歌·失真")]:
        p = os.path.join(EA, path)
        if os.path.exists(p):
            out.append(analyze(p, label))
    json.dump(out, open(os.path.join(HERE, "_voice_bg_out.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved _voice_bg_out.json")


if __name__ == "__main__":
    main()
