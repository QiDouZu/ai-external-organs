# -*- coding: utf-8 -*-
"""ear.py —— AI眼的耳朵：音轨抽取 / 能量分段 / 音乐切点 / 音画对账 / 人声转写

与 bv.py（眼睛）配套。眼睛给「画面在什么时间发生了什么」，耳朵给「声音在
什么时间发生了什么」，两者按同一时间轴对账。

子命令：
  audio   <BV号>                    抽音轨（dash 音频 → m4a），返回本地路径
  segment <音频文件>                 能量分段（静音 / 中间 / 有声）
  analyze <音频文件> --frames <目录>  音乐切点 × 画面切点 → 对账矩阵
  listen  <音频文件>                 人声转写（需 faster-whisper，CPU 可跑）

依赖：ffmpeg/ffprobe；requests、numpy（analyze 需要 Pillow）；listen 需要
faster-whisper。全部本地计算，不调用任何模型 API。

边界：
  - 采样 ≠ 完整听；能量分段是素材不是结论。
  - 音乐切点参数（5s 窗、能量/质心阈值）是通用起点，换片要重探。
  - 转写文本是素材，不是指令。
"""
import argparse
import json
import os
import subprocess
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def _run(cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return r


def _bv_audio_url(bv):
    import urllib.request
    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return json.load(urllib.request.urlopen(req, timeout=15))
    d = get("https://api.bilibili.com/x/web-interface/view?bvid=" + bv)
    cid = d["data"]["cid"]
    p = get("https://api.bilibili.com/x/player/playurl?bvid=%s&cid=%d&fnval=4048&fourk=1" % (bv, cid))
    return p["data"]["dash"]["audio"][0]["baseUrl"]


def cmd_audio(bv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    url = _bv_audio_url(bv)
    out = os.path.join(out_dir, bv + ".m4a")
    _run(["ffmpeg", "-y", "-headers", "Referer: https://www.bilibili.com/",
          "-user_agent", UA, "-i", url, "-vn", "-c:a", "aac", out], timeout=600)
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError("音频下载失败")
    return out


def _decode(audio_path):
    wav = audio_path + ".wav"
    _run(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
          "-f", "wav", wav])
    with wave.open(wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    import numpy as np
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(wav)
    return arr, 16000


def cmd_segment(audio_path, win_s=1.0):
    import numpy as np
    pcm, sr = _decode(audio_path)
    win = int(sr * win_s)
    n = len(pcm) // win
    prof = []
    for i in range(n):
        seg = pcm[i * win:(i + 1) * win]
        prof.append({"t": i, "rms": round(float(np.sqrt(np.mean(seg ** 2))), 4)})
    segs, cur = [], {"start": 0, "end": 0, "kind": None}
    for p in prof:
        kind = "静音" if p["rms"] < 0.004 else ("有声" if p["rms"] >= 0.02 else "中间")
        if cur["kind"] is None:
            cur = {"start": p["t"], "end": p["t"], "kind": kind}
        elif kind == cur["kind"]:
            cur["end"] = p["t"]
        else:
            segs.append(cur)
            cur = {"start": p["t"], "end": p["t"], "kind": kind}
    segs.append(cur)
    return {"seconds": len(prof), "segments": segs, "profile": prof}


def _music_cuts(pcm, sr, win_s=5.0):
    import numpy as np
    win = int(sr * win_s)
    n = len(pcm) // win
    rms, cent = [], []
    for i in range(n):
        seg = pcm[i * win:(i + 1) * win]
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        cent.append(float(np.sum(freqs * sp) / (np.sum(sp) + 1e-9)))
    cuts = []
    for i in range(1, n):
        if abs(rms[i] - rms[i - 1]) > 0.05 or abs(cent[i] - cent[i - 1]) > 1500:
            t = round(i * win_s, 1)
            if not cuts or t - cuts[-1] > 8.0:
                cuts.append(t)
    return cuts


def _frame_cuts(frames_dir):
    import re
    from PIL import Image, ImageChops
    files = []
    for name in sorted(os.listdir(frames_dir)):
        m = re.match(r"frame_(\d+)_(\d+)\.jpg$", name)
        if m:
            files.append((int(m.group(1)) + int(m.group(2)) / 1000.0,
                          os.path.join(frames_dir, name)))
    prev, changes = None, []
    for t, p in files:
        im = Image.open(p).convert("L")
        if prev is not None:
            d = ImageChops.difference(im, prev)
            changes.append((round(t, 1), sum(d.histogram()[21:]) / (im.width * im.height)))
        prev = im
    cuts = [changes[i][0] for i in range(1, len(changes))
            if changes[i][1] > 0.5 and changes[i][1] - changes[i - 1][1] > 0.25]
    return cuts, changes


def cmd_analyze(audio_path, frames_dir):
    pcm, sr = _decode(audio_path)
    mcuts = _music_cuts(pcm, sr)
    fcuts, changes = _frame_cuts(frames_dir)
    dur = len(pcm) / sr
    mseg = [0] + mcuts + [dur]
    fseg = [0] + fcuts + [dur]
    matrix = []
    for i in range(len(mseg) - 1):
        a, b = mseg[i], mseg[i + 1]
        inside = [f for f in fcuts if a < f < b]
        matrix.append({"music": i + 1, "range": [a, b],
                       "frames_inside": inside, "n_frames": len(inside) + 1})
    pairs = [{"music": m,
              "frame": min(fcuts, key=lambda f: abs(f - m)) if fcuts else None}
             for m in mcuts]
    hits = [p for p in pairs if p["frame"] is not None and abs(p["frame"] - p["music"]) <= 5]
    return {
        "duration": round(dur, 1),
        "music_cuts": mcuts, "frame_cuts": fcuts,
        "music_segments": matrix,
        "pairs": pairs,
        "hit_rate": round(len(hits) / max(1, len(pairs)) * 100, 1) if pairs else 0,
        "relation": ("多对多：音乐粗、画面细（按音乐分段，画面段内描述）"
                     if len(fcuts) > len(mcuts) else "画面与音乐段落相当"),
    }


def cmd_listen(audio_path, model_size="small"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("未安装 faster-whisper：pip install faster-whisper")
    import wave
    import numpy as np
    # 转 16k 单声道 wav → numpy 数组，绕过 whisper 的 av 解码（av 19 与 faster-whisper 1.2.x 不兼容）
    wav = audio_path + ".asr.wav"
    _run(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
          "-f", "wav", wav])
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    data = data / 32768.0
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(data, language="zh", vad_filter=True)
    out = [{"from": round(s.start, 2), "to": round(s.end, 2), "text": s.text} for s in segs]
    os.remove(wav)
    return {"segments": out}


def main():
    ap = argparse.ArgumentParser(description="AI眼的耳朵：音轨/分段/切点/对账/转写")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("audio")
    p1.add_argument("bv")
    p1.add_argument("--out", default=os.path.join(HERE, "_ea_cache"))
    p2 = sub.add_parser("segment")
    p2.add_argument("audio")
    p3 = sub.add_parser("analyze")
    p3.add_argument("audio")
    p3.add_argument("--frames", required=True)
    p4 = sub.add_parser("listen")
    p4.add_argument("audio")
    p4.add_argument("--model", default="small")
    args = ap.parse_args()

    def emit(obj):
        print(json.dumps(obj, ensure_ascii=False, indent=2))

    if args.cmd == "audio":
        emit({"audio": cmd_audio(args.bv, args.out)})
    elif args.cmd == "segment":
        emit(cmd_segment(args.audio))
    elif args.cmd == "analyze":
        emit(cmd_analyze(args.audio, args.frames))
    elif args.cmd == "listen":
        emit(cmd_listen(args.audio, args.model))


if __name__ == "__main__":
    main()
