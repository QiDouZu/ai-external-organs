# -*- coding: utf-8 -*-
"""cloud_savor.py —— 本地采集证据，打包发云端端口，收回鉴赏结果。

用法：
    python cloud_port.py          # 另开一个终端/后台
    python cloud_savor.py BV12pp16eEfh

流程（作者 2026-10-28）：
1. 本地收：帧（首尾+均匀）、字幕（bv.py text）、音频特征（逐窗质心/中频/静音）
2. 合成一个证据包
3. POST http://127.0.0.1:53414/savor → 云端一次推理
4. 收回 JSON，落盘 鉴赏_<BV>_云端.json / .md
"""
import base64
import glob
import io
import json
import os
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".dsh", "bili-vision-cache", "v2")
DEFAULT_EACACHE = os.path.join(HERE, "_ea_cache")
PORT = 53414
N_FRAMES = 7


def _python():
    return sys.executable


def gather_frames(bv, cache_root):
    pat = os.path.join(cache_root, bv, "p1", "*", "frames", "*.jpg")
    files = sorted(glob.glob(pat))
    if not files:
        return []
    picks = [files[0], files[-1]] + [files[i] for i in
                                     _even(range(1, len(files) - 1), N_FRAMES - 2)]
    picks = picks[:N_FRAMES]
    out = []
    from PIL import Image
    for f in picks:
        im = Image.open(f)
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        t = 0.0
        m = os.path.basename(f).split("_")
        try:
            t = int(m[1]) / 10000.0
        except Exception:
            pass
        out.append({"t": round(t, 2), "f": f, "b64": b64, "mime": "image/jpeg"})
    return out


def _even(indices, k):
    cand = list(indices)
    if not cand or k <= 0:
        return []
    if k >= len(cand):
        return cand
    step = (len(cand) - 1) / (k - 1) if k > 1 else 0
    return [cand[int(round(i * step))] for i in range(k)]


def gather_subtitles(bv):
    try:
        p = subprocess.run([_python(), os.path.join(HERE, "bv.py"), "text", bv],
                           capture_output=True, cwd=HERE, timeout=180)
        out = p.stdout.decode("utf-8", "replace").strip()
        m = out.find("{")
        if m >= 0:
            data = json.loads(out[m:])
            segs = data.get("segments") or []
            if segs:
                return [{"from": s.get("from"), "to": s.get("to"), "text": s.get("text")}
                        for s in segs[:80]]
    except Exception:  # noqa: BLE001
        pass
    # 兜底：whisper 转写（ear.py listen）
    try:
        p = subprocess.run([_python(), os.path.join(HERE, "ear.py"), "listen",
                            os.path.join(DEFAULT_EACACHE, bv + ".m4a")],
                           capture_output=True, cwd=HERE, timeout=600)
        out = p.stdout.decode("utf-8", "replace").strip()
        m = out.find("{")
        if m >= 0:
            data = json.loads(out[m:])
            segs = data.get("segments") or []
            return [{"from": s.get("from"), "to": s.get("to"), "text": s.get("text")}
                    for s in segs[:80]]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)[:120]}]
    return []


def gather_audio(bv):
    import numpy as np
    import wave
    path = os.path.join(DEFAULT_EACACHE, bv + ".m4a")
    if not os.path.exists(path):
        return None
    tmp = path + ".cs.wav"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1",
                    "-f", "wav", tmp], capture_output=True, check=True)
    with wave.open(tmp, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    os.remove(tmp)
    win = 5 * sr
    rows = []
    for i in range(0, len(x) - win, win):
        seg = x[i:i + win]
        n = len(seg)
        f = np.fft.rfftfreq(n, 1 / sr)
        mag = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
        tot = mag.sum() + 1e-9
        rows.append({"t": round(i / sr, 1),
                     "centroid": round(float((f * mag).sum() / tot)),
                     "mid500_2k": round(float(mag[(f >= 500) & (f < 2000)].sum() / tot) * 100, 1),
                     "low_mid200_500": round(float(mag[(f >= 200) & (f < 500)].sum() / tot) * 100, 1)})
    return {"dur": round(len(x) / sr, 1), "windows": rows[:60]}


def main():
    bv = sys.argv[1] if len(sys.argv) > 1 else "BV12pp16eEfh"
    frames = gather_frames(bv, DEFAULT_CACHE)
    subs = gather_subtitles(bv)
    audio = gather_audio(bv)
    evidence = {"video": {"bv": bv},
                "audio": audio,
                "subtitle_segments": subs,
                "frames": frames}
    body = json.dumps({"evidence": evidence}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/savor" % PORT, data=body,
                                 method="POST", headers={"Content-Type": "application/json"})
    print("证据包：%d 帧 / %d 字幕句 / 音频 %s"
          % (len(frames), len(subs), "有" if audio else "无"))
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            ret = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print("云端请求失败：", e)
        sys.exit(1)
    if not ret.get("ok"):
        print("云端错误：", ret.get("error"))
        sys.exit(1)
    result = ret.get("result", {})
    out_json = os.path.join(HERE, "鉴赏_%s_云端.json" % bv)
    json.dump(result, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已落盘：", out_json)
    print(json.dumps(result, ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    main()
