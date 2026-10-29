# -*- coding: utf-8 -*-
"""cloud_port.py —— 本地云端端口：收证据包，一次请求送云端鉴赏，收回结果。

设计（阿余 2026-10-28）：
- 不是各模块各调一次模型，是「多个访问（字幕/帧/音频特征）结合成一个证据包」，
  在云端一次推理，收回结构化结果。
- 端口：POST /savor（JSON 证据包） → 调 Agnes chat（OpenAI 兼容，带图）。
- GET /health 检查。

运行：python cloud_port.py [--port 53414] [--model agnes-2.5-flash]
key：读 secrets/agnes.key（工作区根），或环境 AGNES_KEY。
"""
import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(HERE))  # 工作区根目录（默认取脚本目录的上两级）
KEY_FILE = os.path.join(WORKSPACE, "secrets", "agnes.key")
API = "https://apihub.agnes-ai.com/v1/chat/completions"
MODEL = "agnes-2.5-flash"
MAX_IMAGES = 8
MAX_CHARS = 6000


def load_key():
    env = os.environ.get("AGNES_KEY", "").strip()
    if env:
        return env
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE, encoding="utf-8").read().strip()
    return None


def build_messages(evidence):
    """证据包 → 一次 cloud 请求的 messages：方法 + 证据文本 + 帧图。"""
    frames = evidence.get("frames") or []
    # 压缩证据文本，防超长
    ev = dict(evidence)
    ev["frames"] = [{"t": f.get("t"), "note": "见附图"} for f in frames]
    txt = json.dumps(ev, ensure_ascii=False, indent=1)
    if len(txt) > MAX_CHARS:
        txt = txt[:MAX_CHARS] + "\n…(截断)"

    system = (
        "你是一位视频/音乐鉴赏者。方法：逐段读，不是一次下结论。"
        "每段先写下「带着：上一段留下了什么」，再读本段（字幕+帧+音频特征），"
        "段末记「改判：改了什么、因为哪句/哪帧/哪段」。最后写「回溯」与「可证伪」。"
        "判断允许矛盾；矛盾是支点。给出你觉得成立的判断，同时写怎么推翻它。"
    )
    user_text = (
        "证据包（字幕/音频特征/帧清单）如下：\n" + txt +
        "\n\n请按逐段读法输出 JSON：{\"segments\": [{\"range\": \"0-30s\", "
        "\"carry\": \"带着…\", \"read\": \"读到…\", \"judge\": \"判断…\", "
        "\"change\": \"改判…\"}], \"retrospect\": \"回溯…\", \"falsify\": [\"…\"]}。"
        "只输出 JSON，不要解释。"
    )
    content = [{"type": "text", "text": user_text}]
    for f in frames[:MAX_IMAGES]:
        b64 = f.get("b64") or ""
        mime = f.get("mime", "image/jpeg")
        if not b64:
            continue
        content.append({"type": "image_url",
                        "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}})
    return [{"role": "system", "content": system},
            {"role": "user", "content": content}]


def cloud_chat(messages, model=MODEL, timeout=300):
    key = load_key()
    if not key:
        raise RuntimeError("无 agnes key（secrets/agnes.key 或 AGNES_KEY）")
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 0.7, "max_tokens": 3000}).encode("utf-8")
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        msg = ""
        try:
            msg = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise RuntimeError("HTTP %s %s" % (e.code, msg))


class Handler(BaseHTTPRequestHandler):
    model = MODEL

    def log_message(self, fmt, *args):  # 安静
        pass

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": self.model,
                             "key": bool(load_key())})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/savor":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            evidence = body.get("evidence") or body
            messages = build_messages(evidence)
            raw = cloud_chat(messages, model=self.model)
            # 抽 JSON（模型可能包了 ```json）
            m = re.search(r"\{.*\}", raw, re.S)
            result = json.loads(m.group(0)) if m else {"raw": raw}
            self._json(200, {"ok": True, "result": result})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"ok": False, "error": str(e)[:500]})


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=53414)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("cloud_port: http://127.0.0.1:%d  model=%s  key=%s"
          % (args.port, MODEL, "yes" if load_key() else "NO"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
