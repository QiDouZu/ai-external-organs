# 云端端口：证据包一次推理

> 设计要点：多个访问（帧、字幕、音频特征）**合成一个证据包**，一次请求云端推理，收回结构化结果。
> 不是各模块各调一次模型——那会让证据互相看不见，也让云端重复计费。

## 结构

```
本地采集（cloud_savor.py）             云端端口（cloud_port.py）
  帧：首尾 + 均匀，缩 768，base64   ──▶ POST /savor {evidence}
  字幕：官方/转写 前 80 句                 │
  音频：逐 5s 窗质心/中频/静音             ▼
                                    组装 messages：
                                      system=鉴赏方法
                                      user=证据 JSON + 全部帧图
                                          │
                                    一次 chat.completions
                                          │
                              ◀── {segments:[carry/read/judge/change],
                                    retrospect, falsify}
```

## 为什么一次

- 证据之间要能互相印证（帧 ↔ 字幕 ↔ 音频），分开调模型等于把「对账」丢掉。
- 一次请求 = 一次成本、一次上下文，不会因分段调用产生「各段自说各话」。
- 结果结构化：逐段读法的四栏（带什么/读到什么/判断/改判）+ 回溯 + 可证伪。

## 端口约定

| 项 | 值 |
|---|---|
| 监听 | 127.0.0.1:53414（默认，仅本机） |
| POST /savor | 证据包 → 鉴赏 JSON |
| GET /health | `{ok, model, key}` |
| key | `secrets/agnes.key` 或环境变量 `AGNES_KEY`（不进日志、不回显） |
| 模型 | 默认 `agnes-2.5-flash`，`--model` 可换 |

## 证据包字段

```json
{
  "video": {"bv": "..."},
  "audio": {"dur": 222.4, "windows": [{"t": 0, "centroid": 306, "mid500_2k": 11.1}]},
  "subtitle_segments": [{"from": 0.0, "to": 2.1, "text": "..."}],
  "frames": [{"t": 0.0, "b64": "<jpeg>", "mime": "image/jpeg"}]
}
```

## 已知边界

- 无官方字幕的视频走本地转写兜底；转写本身是猜，不能当原文。
- 群唱/失真人声转写不可靠——这是特征不是缺陷（词不可辨正是「背景化」的证据）。
- 云端输出是鉴赏，不是判对错；每次都要保留 `falsify`（怎么推翻它）。
