
# -*- coding: utf-8 -*-
"""
简单 Telegram 推送与命令轮询模块（长轮询）
- 从 config_live_v31.yaml 读取 bot_token 与 chat_id
- send(msg): 发送文本
- start_polling(callback): 后台线程轮询消息，将收到的文本传给 callback(text)
"""
import threading
import time
import json
from typing import Optional, Callable
import requests
import yaml
import os

class TelegramNotifier:
    def __init__(self, cfg_path: str = "config_live_v31.yaml"):
        self.enabled = False
        self.token: Optional[str] = None
        self.chat_id: Optional[str] = None
        self._stop = False
        self._offset = None
        self._thread: Optional[threading.Thread] = None
        self._load_cfg(cfg_path)

    def _load_cfg(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            tg = (cfg.get("telegram") or {}) if isinstance(cfg, dict) else {}
            self.token = tg.get("bot_token") or tg.get("token")
            self.chat_id = str(tg.get("chat_id")) if tg.get("chat_id") is not None else None
            self.enabled = bool(self.token and self.chat_id)
        except Exception:
            self.enabled = False

    # --- 发送文本 ---
    def send(self, text: str):
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text}
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass

    # --- 启动后台轮询，回调处理命令 ---
    def start_polling(self, on_text: Optional[Callable[[str], None]] = None):
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._poll_loop, args=(on_text,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True

    def _poll_loop(self, on_text):
        base = f"https://api.telegram.org/bot{self.token}"
        while not self._stop:
            try:
                params = {}
                if self._offset is not None:
                    params["offset"] = self._offset
                r = requests.get(f"{base}/getUpdates", params=params, timeout=20)
                data = r.json()
                if not data.get("ok"):
                    time.sleep(2)
                    continue
                for upd in data.get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue
                    # 限定 chat_id，防止他人滥用
                    if str(msg.get("chat", {}).get("id")) != str(self.chat_id):
                        continue
                    text = (msg.get("text") or "").strip()
                    if on_text and text:
                        try:
                            on_text(text)
                        except Exception:
                            # 防御性；避免线程崩溃
                            pass
            except Exception:
                time.sleep(2)
            time.sleep(1)
