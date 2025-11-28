import os
import requests
import threading
import time

class ChatClient:
    """
    简单的云端大模型客户端封装。
    运行 start() 后会在独立线程中读取终端输入并调用云端 API，输入 '/exit' 停止。
    请通过环境变量 CHAT_API_URL 和 CHAT_API_KEY 配置实际接口。
    """
    def __init__(self, api_url=None, api_key=None, timeout=30):
        self.api_url = api_url or os.getenv("CHAT_API_URL", "https://api.example.com/v1/chat")
        self.api_key = api_key or os.getenv("CHAT_API_KEY", None)
        self.timeout = timeout
        self.running = False
        self.thread = None

    def call_api(self, prompt):
        """
        简单POST调用示例，按目标云端API调整 body/headers/解析。
        返回字符串（模型回复），失败返回错误信息字符串。
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "prompt": prompt,
            "max_tokens": 512
        }
        try:
            resp = requests.post(self.api_url, json=body, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # 下面解析按实际API返回结构调整：
            # 假设返回 {"reply": "..."} 或 {"choices":[{"text":"..."}]}
            if "reply" in data:
                return data["reply"]
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("text", "")
            return str(data)
        except Exception as e:
            return f"[CHAT ERROR] {e}"

    def _input_loop(self):
        print("[CHAT] 聊天模式开启。终端输入与模型对话，输入 /exit 退出聊天模式。")
        while self.running:
            try:
                user = input("You: ")
            except EOFError:
                break
            if not self.running:
                break
            if user.strip() == "/exit":
                break
            if user.strip() == "":
                continue
            reply = self.call_api(user)
            print("Bot:", reply)
            time.sleep(0.05)
        self.running = False
        print("[CHAT] 聊天模式已退出。")

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._input_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        # 尝试唤醒 input 线程（用户需回车），等待线程退出
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None