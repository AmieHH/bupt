"""Streamlit entry point for the 校园人工智能助手."""

from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from zhipuai import ZhipuAI


Message = Dict[str, str]


class MyQAApp:
    """A tiny wrapper around the ZhipuAI chat completion API."""

    def __init__(self) -> None:
        # 使用内置的开发者 API。优先从 Streamlit Secrets 读取，其次读取环境变量。
        self.api_key = self._load_api_key()
        self.client = ZhipuAI(api_key=self.api_key)
        self.response = None
        self.messages: List[Message] = st.session_state.setdefault("messages", [])

    def run(self) -> None:
        st.title("💬 邮邮助手")
        st.caption("🚀 一款北邮学生出品的校园人工智能助手")

        # 不再需要用户输入 API。
        self.display_info()

        if not self.messages:
            welcome = {"role": "assistant", "content": "How can I help you?"}
            self.messages.append(welcome)

        for message in self.messages:
            self._render_message(message)

        if prompt := st.chat_input():
            self.ask_question(prompt)

    def display_info(self) -> None:
        with st.sidebar:
            st.markdown("🔑 Using internal developer API")
            st.markdown("[View the source code](https://github.com/your/repository)")
            st.markdown(
                "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/your/repository?quickstart=1)"
            )

    def ask_question(self, prompt: str) -> None:
        user_message = {"role": "user", "content": prompt}
        self.messages.append(user_message)
        self._render_message(user_message)

        self.response = self.client.chat.completions.create(
            model="glm-4v",
            messages=self.messages,
            stream=True,
            tools=[
                {
                    "type": "retrieval",
                    "retrieval": {
                        "knowledge_id": "1765660633795276800",
                        "prompt_template": "如果用户问文档中的相关问题就直接回答。不是文档里的相关内容你就告诉用户我不太清楚，或者让用户再问的具体一点。不要复述问题，直接开始回答。",
                    },
                }
            ],
        )

        # 从 response 中读取回答，忽略空的增量片段。
        chunks: List[str] = []
        for chunk in self.response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)

        assistant_response = "".join(chunks).strip()
        if assistant_response:
            assistant_message = {"role": "assistant", "content": assistant_response}
            self.messages.append(assistant_message)
            self._render_message(assistant_message)

    def _render_message(self, message: Message) -> None:
        """Render a chat message in Streamlit chat UI."""

        role = message.get("role", "assistant")
        content = message.get("content", "")
        if not content:
            return

        with st.chat_message(role):
            st.markdown(content)

    @staticmethod
    def _load_api_key() -> str:
        """Load the ZhipuAI API key from Streamlit secrets or the environment."""

        api_key = st.secrets.get("ZHIPUAI_API_KEY") if hasattr(st, "secrets") else None
        if not api_key:
            api_key = os.getenv("ZHIPUAI_API_KEY")

        if not api_key:
            st.error("未检测到 ZHIPUAI_API_KEY，请在环境变量或 Streamlit Secrets 中进行配置。")
            raise RuntimeError("Missing ZHIPUAI_API_KEY configuration")

        return api_key


if __name__ == '__main__':
    qa_app = MyQAApp()
    qa_app.run()
