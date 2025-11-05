"""Simplified Streamlit application entry point."""

from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from zhipuai import ZhipuAI


Message = Dict[str, str]


class MyQAApp:
    """Console-style Streamlit interface for the 校园人工智能助手."""

    def __init__(self) -> None:
        self.api_key = self._load_api_key()
        self.client = ZhipuAI(api_key=self.api_key)
        self.response = None
        self.messages: List[Message] = st.session_state.setdefault("legacy_messages", [])

    def run(self) -> None:
        st.title("💬 邮邮助手")
        st.caption("🚀 一款北邮学生出品的校园人工智能助手")

        if not self.messages:
            welcome = {"role": "assistant", "content": "How can I help you?"}
            self.messages.append(welcome)

        for message in self.messages:
            st.write(f"{message['role']}: {message['content']}")

        prompt = st.text_input("请输入您的问题：", key="legacy_prompt")
        if prompt:
            self.ask_question(prompt)
            st.session_state["legacy_prompt"] = ""

    def ask_question(self, prompt: str) -> None:
        user_message = {"role": "user", "content": prompt}
        self.messages.append(user_message)
        st.write(f"{user_message['role']}: {user_message['content']}")

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

        chunks: List[str] = []
        for chunk in self.response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)

        assistant_response = "".join(chunks).strip()
        if assistant_response:
            assistant_message = {"role": "assistant", "content": assistant_response}
            self.messages.append(assistant_message)
            st.write(f"{assistant_message['role']}: {assistant_message['content']}")

    @staticmethod
    def _load_api_key() -> str:
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
