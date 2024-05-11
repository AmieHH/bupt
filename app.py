import streamlit as st
from zhipuai import ZhipuAI


class MyQAApp:
    def __init__(self):
        # 使用固定的API Key
        self.api_key = "5022cd3dacef67b05354432743d43bbd.5gKOTuIkuWQoLbl9"
        self.client = ZhipuAI(api_key=self.api_key)
        self.response = None
        self.messages = []

    def run(self):
        st.title("💬 邮邮助手")
        st.caption("🚀 一款北邮学生出品的校园人工智能助手")

        # 显示初始的助手消息
        if not self.messages:
            self.messages.append({"role": "assistant", "content": "How can I help you?"})
            st.write("助手: How can I help you?")

        for msg in self.messages:
            st.write(f"{msg['role']}: {msg['content']}")

        if prompt := st.text_input("请输入您的问题："):
            self.ask_question(prompt)

    def ask_question(self, prompt):
        # 修正为正确的键 'role' 而不是 'store_role'
        self.messages.append({"role": "user", "content": prompt})

        # 发送问题并获取回答
        self.response = self.client.chat.completions.create(
            model="glm-4v",
            messages=self.messages,
            stream=True,
            tools=[
                {
                    "type": "retrieval",
                    "retrieval": {
                        "knowledge_id": "1765660633795276800",
                        "prompt_template": "如果用户问文档中的相关问题就直接回答。不是文档里的相关内容你就告诉用户我不太清楚，或者让用户再问的具体一点。不要复述问题，直接开始回答。"
                    }
                }
            ],
        )

        # 从response中读取回答
        msg = ""
        for chunk in self.response:
            msg += chunk.choices[0].delta.content

        self.messages.append({"role": "assistant", "content": msg})
        st.write(f"助手: {msg}")


if __name__ == '__main__':
    qa_app = MyQAApp()
    qa_app.run()
