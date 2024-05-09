import streamlit as st
from zhipuai import ZhipuAI

class MyQAApp:
    def __init__(self):
        self.client = ZhipuAI(api_key="50ea3c031b07edc77a6a640ccb1526d1.NUhtei288b3OrwF4")
        self.response = None
        self.messages = []

    def run(self):
        st.title("💬 邮邮助手")
        st.caption("🚀 一款北邮学生出品的校园人工智能助手")
        self.get_api_key()

        if not self.messages:
            self.messages.append({"role": "assistant", "content": "How can I help you?"})

        for msg in self.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input():
            self.ask_question(prompt)

    def get_api_key(self):
        with st.sidebar:
            self.api_key = st.text_input("ZhipuAI API Key", key="chatbot_api_key", type="password")
            "[Get a ZhipuAI API key](https://www.zhipuai.cn/)"
            "[View the source code](https://github.com/your/repository)"
            "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/your/repository?quickstart=1)"

    def ask_question(self, prompt):
        if not self.api_key:
            st.info("Please add your ZhipuAI API key to continue.")
            return

        self.messages.append({"role": "user", "content": prompt})
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
        st.chat_message("assistant").write(msg)


if __name__ == '__main__':
    qa_app = MyQAApp()
    qa_app.run()
