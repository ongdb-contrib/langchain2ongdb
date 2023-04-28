import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

from backend.src2.env import getEnv


class CustomLLM(LLM):
    model_name = 'GPT'
    type = 'gpt3.5'

    def get_gpt3_5(self, msg_list, max_tokens=1200):
        # 设置请求头

        headers = {"Content-type": "application/json"}
        url = getEnv('GPT3_5_HTTP_URL')
        data = {
            "messages": msg_list,
            "temperature": 1,
            "frequency_penalty": 0,
            "max_tokens": max_tokens,
            "presence_penalty": 0
        }

        response = requests.post(url, json=data, headers=headers)

        return response.json()

    def get_gpt4(self, msg):
        # 设置请求头
        # 设置请求头

        headers = {"Content-type": "application/json"}
        url = getEnv('GPT4_HTTP_URL')
        data = {
            "ask_str": msg,

        }

        response = requests.post(url, json=data, headers=headers)
        return response.json()['data']['content']

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.type == 'gpt4':
            return self.get_gpt4(prompt)
        else:
            msg_list = [{"role": "system", "content": prompt}]

            respone = self.get_gpt3_5(msg_list, max_tokens=1200)
            return respone["data"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
