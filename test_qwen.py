import os
from openai import OpenAI
# from GMLLM.extractAST.prompts import  PROMPT_COMM
# from GMLLM.extractAST.rules_fallback import BEHAVIOR_RULES
from typing import List
# ALL_BEHAVIOR_TAGS: List[str] = [k for k in BEHAVIOR_RULES.keys()]
import json

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key=os.getenv("OPENAI_API_KEY"),
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="123",
    base_url="http://192.168.1.101:60102/v1"
)
# user_prompt = json.dumps({"allowed_hint": list(ALL_BEHAVIOR_TAGS)}, ensure_ascii=False)
completion = client.chat.completions.create(
    model="qwen-2.5-coder-32b",
    messages=[
        # {"role": "system", "content": PROMPT_COMM},
        {"role": "user", "content": "hello, what's your name?"},
    ],
    # stream=True
)
# for chunk in completion:
#     print(chunk.choices[0].delta.content, end="", flush=True)
print(completion.choices[0].message.content.strip())