# from modelscope import snapshot_download
# from sentence_transformers import SentenceTransformer
# from modelscope.hub.api import HubApi
# import os
# # 在下载前登录
# api = HubApi()
# api.login(os.environ.get('MODELSCOPE_TOKEN'))


# # 1. 从 ModelScope 下载模型到本地
# model_dir = snapshot_download('ZhipuAI/gte-base-en-v1.5', cache_dir='./models')

# # 2. 加载本地模型
# model = SentenceTransformer(model_dir)

import os
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5',trust_remote_code=True)


# 2. 你的增量节点名称列表 (假数据)
node_names = [
    "requests.get", 
    "subprocess.Popen", 
    "a1b2_obfuscated_var"
]

# 3. 离线批量提取特征
start_time = time.time()
embeddings = model.encode(node_names, batch_size=128, show_progress_bar=True)
end_time = time.time()
print(f"encode 耗时: {end_time - start_time:.2f} 秒")
# embeddings 现在的 shape 是 (3, 768)

# 4. 把 embeddings 存入你的 PyG Data 对象中，保存为 .pt 文件
print(embeddings.shape)  # 输出 (3, 768)