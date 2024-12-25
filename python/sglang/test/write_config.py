import json
from collections import defaultdict
import os 

with open('/home/hadoop-hdp/sglang/python/sglang/test/tune_results.log', 'r', encoding='utf-8') as file:
    data = file.read()

entries = data.strip().split('\n\n')
configs = defaultdict(dict)

for entry in entries:
    lines = entry.split('\n')
    # 提取 M, N, K
    m_line = lines[0]
    m_value = m_line.split(', ')[0].split(': ')[1]
    n_value = m_line.split(', ')[1].split(': ')[1]
    k_value = m_line.split(', ')[2].split(': ')[1]
    
    # 提取配置
    config_line = lines[1].split('selected: ')[1]
    config_parts = config_line.split(', ')
    config_dict = {part.split(': ')[0]: int(part.split(': ')[1]) for part in config_parts if 'None' not in part and 'num_ctas' not in part}
    
    # 组织数据
    configs[(n_value, k_value)][m_value] = config_dict

# 保存为 JSON 文件
from sglang.srt.utils import get_device_name
block_size=[128, 128]
device_name = get_device_name().replace(" ", "_")
for (n_value, k_value), config in configs.items():
    filename = f"N={n_value},K={k_value},{device_name},dtype=fp8_w8a8,block_size={block_size}.json"
    path = os.path.join("/home/hadoop-hdp/sglang/python/sglang/srt/layers/quantization/configs", filename)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Saved to {path}")