import json
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class KEIDDataset(Dataset):
    def __init__(self, data_path, type_mapping):
        # 加载数据
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # 存储类型与编号的映射
        self.type_mapping = type_mapping
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取数据项
        item = self.data[idx]
        # 输入句子
        input_text = item["query"]
        if item["type"] == "Non edited":
            rephrased_text = input_text
        else:
            rephrased_text = item["paraphrased_query"]
        # 标签转换为编号
        label = self.type_mapping[item["type"]]
        
        return input_text, rephrased_text, label
