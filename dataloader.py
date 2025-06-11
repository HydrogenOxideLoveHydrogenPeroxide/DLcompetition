import json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

DataDir=Path(os.getcwd())/'data'#数据目录
TrainDataPath=DataDir/'train'/'train.jsonl'#训练集路径
#TestDataPath=DataDir/'test_521'/'test.jsonl'#测试集路径    test_dataset=TextDataset(TestDataPath)

class TextDataset(Dataset):
    def __init__(self,path):
        self.data = self.__get_data(path)
        
    def __get_data(self,path)->np.array:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(list(json.loads(line.strip()).values()))
        return np.array(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return texts, labels

def get_data_loader(batch_size,train_per=0.8):
    # 创建数据集
    dataset = TextDataset(TrainDataPath)
    
    # 将数据集划分为训练集和测试集
    train_size = int(train_per * len(dataset))  # 80% 的数据用于训练
    test_size = len(dataset) - train_size  # 剩下的 20% 用于测试
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader
    
if __name__ == '__main__':
    train_dataset=TextDataset(TrainDataPath)
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)