import json,os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset

DataDir=Path(os.getcwd())/'data'#数据目录
TrainDataPath=DataDir/'train'/'train.jsonl'#训练集路径
TestDataPath=DataDir/'test_521'/'test.jsonl'#测试集路径

class TextDataset(Dataset):
    def __init__(self,path):
        self.data = self.__get_data(path)
        
    def __get_data(self,path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return np.array(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == '__main__':
    train_dataset=TextDataset(TrainDataPath)
    test_dataset=TextDataset(TestDataPath)
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)