import json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertModel,BertTokenizer

DataDir=Path(os.getcwd())/'data'#数据目录
TrainDataPath=DataDir/'train'/'train_cleaned.jsonl'#训练集路径
BERT_PATH = '/my-bert-base-uncased'

class TextDataset(Dataset):
    def __init__(self,path):
        self.texts,self.labels=self.__get_data(path)
        
    def __get_data(self,path)->np.array:
        texts = []
        labels = []
    
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(json.loads(line)['text'])
                labels.append(json.loads(line)['label'])
        return np.array(texts),np.array(labels)
           
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def save_data(self,path):
        with open(path, 'w', encoding='utf-8') as f:
            for text_encoded, label in zip(self.texts_encoded, self.labels):
                f.write(json.dumps({'text_encoded': text_encoded, 'label': label}) + '\n')

class TextDataLoader(DataLoader):
    def __init__(self,path,batch_size=32,train_percent=0.8):
        self.dataset=TextDataset(path)
        self.batch_size=batch_size
        self.train_size = int(train_percent * len(self.dataset))  # 80% 的数据用于训练
        self.test_size = len(self.dataset) - self.train_size  # 剩下的 20% 用于测试
        self.max_length=512
        
    def get_data_loader(self):
        def collate_fn(batch):
            texts = [x[0] for x in batch]
            labels = torch.tensor([x[1] for x in batch])
            return texts, labels
        train_dataset, test_dataset = random_split(self.dataset, [self.train_size, self.test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        return train_loader, test_loader

def get_data_loader(batch_size,train_per=0.8):
    text_loader=TextDataLoader(TrainDataPath,batch_size=batch_size,train_percent=train_per)
    train_loader,test_loader=text_loader.get_data_loader()
    return train_loader,test_loader

if __name__ == '__main__':
    textDataset=TextDataset(TrainDataPath)
    print(textDataset[1])
    