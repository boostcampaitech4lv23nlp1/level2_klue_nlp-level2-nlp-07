import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, dataset,labels,tokenizer):
        self.labels = labels
        self.tokenizer = tokenizer
        self.dataset = self.tokenizing(dataset)
        
    def __getitem__(self, idx):
        if len(self.labels) ==0:
            return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                    'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                    'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0)
                           }
        else:
            return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                    'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                    'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                    'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
            
    def __len__(self):
        return len(self.dataset)
    
    def tokenizing(self,dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            concat_entity = eval(item['subject_entity'])['word'] + '[SEP]' + eval(item['object_entity'])['word']
            outputs = self.tokenizer(concat_entity,
                                    item['sentence'], 
                                    add_special_tokens=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    padding='max_length',
                                    max_length=256
                                    )
            data.append(outputs)
        return data
    
def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset
