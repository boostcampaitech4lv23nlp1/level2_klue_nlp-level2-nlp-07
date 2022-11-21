import pickle as pickle
import os
import pandas as pd
import torch
from tqdm.auto import tqdm
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
            # text = '[SEP]'.join([concat_entity, item['sentence']])
            outputs = self.tokenizer(concat_entity,item['sentence'], add_special_tokens=True,
                                          truncation=True,
                                          return_tensors="pt",
                                          padding='max_length',
                                          max_length=128
                                    )
            data.append(outputs)
        return data
# class RE_Dataset(torch.utils.data.Dataset):
#     """ Dataset 구성을 위한 class."""
#     def __init__(self, pair_dataset, labels):
#         self.pair_dataset = pair_dataset
#         self.labels = labels
#     def __getitem__(self, idx):
#         item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#     def __len__(self):
#         return len(self.labels)

# def preprocessing_dataset(dataset):
#     """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
#     subject_entity = []
#     object_entity = []
#     for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
#         i = i[1:-1].split(',')[0].split(':')[1] # text만 추출한다 이 말씀
#         j = j[1:-1].split(',')[0].split(':')[1] # 

#         subject_entity.append(i)
#         object_entity.append(j)
#     out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
#     return out_dataset

def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset

# def tokenized_dataset(dataset, tokenizer):
#     """ tokenizer에 따라 sentence를 tokenizing 합니다."""
#     concat_entity = []
#     for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
#         temp = ''
#         temp = eval(e01)['word'] + '[SEP]' + eval(e02)['word']
#         concat_entity.append(temp)
#     tokenized_sentences = tokenizer(
#           concat_entity,
#           list(dataset['sentence']),
#           return_tensors="pt",
          # padding=True,
#           truncation=True,
#           max_length=128,
#           add_special_tokens=True,
#       )
#     return tokenized_sentences
