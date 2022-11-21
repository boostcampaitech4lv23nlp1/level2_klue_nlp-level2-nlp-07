import pickle as pickle
import os
import pandas as pd
import torch
from tqdm.auto import tqdm
# class RE_Dataset(torch.utils.data.Dataset):
#     """ Dataset 구성을 위한 class."""
#     def __init__(self, dataset,labels,tokenizer):
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.dataset = self.tokenizing(dataset)
#     def __getitem__(self, idx):
#         if len(self.labels) ==0:
#             return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
#                     'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
#                     'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0)
#                            }
#         else:
#             return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
#                     'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
#                     'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
#                     'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
#     def __len__(self):
#         return len(self.dataset)
#     def tokenizing(self,dataframe):
#         data = []
#         for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
#             # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
#             concat_entity = eval(item['subject_entity'])['word'] + '[SEP]' + eval(item['object_entity'])['word']
#             # text = '[SEP]'.join([concat_entity, item['sentence']])
#             outputs = self.tokenizer(concat_entity,item['sentence'], add_special_tokens=True,
#                                           truncation=True,
#                                           return_tensors="pt",
#                                           padding='max_length',
#                                           max_length=128
#                                     )
#             data.append(outputs)
#         return data

       
class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# eval 함수를 활용해서 entity 내의 정보들을 dictionary로 받고 활용할 수 있게 데이터셋으로 만듦
def unzip_entity(dataset):
    subject_word = []
    subject_start_idx = []
    subject_end_idx = []
    subject_entity = []
    object_word = []
    object_start_idx = []
    object_end_idx = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'],dataset['object_entity']):
        i = eval(i)
        j = eval(j)
        subject_word.append(i['word'])
        subject_start_idx.append(int(i['start_idx']))
        subject_end_idx.append(int(i['end_idx']))
        subject_entity.append(i['type'])
        object_word.append(j['word'])
        object_start_idx.append(int(j['start_idx']))
        object_end_idx.append(int(j['end_idx']))
        object_entity.append(j['type'])
    out_dataset = pd.DataFrame({'id':dataset['id'],'sentence':dataset['sentence'],\
        'subject_word':subject_word,'subject_start_idx':subject_start_idx,'subject_end_idx':subject_end_idx,'subject_entity':subject_entity,\
        'object_word':object_word,'object_start_idx':object_start_idx,'object_end_idx':object_end_idx,'object_entity':object_entity,\
        'label':dataset['label'],})
    return out_dataset

# 필요한 column들만 뽑아서 쓸 수 있게 간단하게 작성
def column_selection(dataset, columns):
  out_columns = dataset.loc[:,columns]
  return out_columns

# token1 column, token2 column을 받아서 여러개의 column을 사용해 sentence1을 만들수 있게 customize
def make_sentence(dataset,token1_column,token2_column):
    concat_entity = []
    for i in range(len(dataset)):
        temp = ''
        e01 = ' '.join(dataset.loc[i,token1_column].values)
        e02 = ' '.join(dataset.loc[i,token2_column].values)
        temp = e01 + ' [SEP] ' + e02
        concat_entity.append(temp)
    return concat_entity, list(dataset['sentence'])

# subject word와 object word를 바꾸고 그 내용을 원 문장과 다른 column들에 반영할 수 있게 만든 method
# 다만 동일 길이의 list를 넣어줘야 함.
def change_sentence(dataset,subject_word_list,obejct_word_list):
    subject_word = []
    subject_start_idx = []
    subject_end_idx = []
    subject_entity = []
    object_word = []
    object_start_idx = []
    object_end_idx = []
    object_entity = []
    sentence = []
    label = []
    for (i, d), s, o in zip(dataset.iterrows(),subject_word_list,obejct_word_list):    
        if d['subject_start_idx'] < d['object_start_idx']:
            subject_start_idx.append(d['subject_start_idx'])
            subject_end_idx.append(d['subject_end_idx'] + len(s) - len(d['subject_word']))
            object_start_idx.append(d['object_start_idx'] + len(s) - len(d['subject_word']))
            object_end_idx.append(d['object_start_idx'] + max(len(d['object_word']), len(o)) -1)
        else:
            object_start_idx.append(d['object_start_idx'])
            object_end_idx.append(d['object_end_idx'] + len(o) - len(d['object_word']))
            subject_start_idx.append(d['subject_start_idx'] + len(o) - len(d['object_word']))
            subject_end_idx.append(d['subject_start_idx'] + max(len(d['subject_word']), len(s)) -1)
        temp = ''
        temp = d['sentence'][:d['subject_start_idx']]+d['sentence'][d['subject_start_idx']:].replace(d['subject_word'],s)
        sentence.append(temp[:d['object_start_idx']]+temp[d['object_start_idx']:].replace(d['object_word'],o))
        subject_word = s
        object_word = o
        subject_entity = None
        object_entity = None 
        label.append('no_relation')
    out_dataset = pd.DataFrame({'id':dataset['id'],'sentence':sentence,\
        'subject_word':subject_word,'subject_start_idx':subject_start_idx,'subject_end_idx':subject_end_idx,'subject_entity':subject_entity,\
        'object_word':object_word,'object_start_idx':object_start_idx,'object_end_idx':object_end_idx,'object_entity':object_entity,\
        'label':label,})
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset

def tokenized_dataset(one_sentence,sentence1, sentence2, tokenizer):
    if one_sentence:
        tokenized_sentences = tokenizer(
            sentence1 + ' [SEP] '  + sentence2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
    else:
        tokenized_sentences = tokenizer(
            sentence1,
            sentence2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
    return tokenized_sentences
