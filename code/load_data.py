import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, dataset,labels,tokenizer,marker_mode= 'TEM_prunct'):
        self.labels = labels
        self.tokenizer = tokenizer
        self.new_tokens = []
        if self.args.input_format == 'entity_marker':
            self.new_tokens = ['<subj>', '</subj>', '<obj>', '</obj>']
        self.tokenizer.add_tokens(self.new_tokens)
        self.marker_mode = marker_mode
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
            text = self.add_special_enti(item)
            # text = '[SEP]'.join([concat_entity, item['sentence']])
            outputs = self.tokenizer(text, add_special_tokens=True,
                                          truncation=True,
                                          return_tensors="pt",
                                          padding='max_length',
                                          max_length=256
                                    )
            data.append(outputs)
        return data
    
    def add_special_enti(self,df,marker_mode= self.marker_mode):
        def change_enti(sub,obj,marker_mode = self.marker_mode):
            if marker_mode == 'TEM_prunct':
                marked_sub = ['@']+['*']+list(sub['type']) + ['*']+list(sub['word'])+['@']
                marked_obj = ['#']+['^']+list(obj['type']) + ['^']+list(obj['word'])+['#']
            elif marker_mode == 'TEM':
                marked_sub = ['<s:']+list(sub['type']) + ['>']+list(sub['word'])+['</s:']+list(sub['type']) + ['>']
                marked_obj = ['<s:']+list(obj['type']) + ['>']+list(obj['word'])+['</s:']+list(obj['type']) + ['>']
            elif marker_mode == "EM":
                marked_sub = ['<subj>']+list(sub['word'])+['</subj>']
                marked_obj = ['<obj>']+list(obj['word'])+['</obj>']
            return marked_sub, marked_obj
        marked = []
        sub = eval(df['subject_entity'])
        s_s, s_e = sub['start_idx'], sub['end_idx']+1
        obj = eval(df['object_entity'])
        o_s, o_e = obj['start_idx'], obj['end_idx']+1
        marked_sub,marked_obj = change_enti(sub,obj)
        if s_s < o_s:
            marked += df['sentence'][:s_s]
            marked += marked_sub
            marked += df['sentence'][s_e:o_s]
            marked += marked_obj
            marked += df['sentence'][o_e:]
            marked = ''.join(marked)
        else:
            marked += df['sentence'][:o_s]
            marked += marked_obj
            marked += df['sentence'][o_e:s_s]
            marked += marked_sub
            marked += df['sentence'][s_e:]
            marked = ''.join(marked)
        return marked

    
def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset
