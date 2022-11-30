import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, dataset,labels,tokenizer,cfg):
        self.labels = labels
        self.tokenizer = tokenizer
        self.new_tokens = []
        self.marker_mode = cfg.train.marker_mode
        if self.marker_mode == 'EMask':
            self.new_tokens = ['<subj-ORG>','<subj-PER>','<obj-ORG>','<obj-PER>','<obj-DAT>','<obj-LOC>','<obj-POH>','<obj-NOH>']
        elif self.marker_mode == "EM":
            self.new_tokens = ['<subj>', '</subj>', '<obj>', '</obj>']
        elif self.marker_mode == "TEM":
            self.new_tokens = ['<s:ORG>', '<s:PER>', '<o:ORG>', '<o:PER>', '<o:DAT>', '<o:LOC>', '<o:POH>', '<o:NOH>', '</s:ORG>', '</s:PER>', '</o:ORG>', '</o:PER>', '</o:DAT>', '</o:LOC>', '</o:POH>', '</o:NOH>']
        self.tokenizer.add_tokens(self.new_tokens)
        
        self.dataset = self.tokenizing(dataset)
        
        self.cfg = cfg
    def __getitem__(self, idx):
        if self.cfg.model.model_name == 'xlm-roberta-base' or self.cfg.model.model_name == 'xlm-roberta-large':
            if self.cfg.train.entity_embedding:
                if len(self.labels) ==0: ## test_data일때
                    return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                            'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                            'Entity_type_embedding': torch.LongTensor(self.dataset[idx]['Entity_type_embedding']).squeeze(0),
                            'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0)                    
                                }
                else:
                    return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                            'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                            'Entity_type_embedding': torch.LongTensor(self.dataset[idx]['Entity_type_embedding']).squeeze(0),
                            'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0),
                            'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
            else:
                if len(self.labels) ==0:
                    return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                            'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),               
                                }
                else:
                    return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                            'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                            'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
        else:
            if self.cfg.train.entity_embedding:
                if len(self.labels) ==0: ## test_data일때
                    return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                            'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                            'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                            'Entity_type_embedding': torch.LongTensor(self.dataset[idx]['Entity_type_embedding']).squeeze(0),
                            'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0)                    
                                }
                else:
                    return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                            'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                            'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                            'Entity_type_embedding': torch.LongTensor(self.dataset[idx]['Entity_type_embedding']).squeeze(0),
                            'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0),
                            'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
            else:
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
            text = self.add_special_enti(item,marker_mode = self.marker_mode)
            # text = '[SEP]'.join([concat_entity, item['sentence']])
            outputs = self.tokenizer(text, add_special_tokens=True,
                                          truncation=True,
                                          return_tensors="pt",
                                    )
            data.append(outputs)
        return data
    
    def add_special_enti(self,df,marker_mode= None):
        def change_enti(sub,obj,marker_mode = None):
            if marker_mode == 'TEM_punct':
                Eng_type_to_Kor = {"PER":"사람", "ORG":"단체", "POH" : "기타", "LOC" : "장소", "NOH" : "수량", "DAT" : "날짜"}
                marked_sub = ['@']+['*']+list(Eng_type_to_Kor[sub['type']]) + ['*']+list(sub['word'])+['@']
                marked_obj = ['#']+['^']+list(Eng_type_to_Kor[obj['type']]) + ['^']+list(obj['word'])+['#']
            elif marker_mode == 'TEM':
                marked_sub = ['<s:']+list(sub['type']) + ['>']+list(sub['word'])+['</s:']+list(sub['type']) + ['>']
                marked_obj = ['<o:']+list(obj['type']) + ['>']+list(obj['word'])+['</o:']+list(obj['type']) + ['>'] ## typo
            elif marker_mode == "EM":
                marked_sub = ['<subj>']+list(sub['word'])+['</subj>']
                marked_obj = ['<obj>']+list(obj['word'])+['</obj>']
            elif marker_mode == "EMask":
                marked_sub = [f'<subj-{sub["type"]}>']
                marked_obj = [f'<obj-{obj["type"]}>']
            return marked_sub, marked_obj
        marked = []
        sub = eval(df['subject_entity'])
        s_s, s_e = sub['start_idx'], sub['end_idx']+1
        obj = eval(df['object_entity'])
        o_s, o_e = obj['start_idx'], obj['end_idx']+1
        marked_sub,marked_obj = change_enti(sub,obj,marker_mode = marker_mode)
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
