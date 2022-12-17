import pickle as pickle
import os
import pandas as pd
import torch
from tqdm.auto import tqdm

def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset
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
    
    def add_special_enti(self,df,marker_mode= 'TEM_prunct'):
        eng= ['PER', 'ORG', 'DAT', 'LOC', 'POH', 'NOH']
        kor = ['사람', '기관',  '날짜', '장소','기타 명사','수량']
        change_dict = {i : j for i,j in zip(eng,kor)}
        def change_enti(sub,obj,marker_mode = 'TEM_prunct'):
            if marker_mode == 'TEM_prunct':
                marked_sub = ['@']+['*']+list(change_dict[sub['type']]) + ['*']+list(sub['word'])+['@']
                marked_obj = ['#']+['^']+list(change_dict[obj['type']]) + ['^']+list(obj['word'])+['#']
            elif marker_mode == 'TEM':
                marked_sub = ['<s:']+list(change_dict[sub['type']]) + ['>']+list(sub['word'])+['</s:']+list(change_dict[sub['type']]) + ['>']
                marked_obj = ['<s:']+list(change_dict[obj['type']]) + ['>']+list(obj['word'])+['</s:']+list(change_dict[obj['type']]) + ['>']
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

    
def bin_num_to_label(cfg, label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    dict_num_to_label = {0 : 'no_relation', 1 : 'relation' }
    origin_label = []

    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label

def per_num_to_label(cfg, label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    per_label_list = ['per:title', 
       'per:employee_of', 'per:product', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'per:parents','per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'per:religion']
    dict_num_to_label = {i :v for i,v in enumerate(per_label_list)}
    origin_label = []
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label

def org_num_to_label(cfg, label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    org_label_list = ['org:top_members/employees', 'org:members',
       'org:product', 'org:alternate_names','org:place_of_headquarters',
       'org:number_of_employees/members','org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'org:dissolved','org:founded_by']
    dict_num_to_label = {i :v for i,v in enumerate(org_label_list)}
    origin_label = []
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label


def change_prob(prob,mode ='bin'):
    total_prob = [0 for _ in range(30)]
    if mode == 'bin':
        bin_idx = [0,1]
        for i,v in enumerate(bin_idx):
            total_prob[v] = prob[i]
    elif mode == 'per':
        per_idx = [4,6,8,10,11,12,13,14,15,16,17,21,23,24,25,26,27,29]
        for i,v in enumerate(per_idx):
            total_prob[v] = prob[i]
    elif mode == 'org':
        org_idx = [1,2,3,5,7,9,18,19,20,22,28]
        for i,v in enumerate(org_idx):
            total_prob[v] = prob[i]
    return total_prob
                
            
        
         