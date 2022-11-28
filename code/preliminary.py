import numpy as np
import pandas as pd
import random

def unzip_entity(dataset):
    subjt=dataset['subject_entity'].apply(lambda x: pd.Series(eval(x))).add_prefix('subj_')
    objt=dataset['object_entity'].apply(lambda x: pd.Series(eval(x))).add_prefix('obj_')
    out_dataset = pd.concat([dataset[['id','sentence','label','source']],subjt,objt],axis=1)
    return out_dataset

def noise_data(dataset):
    subjt=dataset['subject_entity'].apply(lambda x: pd.Series(eval(x))).add_prefix('subj_')
    objt=dataset['object_entity'].apply(lambda x: pd.Series(eval(x))).add_prefix('obj_')
    dataset['rnd_id']=dataset['id']
    random.seed(42)
    random.shuffle(dataset['rnd_id'].values)
    rnd_subjWord=subjt.iloc[dataset['rnd_id'],0]
    rnd_objWord =objt.iloc[dataset['rnd_id'],0]
    dataset.drop(columns='rnd_id',inplace=True)
    return rnd_subjWord, rnd_objWord

# 필요한 column들만 뽑아서 쓸 수 있게 간단하게 작성
def column_selion(dataset, columns):
  out_columns = dataset.loc[:,columns]
  return out_columns

# special token을 포함한 단일 문장 분류 sentence 만들기 위한 전초 작업, 이후 change sentence로 원 문장 변경
def make_sentence1(dataset):
    subj_entity = []
    obj_entity = [] 
    for i in range(len(dataset)):
        e01 = ''.join(['<subj>',dataset.loc[i]['subj_word'],'</subj>'])
        e02 = ''.join(['<obj>',dataset.loc[i]['obj_word'],'</obj>'])
        subj_entity.append(e01)
        obj_entity.append(e02)
    return subj_entity, obj_entity

# token1 column, token2 column을 받아서 여러개의 column을 사용해 sentence1을 만들수 있게 customize
def make_sentence2(dataset,token1_column,token2_column):
    concat_entity = []
    for i in range(len(dataset)):
        temp = ''
        e01 = ' '.join(dataset.loc[i,token1_column].values)
        e02 = ' '.join(dataset.loc[i,token2_column].values)
        temp = e01 + ' [SEP] ' + e02
        concat_entity.append(temp)
    return concat_entity, list(dataset['sentence'])


# subj word와 obj word를 바꾸고 그 내용을 원 문장과 다른 column들에 반영할 수 있게 만든 method
# 다만 동일 길이의 list를 넣어줘야 함.
def change_sentence(dataset,subj_word_list,obj_word_list):
    subjt=dataset['subject_entity'].apply(lambda x: pd.Series(eval(x))).add_prefix('subj_')
    objt=dataset['object_entity'].apply(lambda x: pd.Series(eval(x))).add_prefix('obj_')
    concat_dataset = pd.concat([dataset[['id','sentence','label','source']],subjt,objt],axis=1)
    sentence = []
    subj_entity =[]
    obj_entity = []
    label = []
    for (i, d), s, o in zip(concat_dataset.iterrows(),subj_word_list,obj_word_list):   
        subj_dict = {}
        obj_dict = {}
        subj_dict['word'] = s
        obj_dict['word'] = o
        if d['subj_start_idx'] < d['obj_start_idx']:
            subj_dict['start_idx'] = d['subj_start_idx']
            subj_dict['end_idx'] = d['subj_end_idx'] + len(s) - len(d['subj_word'])
            obj_dict['start_idx'] = d['obj_start_idx'] + len(s) - len(d['subj_word'])
            obj_dict['end_idx'] = d['obj_end_idx'] + len(o)-len(d['subj_word'])+len(s) - len(d['obj_word'])
        else:
            obj_dict['start_idx'] = d['obj_start_idx']
            obj_dict['end_idx'] = d['obj_end_idx'] + len(o) - len(d['obj_word'])
            subj_dict['start_idx'] = d['subj_start_idx'] + len(o) - len(d['obj_word'])
            subj_dict['end_idx'] = d['subj_end_idx'] + len(s)-len(d['subj_word'])+len(o) - len(d['obj_word'])
        temp = ''
        temp = d['sentence'][:d['subj_start_idx']]+d['sentence'][d['subj_start_idx']:].replace(d['subj_word'],s,1)
        sentence.append(temp[:d['obj_start_idx']]+temp[d['obj_start_idx']:].replace(d['obj_word'],o,1))
        subj_dict['type'] =None
        obj_dict['type'] =None
        subj_entity.append(str(subj_dict))
        obj_entity.append(str(obj_dict))
        label.append('no_relation')
    out_dataset = pd.DataFrame({'id':concat_dataset['id'],'sentence':sentence,\
        'subject_entity':subj_entity,'object_entity':obj_entity,
        'label':label,'source':concat_dataset['source']})
    return out_dataset