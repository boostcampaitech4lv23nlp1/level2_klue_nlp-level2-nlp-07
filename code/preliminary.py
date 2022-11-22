
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

# special token을 포함한 단일 문장 분류 sentence 만들기 위한 전초 작업, 이후 change sentence로 원 문장 변경
def make_sentence1(dataset):
    subject_entity = []
    object_entity = [] 
    for i in range(len(dataset)):
        e01 = ''.join(['<subj>',dataset.loc[i]['subject_word'],'</subj>'])
        e02 = ''.join(['<obj>',dataset.loc[i]['object_word'],'</obj>'])
        subject_entity.append(e01)
        object_entity.append(e02)
    return subject_entity, object_entity

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