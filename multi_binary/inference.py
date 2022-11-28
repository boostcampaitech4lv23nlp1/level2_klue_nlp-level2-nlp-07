from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
from utils import *
import pickle as pickle
import numpy as np
from tqdm import tqdm

def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후, batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False) # batch_size= 16
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              token_type_ids=data['token_type_ids'].to(device)
              )
            logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
            output_prob.append(prob)
  
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label,type_pair_id):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    if type_pair_id == None:
        with open('/opt/ml/code/dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
    else:
        label2id = LABEL_TO_ID[type_pair_id]
        dict_num_to_label = {i:label for label, i in label2id.items()}
    for v in label:
        origin_label.append(dict_num_to_label[v])
    return origin_label

def load_test_dataset(dataset_dir,type_pair_id):
    """
    test dataset을 불러온 후, tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    if type_pair_id != None:
        subj_type, obj_type = ID_TO_TYPE_PAIR[type_pair_id].split('_')
        subj_data = test_dataset[test_dataset['subject_entity'].apply(lambda x: eval(x)['type']==subj_type)]
        obj_data = test_dataset[test_dataset['object_entity'].apply(lambda x: eval(x)['type']==obj_type)]
        test_dataset = pd.merge(subj_data, obj_data, how='inner')
    test_label = list(map(int,test_dataset['label'].values))

    return test_dataset['id'], test_dataset, test_label

def test(cfg):
    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.saved_model)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = cfg.data.test_data
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label, tokenizer)
    
    ## predict answer ## 절대 바꾸지 말 것 ##
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
    pred_answer = num_to_label(cfg, pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(cfg.test.output_csv, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.