from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
import random

def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후, batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False) # batch_size= 16
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        data = {k:v.to(device) for k,v in data.items()}
        with torch.no_grad():
            outputs = model(**data) # default : input, token  다 넣어줬음 원래
            logits = outputs['logits']
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
            output_prob.append(prob)

  
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def load_test_dataset(dataset_dir):
    """
    test dataset을 불러온 후, tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = []
    test_sub_type = list(test_dataset['subject_entity'].apply(lambda x : eval(x)['type']))

    return test_dataset['id'], test_dataset, test_label,test_sub_type

def double_check(output,sub_type,tokenizer):
    new_df = output[output['sub_type']==sub_type]
    RE_data = RE_Dataset(new_df,[],tokenizer)
    return new_df['id'],RE_data

def test(cfg):
    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## load Model & Tokenizer
    bin_tokenizer = AutoTokenizer.from_pretrained(cfg.model.bin_plm)
    # model = AutoModelForSequenceClassification.from_pretrained(cfg.model.saved_model)
    bin_model = AutoModelForSequenceClassification.from_pretrained(cfg.model.binary_model)
    bin_model.parameters
    bin_model.to(device)
    
    per_tokenizer = AutoTokenizer.from_pretrained(cfg.model.sec_plm)
    per_model = AutoModelForSequenceClassification.from_pretrained(cfg.model.per_model)
    per_model.parameters
    per_model.to(device)
    
    org_tokenizer = AutoTokenizer.from_pretrained(cfg.model.sec_plm)
    org_model = AutoModelForSequenceClassification.from_pretrained(cfg.model.org_model)
    org_model.parameters
    org_model.to(device)
    

    ## load test datset
    test_dataset_dir = cfg.data.test_data
    test_id, test_dataset, test_label,test_sub_type = load_test_dataset(test_dataset_dir)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label, bin_tokenizer)
    
    ## predict answer ## 절대 바꾸지 말 것 ##
    pred_answer, output_prob = inference(bin_model, Re_test_dataset, device) # model에서 class 추론
    pred_answer = bin_num_to_label(cfg, pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    output_prob = [change_prob(prob,'bin') for prob in output_prob]
    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,'sub_type' : test_sub_type,
                           'subject_entity' : test_dataset['subject_entity'],'object_entity' :    test_dataset['object_entity'],
                            'sentence' :test_dataset['sentence']})
    no_rel_output, rel_output = output[output['pred_label']=='no_relation'],output[output['pred_label']=='relation']
    print(f'first split by binary cf no_relation : {len(no_rel_output)} relation {len(rel_output)}')
    # check output
    per_id, RE_PER = double_check(rel_output,'PER',per_tokenizer)
    org_id, RE_ORG = double_check(rel_output,'ORG',org_tokenizer)
    print(f'this is error in split error maybe  per : {len(per_id)} org : {len(org_id)}')
    print(f'this is error in split error maybe  real_per : {len(rel_output[rel_output["sub_type"]=="PER"])} real_org : {len(rel_output[rel_output["sub_type"]=="ORG"])}')

    
    per_answer, per_output_prob = inference(per_model, RE_PER, device) # model에서 class 추론
    per_answer = per_num_to_label(cfg, per_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    per_output_prob = [change_prob(prob,'per') for prob in per_output_prob]
    
    org_answer, org_output_prob = inference(org_model, RE_ORG, device) # model에서 class 추론
    org_answer = org_num_to_label(cfg, org_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    org_output_prob = [change_prob(prob,'org') for prob in org_output_prob]
    
    per_output = pd.DataFrame({'id':per_id,'pred_label':per_answer,'probs':per_output_prob})
    org_output = pd.DataFrame({'id':org_id,'pred_label':org_answer,'probs':org_output_prob})
    
    output = pd.concat([no_rel_output[['id','pred_label','probs']],per_output,org_output])
    output = output.sort_values('id',ascending=True)

    # return output
    output.to_csv(cfg.test.output_csv, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

    
    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)               # 시드를 고정해도 함수를 호출할 때 다른 결과가 나오더라..?
    random.seed(seed)
    print('lock_all_seed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-test')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    seed_everything(cfg.train.seed)
    test(cfg)