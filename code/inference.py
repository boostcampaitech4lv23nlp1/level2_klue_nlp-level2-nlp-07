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
import wandb

def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False) # batch_size= 16
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
            # argmax results
            result = np.argmax(logits, axis=-1)

            # prob threshold 
            # result = []
            # for res in prob:
            #     sorted_res = sorted(res,reverse=True)
            #     first,second = sorted_res[0],sorted_res[1]
            #     if first >= 0.9:
            #         result.append(list(res).index(first)) #res.index(tmp)
            #     else:
            #         # if not pass threshold make label to 0, meaning 'no_relation'
            #         # result.append(0)
            #         # if not pass threshold make label to second-highest label
            #         result.append(list(res).index(second))

            # prob threshold by label
            # result = []
            # for res in prob:
            #     sorted_res = sorted(res,reverse=True)
            #     first,second = sorted_res[0],sorted_res[1]
            #     first_idx = list(res).index(first)
            #     second_idx = list(res).index(second)
            #     #0:'no_relation', 1:'org:top_members/employees', 2:'org:members', 3:'org:product', 4:'per:title', 5:'org:alternate_names', 
            #     #6:'per:employee_of', 7:'org:place_of_headquarters', 8:'per:product', 9:'org:number_of_employees/members', 10:'per:children', 11:'per:place_of_residence', 
            #     #12:'per:alternate_names', 13:'per:other_family', 14:'per:colleagues', 15:'per:origin', 16:'per:siblings', 17:'per:spouse', 
            #     #18:'org:founded', 19:'org:political/religious_affiliation', 20:'org:member_of', 21:'per:parents', 22:'org:dissolved', 23:'per:schools_attended', 
            #     #24:'per:date_of_death', 25:'per:date_of_birth', 26:'per:place_of_birth', 27:'per:place_of_death', 28:'org:founded_by', 29:'per:religion'
            #     thresh_label ={0:[0.9,0.1], 1:[0.9,0.1], 2:[0.9,0.1], 3:[0.9,0.1], 4:[0.9,0.1] ,5:[0.9,0.1],
            #                     6:[0.9,0.1], 7:[0.9,0.1] ,8:[0.9,0.1] ,9:[0.9,0.1] ,10:[0.9,0.1] ,
            #                     11:[0.9,0.1], 12:[0.9,0.1], 13:[0.9,0.1], 14:[0.9,0.1], 15:[0.9,0.1],  
            #                     16:[0.9,0.1], 17:[0.9,0.1], 18:[0.9,0.1], 19:[0.9,0.1], 20:[0.9,0.1],
            #                     21:[0.9,0.1], 22:[0.9,0.1], 23:[0.9,0.1], 24:[0.9,0.1], 25:[0.9,0.1], 26:[0.9,0.1], 
            #                     27:[0.9,0.1], 28:[0.9,0.1], 29:[0.9,0.1]}
            #     if first >= thresh_label[first_idx][0]:
            #         result.append(first_idx) #res.index(tmp)
            #     else:
            #         # if not pass threshold make label to 0, meaning 'no_relation'
            #         # result.append(0)
            #         # if pass second threshold make label to second-highest label
            #         if second >= thresh_label[second_idx][1]:
            #             result.append(second_idx)
            #         else:
            #             result.append(0)

            # prob threshold by first prob - second prob
            # result = []
            # for res in prob:
            #     sorted_res = sorted(res,reverse=True)
            #     first,second = sorted_res[0],sorted_res[1]
            #     if first-second <= 0.3 :#임계점
            #         result = list(res).index(second)
            #     else:
            #         result = list(res).index(first)

            output_pred.append(result)
            output_prob.append(prob)
  
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(cfg, label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open(cfg.test.num_to_label, 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label

def load_test_dataset(dataset_dir):
    """
    test dataset을 불러온 후, tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))

    return test_dataset['id'], test_dataset, test_label

def test(cfg):
    ## Device
    wandb_config = wandb.config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    # if best model
    wandb_params = '/batch-{}'.format(wandb_config.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.saved_model+wandb_params)
    # elif checkpoint
    # model = AutoModelForSequenceClassification.from_pretrained(cfg.test.load_cp)
    model.parameters
    model.to(device)

    ## load test dataset
    test_dataset_dir = cfg.data.test_data
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label, tokenizer, cfg)
    
    ## Load dev dataset
    dev_dataset_dir = cfg.data.dev_data
    dev_id, dev_dataset, dev_label = load_test_dataset(dev_dataset_dir)
    Re_dev_dataset = RE_Dataset(dev_dataset ,dev_label, tokenizer, cfg)

    ## predict answer ## 절대 바꾸지 말 것 ##
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
    pred_answer = num_to_label(cfg, pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## dev predict & gold label
    dev_pred_answer, dev_output_prob = inference(model, Re_dev_dataset, device) # model에서 class 추론
    dev_pred_answer = num_to_label(cfg, dev_pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    gold_answer = num_to_label(cfg, dev_label)

    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(cfg.test.output_csv + wandb_params +'.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    
    ## make dev csv file with predicted answer
    dev_output = pd.DataFrame({'id':dev_id,'gold_label':dev_label,'pred_label':dev_pred_answer,'probs':dev_output_prob,})
    dev_output.to_csv(cfg.test.dev_csv + wandb_params +'.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.