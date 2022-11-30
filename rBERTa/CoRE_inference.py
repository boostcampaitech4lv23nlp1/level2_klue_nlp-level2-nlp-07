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
from model import *
from transformers import DataCollatorWithPadding


def inference(model, tokenized_sent, device,tokenizer):
    """
    test dataset을 DataLoader로 만들어 준 후, batch_size로 나눠 model이 예측 합니다.
    """

    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False,collate_fn = DataCollatorWithPadding(tokenizer)) # batch_size= 16
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        data = {k:v.to(device) for k,v in data.items()}
        with torch.no_grad():
            outputs = model(data) # default : input, token  다 넣어줬음 원래
            logits = outputs['output']
            print(logits)
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)
            print('this is prob', prob)
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(cfg.model.saved_model)
    # model = REModel()
    # model.load_state_dict(torch.load(cfg.model.saved_model))
    model = torch.load(cfg.model.saved_model)
    print(model)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = cfg.data.test_data
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir)
    # Re_test_dataset = RE_Dataset(test_dataset ,test_label, tokenizer,cfg)
    print('#'*10,'entity_bias')
    CoRE_mask1_dataset = CoRE_Dataset(test_dataset ,test_label, tokenizer,cfg,mode ='mask1')
    print(CoRE_mask1_dataset[0])
    print(CoRE_mask1_dataset[1])
    print(CoRE_mask1_dataset[2])
    # print('#'*10,'label_bias')
    # print(model(CoRE_mask1_dataset[0]))
    # CoRE_mask2_dataset = CoRE_Dataset(test_dataset ,test_label, tokenizer,cfg,mode ='mask2')
    
    ## predict answer ## 절대 바꾸지 말 것 ##
    # pred_answer, output_prob = inference(model, Re_test_dataset, device,tokenizer) # model에서 class 추론
    # pred_answer = num_to_label(cfg, pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

    mask_pred_answer, mask_output_prob = inference(model, CoRE_mask1_dataset, device,tokenizer) # model에서 class 추론
    print('#'*10,'entity_bias',mask_output_prob[:2])
    print()
    # mask2_pred_answer, mask2_output_prob = inference(model, CoRE_mask2_dataset, device,tokenizer) # model에서 class 추론
    # print('#'*10,'label_bias',mask2_output_prob[:2])
    
    
    
    # challenge_set
    # print('output_prob shape : ',len(output_prob),len(output_prob[0]))
    # print('output_prob shape : ',len(mask_output_prob),len(mask_output_prob[0]))
    # print('output_prob shape : ',len(mask2_output_prob),len(mask2_output_prob[0]))
    # new prob
    
    lamb_1 = -1.6
    lamb_2 = 0.1
    def la(a,b,c):
        return a+lamb_1*b+lamb_2*c +10#음수발생!

    new_prob = []
    # for i in tqdm(range(5)):
    #     tmp = list(map(la,output_prob[i],mask_output_prob[i],mask2_output_prob[i]))
    #     tmp = np.array(tmp)/sum(tmp)
    #     print('tmp is is is is',tmp)
    #     new_prob.append(tmp.tolist())
    #     print(new_prob)
    
    # print('output_prob shape : ',len(new_prob),len(new_prob[0]))

    # new_preds = np.array(new_prob).argmax(1)
    # new_preds = num_to_label(cfg, new_preds)
    # print(new_preds)
    # label_constraint : subject _ner 에서 나올 수 있는 라벨 constraint
    ## make csv file with predicted answer
    # output = pd.DataFrame({'id':test_id,'pred_label':new_preds,'probs':new_prob})
    # output.to_csv(cfg.test.output_csv, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.