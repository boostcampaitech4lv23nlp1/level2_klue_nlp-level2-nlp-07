from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn
import numpy as np
import pickle as pickle
from torch import nn
import torch
from transformers import Trainer
from balanced_loss import Loss
from omegaconf import OmegaConf
from transformers import EarlyStoppingCallback
from typing import Dict, List, Optional, Union

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    
    # no_relation class를 제외한 micro F1 score
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
      'auprc' : auprc,
      'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

class TrainerwithLosstuning(Trainer):
    def __init__(
        self,
        samples_per_class = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.samples_per_class = samples_per_class

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        cfg = OmegaConf.load(f'./config/config.yaml')
        if cfg.train.loss == "focal_loss":
            loss_fct = Loss(
                loss_type=cfg.train.loss,
                beta=cfg.train.beta,
                fl_gamma=cfg.train.gamma,
                samples_per_class=self.samples_per_class,
                class_balanced=True,
                )
            
        elif cfg.train.loss == "cross_entropy":
            loss_fct = nn.CrossEntropyLoss()
        elif cfg.train.loss == "class_balanced_cross_entropy":
            loss_fct = Loss(
                loss_type=cfg.train.loss,
                samples_per_class=self.samples_per_class,
                class_balanced=True,
            )

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_entity_idxes(tokenizer, token_list, cfg):
    """
        entity 표현 방식에 따른 entity 위치 계산
    """
    entity_embedding = np.zeros(len(token_list))
    if cfg.train.marker_mode == 'EM':
        # 스페셜 토큰 위치로 쉽게 찾을 수 있음 ## [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0] ## ['<subj>', '</subj>', '<obj>', '</obj>']
        vocab_len = len(tokenizer)-4 ## special_token start_idx
        subj_start_idx = np.where(token_list==vocab_len)[0][0]+1
        subj_end_idx = np.where(token_list==vocab_len+1)[0][0]
        obj_start_idx = np.where(token_list==vocab_len+2)[0][0]+1
        obj_end_idx = np.where(token_list==vocab_len+3)[0][0]
        entity_embedding[subj_start_idx:subj_end_idx] = 1
        entity_embedding[obj_start_idx:obj_end_idx] = 2
        
        return entity_embedding, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx
    elif cfg.train.marker_mode == 'EMask':
        # entity word만 1로함. ## [0,0,0,1,1,1,1,0,0,0,2,2,2,0,0] ## ['<subj-ORG>','<subj-PER>','<obj-ORG>','<obj-PER>','<obj-DAT>','<obj-LOC>','<obj-POH>','<obj-NOH>']
        subj_1 = tokenizer.convert_tokens_to_ids(['<subj-ORG>','<subj-PER>'])
        obj_1 = tokenizer.convert_tokens_to_ids(['<obj-ORG>','<obj-PER>','<obj-DAT>','<obj-LOC>','<obj-POH>','<obj-NOH>'])

        ## subj의 start_idx, end_idx를 찾는 과정. tokenized entity word 만 1로 구성할 것임.
        ## '<subj-ORG>'  로 구성되어 있음.그래서 '<subj-ORG>'.idx만 찾아서 1로함
        for idx, t in enumerate(token_list):
            if (t in subj_1):
                entity_embedding[idx] = 1
                subj_start_idx = idx
                subj_end_idx = idx+1
                break

        for idx, t in enumerate(token_list):
            if (t in obj_1):
                entity_embedding[idx] = 2
                obj_start_idx = idx
                obj_end_idx = idx+1
                break

        return entity_embedding, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx
    elif cfg.train.marker_mode == 'TEM': ## check complete
        # entity word만 1로함 ## [0,0,0,1,1,1,1,0,0,0,2,2,2,0,0] ## ['<s:ORG>', '<s:PER>', '<o:ORG>', '<o:PER>', '<o:DAT>', '<o:LOC>', '<o:POH>', '<o:NOH>', '</s:ORG>', '</s:PER>', '</o:ORG>', '</o:PER>', '</o:DAT>', '</o:LOC>', '</o:POH>', '</o:NOH>']
        subj_1 = tokenizer.convert_tokens_to_ids(['<s:ORG>', '<s:PER>'])
        subj_2 = tokenizer.convert_tokens_to_ids(['</s:ORG>', '</s:PER>'])
        obj_1 = tokenizer.convert_tokens_to_ids(['<o:ORG>', '<o:PER>', '<o:DAT>', '<o:LOC>', '<o:POH>', '<o:NOH>'])
        obj_2 = tokenizer.convert_tokens_to_ids(['</o:ORG>', '</o:PER>', '</o:DAT>', '</o:LOC>', '</o:POH>', '</o:NOH>'])

        subj_start_idx = 0
        subj_end_idx = 0
        ## subj의 start_idx, end_idx를 찾는 과정. tokenized entity word 만 1로 구성할 것임.
        ## '<s:ORG>' word '</s:ORG>'  로 구성되어 있음.그래서 '<s:ORG>'.idx + 1 = word의 첫 시작 token
        for idx, t in enumerate(token_list):
            if (t in subj_1):
                subj_start_idx = idx + 1
                subj_end_idx = subj_start_idx + 1
                while token_list[subj_end_idx] not in subj_2:
                    subj_end_idx += 1
                break

        entity_embedding[subj_start_idx:subj_end_idx] = 1

        obj_start_idx = 0
        obj_end_idx = 0
        for idx, t in enumerate(token_list):
            if (t in obj_1):
                obj_start_idx = idx + 1
                obj_end_idx = obj_start_idx + 1
                while token_list[obj_end_idx] not in obj_2:
                    obj_end_idx += 1
                break

        entity_embedding[obj_start_idx:obj_end_idx] = 2
        return entity_embedding, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx
    elif cfg.train.marker_mode == 'TEM_punct':
    # 패턴을 이용해 찾기
        subj_1 = tokenizer.convert_tokens_to_ids('@')
        subj_2 = tokenizer.convert_tokens_to_ids('*')
        obj_1 = tokenizer.convert_tokens_to_ids('#')
        obj_2 = tokenizer.convert_tokens_to_ids('^')
        names = tokenizer.convert_tokens_to_ids(['단체','사람','날짜','장소','기타','수량'])

        subj_start_idx = 0
        subj_end_idx = 0
        ## subj의 start_idx, end_idx를 찾는 과정. tokenized entity word 만 1로 구성할 것임.
        ## @ * type * word @ 로 구성되어 있음.그래서 @.idx + 4 = word의 첫 시작 token -> 이게 아닐 수도 있다. idx + 4 가 꼭 word의 시작점은 아님. type이 여러개의 token으로 tokenize될 수도 있음.
        ## 한국어 PLM에 'ORG','DAT','LOC','POH','NOH'가 vocab에 없다. 물론 그대로 진행할 수도 있지만, TEM_punct의 성능 증가 전제에 맞지 않는다. 차라리 한국어로 번역해서 type을 넣어주는게 좋을 수도 있다.
        for idx, t in enumerate(token_list):
            if t == subj_1 and token_list[idx+1] == subj_2 and (token_list[idx+2] in names):
                subj_start_idx = idx + 4
                subj_end_idx = subj_start_idx + 1
                while token_list[subj_end_idx] != subj_1:
                    subj_end_idx += 1
                break

        entity_embedding[subj_start_idx:subj_end_idx] = 1

        obj_start_idx = 0
        obj_end_idx = 0
        for idx, t in enumerate(token_list):
            if t == obj_1 and token_list[idx+1] == obj_2 and (token_list[idx+2] in names):
                obj_start_idx = idx + 4
                obj_end_idx = obj_start_idx + 1
                while token_list[obj_end_idx] != obj_1:
                    obj_end_idx += 1
                break
        
        entity_embedding[obj_start_idx:obj_end_idx] = 2
        return entity_embedding, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx

    return entity_embedding, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx


def insert_entity_idx_tokenized_dataset(tokenizer, dataset, cfg):
    """
    entity 표현 방식에 따른 entity 위치를 계산한 것 반환 받아 dataset에 넣어줍니다.
    """
    for data in dataset:
        entity_embeddings = []
        entity_idxes = []
        for ids in data['input_ids'].numpy():
            entity_embedding, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx = get_entity_idxes(tokenizer, ids, cfg)
            entity_embeddings.append(entity_embedding)
            entity_idxes.append([subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx])
        data['Entity_type_embedding'] = torch.tensor(entity_embeddings).to(torch.int64)
        data['Entity_idxes'] = torch.tensor(entity_idxes).to(torch.int64)
class EarlyStoppingEval(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # metric_to_check = args.metric_for_best_model
        metric_to_check = "eval_loss"
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
