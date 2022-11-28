from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn
import numpy as np
import pickle as pickle
from torch import nn
import torch

from transformers import Trainer

ID_TO_TYPE_PAIR = {
    0: 'ORG_PER', 1: 'ORG_ORG', 2: 'ORG_DAT', 3: 'ORG_LOC', 4: 'ORG_POH', 5: 'ORG_NOH',
    6: 'PER_PER', 7: 'PER_ORG', 8: 'PER_DAT', 9: 'PER_LOC', 10: 'PER_POH', 11: 'PER_NOH'
}

LABEL_TO_ID = {
    0: # ORG_PER
    {
        'org:top_members/employees': 1,
        'org:founded_by': 28,
        'org:alternate_names': 5,
    },
    1: # ORG_ORG
    {
        'org:member_of': 20,
        'org:alternate_names': 5,
        'org:members': 2,
        'org:place_of_headquarters': 7,
        'org:political/religious_affiliation': 19,
        'org:product': 3,
        'org:top_members/employees' : 1,
        'org:founded_by' : 28,
    },
    2: # ORG_DAT
    {
        'org:founded': 18,
        'org:dissolved': 22,
    },
    3: # ORG_LOC
    {
        'org:place_of_headquarters': 7,
        'org:member_of': 20,
        'org:members': 2,
        'org:product': 3,
        'org:alternate_names': 5,
    },
    4: # ORG_POH
    {
        'org:member_of': 20,
        'org:product': 3,
        'org:alternate_names': 5,
        'org:top_members/employees': 1,
        'org:place_of_headquarters': 7,
        'org:political/religious_affiliation': 19,
        'org:members':2,
        'org:founded_by':28,
    },
    5: # ORG_NOH
    {
        'org:number_of_employees/members': 9,
        'org:member_of' :20,
    },
    6: # PER_PER
    {
        'per:alternate_names': 12,
        'per:spouse': 17,
        'per:colleagues': 14,
        'per:parents': 21,
        'per:employee_of': 6,
        'per:children': 10,
        'per:other_family':13,
        'per:siblings':16,
        'per:origin': 15,  
        'per:title':4,
    },
    7: # PER_ORG
    {
        'per:employee_of': 6,
        'per:origin': 15,
        'per:title':4,
        'per:schools_attended':23,
        'per:religion': 29,
        'per:alternate_names': 12, 
    },
    8: # PER_DAT
    {
        'per:date_of_birth': 25,
        'per:date_of_death': 24,
        'per:origin': 15,
        'per:employee_of': 6,
    },
    9: # PER_LOC
    {
        'per:origin': 15,
        'per:place_of_residence': 11,
        'per:employee_of': 6,
        'per:place_of_birth': 26,
        'per:title': 4,
        'per:place_of_death': 27,
        'per:alternate_names': 12,
    },
    10: # PER_POH
    {
        'per:title': 4,
        'per:employee_of': 6,
        'per:product': 8,
        'per:alternate_names':12,
        'per:parents':21,
        'per:origin':15,
        'per:spouse':17,
        'per:siblings':16,
        'per:children':10,
    },
    11: # PER_NOH
    {
        'per:title': 4,
        'per:employee_of': 6,
    },}

def bi_klue_re_micro_f1(preds, labels):
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

def bi_klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def multi_klue_re_micro_f1(preds, labels, type_pair_id):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = list(LABEL_TO_ID[type_pair_id].keys())

    # no_relation class를 제외한 micro F1 score
    # no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def multi_klue_re_auprc(probs, labels, type_pair_id):
    """KLUE-RE AUPRC (with no_relation)"""
    num_labels = len(LABEL_TO_ID[type_pair_id])
    labels = np.eye(num_labels)[labels]

    score = np.zeros((num_labels,))
    for c in range(num_labels):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred,):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = multi_klue_re_micro_f1(preds, labels)
    auprc = multi_klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
      'auprc' : auprc,
      'accuracy': acc,
    }

def label_to_num(label,type_pair_id):
    num_label = []
    if type_pair_id == None:
        with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
    else:
        dict_label_to_num = LABEL_TO_ID[type_pair_id]
    for v in label:
        num_label.append(dict_label_to_num[v])
    print(num_label)
    return num_label

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TrainerwithFocalLoss(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

