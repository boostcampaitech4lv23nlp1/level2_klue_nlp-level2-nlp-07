from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn
import numpy as np
import pickle as pickle
from torch import nn
import torch
import argparse
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from transformers import Trainer

ID_TO_TYPE_PAIR = {
    0: 'ORG_PER', 1: 'ORG_ORG_POH', 2: 'ORG_DAT', 3: 'ORG_LOC', 4: 'ORG_NOH',
    5: 'PER_PER_POH_ORG', 6: 'PER_DAT_NOH', 7: 'PER_LOC'
}

LABEL_TO_ID = {
    0: # ORG_PER
    {
        'org:top_members/employees': 0,
        'org:founded_by': 1,
        'org:alternate_names': 2,       
    },
    1: # ORG_ORG_POH
    {
        'org:member_of': 0,
        'org:alternate_names': 1,
        'org:members': 2,
        'org:place_of_headquarters': 3,
        'org:political/religious_affiliation': 4,
        'org:product': 5,
        'org:top_members/employees' : 6,
        'org:founded_by':7,
    },
    2: # ORG_DAT
    {
        'org:founded': 0,
        'org:dissolved': 1,
        'org:member_of':2,
    },
    3: # ORG_LOC
    {
        'org:place_of_headquarters': 0,
        'org:member_of': 1,
        'org:members': 2,
        'org:product': 3,
        'org:alternate_names': 4,
        'org:top_members/employees': 5,
        'org:political/religious_affiliation': 6,
    },
    4: # ORG_NOH
    {
        'org:number_of_employees/members': 0,
        'org:member_of' : 1,
        'org:alternate_names':2,
    },
    5: # PER_PER_POH_ORG
    {
        'per:alternate_names': 0,
        'per:spouse': 1,
        'per:colleagues': 2,
        'per:parents': 3,
        'per:employee_of': 4,
        'per:children': 5,
        'per:other_family': 6,
        'per:siblings': 7, 
        'per:origin': 8,  
        'per:title': 9,
        'per:product':10,
        'per:religion':11,
        'per:schools_attended':12,
    },
    6: # PER_DAT_NOH
    {
        'per:date_of_birth': 0,
        'per:date_of_death': 1,
        'per:origin':2,
        'per:employee_of': 3,
        'per:title': 4,
        'per:children': 5, 
    },
    7: # PER_LOC
    {
        'per:origin': 0,
        'per:place_of_residence': 1,
        'per:employee_of': 2,
        'per:place_of_birth': 3,
        'per:title': 4,
        'per:place_of_death': 5,
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

def bi_compute_metrics(pred,):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    # calculate accuracy using sklearn's function
    f1 = bi_klue_re_micro_f1(preds, labels)
    auprc = bi_klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
      'auprc' : auprc,
      'accuracy': acc,
    }

def multi_compute_metrics(pred, ):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = multi_klue_re_micro_f1(preds, labels, args.type_pair_id)
    auprc = multi_klue_re_auprc(probs, labels, args.type_pair_id)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
      'auprc' : auprc,
      'accuracy': acc,
    }

def label_to_num(label, type_pair_id):
    num_label = []
    if type_pair_id == None:
        with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
    else:
        dict_label_to_num = LABEL_TO_ID[type_pair_id]
    for v in label:
        num_label.append(dict_label_to_num[v])
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

class binary_TrainerwithFocalLoss(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        #print(labels)
        # forward pass
        outputs = model(**inputs)
        #print(outputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss()
        #print(logits.shape)
        labels_1 = labels.eq(0).eq(0).float().half()
        lab = torch.sigmoid(logits)
        #print(lab)
        #print(lab.shape)
        print('-----------')
        #print(logits.shape, labels_1.shape)
        print(logits[0])
        print(logits[0].shape)
        #print(labels_1.shape, logits.shape, logits.shape, labels_1.shape, logits.view(-1), logits.view(-1).shape, labels_1.view(-1).shape)
        #layer_1 = nn.Linear(30, 1)
        #print(logits.view(-1), layer_1(logits).view(-1), labels_1.view(-1))
        loss = loss_fct(logits.view(-1), labels_1.view(-1))
        
        #probs = torch.sigmoid(logits).squeeze(-1)
        return (loss, outputs) if return_outputs else loss