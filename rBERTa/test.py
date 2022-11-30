from transformers import ElectraTokenizer,ElectraConfig,ElectraModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback, AutoModel
from torch import nn
import torch
from collections import OrderedDict
import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback

import argparse
from omegaconf import OmegaConf
from load_data import *
from utils import *
from model import *
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class REModel(nn.Module):
    def __init__(
        self,
        model_config = None,
        pretrained_id: str = "klue/roberta-large",
        num_labels: int = 30,
        dropout_rate: float = 0.1,
        device = 'cuda:0'
    ):
        super(REModel, self).__init__()
        self.num_labels = num_labels
        if model_config is None:
            model_config = AutoConfig.from_pretrained(pretrained_id)

        self.hidden_size = model_config.hidden_size
        self.device = device
        self.tanh = nn.Tanh()
        if pretrained_id:
            self.plm = AutoModel.from_pretrained(pretrained_id,add_pooling_layer = False) # 5e-5
            self.dense_for_cls = nn.Sequential(OrderedDict({'cls_Linear' : nn.Linear(self.hidden_size , self.hidden_size)}))
            self.dense_for_e1 =  nn.Sequential(OrderedDict({'e1_Linear' : nn.Linear(self.hidden_size , self.hidden_size)}))
            self.dense_for_e2 = nn.Sequential(OrderedDict({'e2_Linear' : nn.Linear(self.hidden_size , self.hidden_size)}))
            self.entity_classifier = nn.Sequential(OrderedDict({                          # 10e-5
                                    'dense1': nn.Linear(self.hidden_size * 3, self.hidden_size),
                                    'dropout1': nn.Dropout(dropout_rate),
                                    'dense2': nn.Linear(self.hidden_size, self.hidden_size),
                                    # 'dropout2': nn.Dropout(dropout_rate),
                                    'out_proj': nn.Linear(self.hidden_size, num_labels)
                                    }))
        self.num_labels = num_labels

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        entity_idx= inputs['Entity_idxes']

        x = self.plm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
        ).last_hidden_state
        
        
        subject_entity_avgs = []
        oject_entity_avgs = []
        for idx in range(entity_idx.shape[0]):
            subject_entity_avg = x[idx, entity_idx[idx][0]:entity_idx[idx][1], :] # 1,n,hidden -> n,hidden -> 1,hidden

            subject_entity_avg = torch.mean(subject_entity_avg, dim=0) # 1,hidden -> 1,hidden

            subject_entity_avgs.append(subject_entity_avg.cpu().detach().numpy())

            oject_entity_avg = x[idx, entity_idx[idx][2]:entity_idx[idx][3], :] # 1, len, hidden
            oject_entity_avg = torch.mean(oject_entity_avg, dim=0)           #  1, hidden
            oject_entity_avgs.append(oject_entity_avg.cpu().detach().numpy()) # 1, 1, hidden

        subject_entity_avgs = torch.tensor(subject_entity_avgs).to(self.device) # batch,1,hidden
        oject_entity_avgs = torch.tensor(oject_entity_avgs).to(self.device)
        
        x0 = x[:, 0, :]                                         # cls 16, hidden
        x0 = self.tanh(self.dense_for_cls(x0))
        subject_entity_avgs = self.tanh(self.dense_for_e1(subject_entity_avgs))
        oject_entity_avgs = self.tanh(self.dense_for_e2(oject_entity_avgs))

        x = torch.cat((x0 ,subject_entity_avgs, oject_entity_avgs), dim=1)
        x = self.entity_classifier(x)
        return {'output' :x}
    
    
def train(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    model = REModel(pretrained_id = cfg.model.model_name,device = device)
        
    model.parameters
    model.to(device)

    ## load dataset 
    train_dataset = load_data(cfg.data.train_data).sample(frac=0.5)
    train_label = label_to_num(train_dataset['label'].values)

    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
    train_data, dev_data, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2, random_state=cfg.train.seed, stratify=train_label)
    train_data.reset_index(drop=True, inplace = True)
    dev_data.reset_index(drop=True, inplace = True)
    RE_train_dataset = RE_Dataset(train_data, train_label, tokenizer, cfg)
    RE_dev_dataset = RE_Dataset(dev_data, dev_label, tokenizer, cfg)
    model.plm.resize_token_embeddings(len(RE_train_dataset.tokenizer))
    
    insert_entity_idx_tokenized_dataset(tokenizer, RE_train_dataset.dataset, cfg)
    insert_entity_idx_tokenized_dataset(tokenizer, RE_dev_dataset.dataset, cfg)
    
    ## train arguments
    training_args = TrainingArguments(
        output_dir=cfg.train.checkpoint,
        save_total_limit=5,
        save_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        
        per_device_train_batch_size=cfg.train.batch_size,    # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 16
        warmup_steps=cfg.train.warmup_steps,               
        weight_decay=cfg.train.weight_decay,               
        
        # for log
        logging_steps=cfg.train.logging_step,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.warmup_steps,                 # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'loss',
        greater_is_better=False,                             # False : loss 기준으로 최적화 해봄 도르
        dataloader_num_workers=cfg.train.num_workers,
        fp16=True,

        # wandb
        # report_to="wandb",
        # run_name= cfg.wandb.exp_name
        )
    
    trainer = TrainerwithFocalLoss(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics,  # define metrics function
        # tokenizer = RE_train_dataset.tokenizer
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    ## train model
    trainer.train()
    
if __name__=='__main__':
    torch.cuda.empty_cache()
    ## parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    train(cfg)
