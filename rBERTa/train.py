import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback

from omegaconf import OmegaConf
from load_data import *
from utils import *
from model import *
import random

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts
# from transformers import DataCollatorWithPadding

def train(cfg):
    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if cfg.model.plm =='BERT':
        model = REModel(pretrained_id = cfg.model.model_name,device = device)
    elif cfg.model.plm == 'electra':
        model = REModel(pretrained_id = cfg.model.model_name,device = device)
        
    model.parameters
    model.to(device)
    
    optimizer = optim.AdamW([
                {'params': model.plm.parameters()},
                {'params': model.dense_for_cls.parameters(), 'lr': cfg.train.second_lr},
                {'params': model.dense_for_e1.parameters(), 'lr': cfg.train.second_lr},
                {'params': model.dense_for_e2.parameters(), 'lr': cfg.train.second_lr},
                {'params': model.entity_classifier.parameters(), 'lr': cfg.train.second_lr}
                    ], lr=cfg.train.lr,eps = 1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.T_0, T_mult=cfg.train.T_mult, eta_min=cfg.train.eta_min)
    optimizers = (optimizer,scheduler)
    
    ## load dataset 
    train_dataset = load_data(cfg.data.train_data)
    train_label = label_to_num(train_dataset['label'].values)

    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
    train_data, dev_data, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2, random_state=cfg.train.seed, stratify=train_label)
    train_data.reset_index(drop=True, inplace = True)
    dev_data.reset_index(drop=True, inplace = True)

    ## make dataset for pytorch
    RE_train_dataset = RE_Dataset(train_data, train_label, tokenizer, cfg)
    RE_dev_dataset = RE_Dataset(dev_data, dev_label, tokenizer, cfg)
    model.plm.resize_token_embeddings(len(RE_train_dataset.tokenizer))

    
    
    ## train arguments
    training_args = TrainingArguments(
        output_dir=cfg.train.checkpoint,
        save_total_limit=5,
        save_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        
        label_smoothing_factor = 0.1,
        
        per_device_train_batch_size=cfg.train.batch_size,    # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 16
        warmup_steps=cfg.train.warmup_steps,               
        weight_decay=cfg.train.weight_decay,               
    
        # for log
        logging_steps=cfg.train.logging_step,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.warmup_steps,                 # evaluation step.
        load_best_model_at_end = True,
        
        metric_for_best_model= 'eval_loss',
        greater_is_better=False,                             # False : loss 기준으로 최적화 해봄 도르
        dataloader_num_workers=cfg.train.num_workers,
        fp16=True,
        group_by_length = True,


        # wandb
        report_to="wandb",
        run_name= cfg.wandb.exp_name
        )
    # data_collator = DataCollatorWithPadding(tokenizer,padding=True)
    trainer = TrainerwithFocalLoss(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        # data_collator = data_collator,
        train_dataset= RE_train_dataset,  # training dataset
        eval_dataset= RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics,  # define metrics function
        optimizers = optimizers
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    ## train model
    trainer.train()
    
    ## save model
    # model.save_model(cfg.model.saved_model)
    torch.save(model,cfg.model.saved_model)