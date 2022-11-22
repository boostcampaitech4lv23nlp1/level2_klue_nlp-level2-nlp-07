import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback

from omegaconf import OmegaConf
from load_data import *
from utils import *
import random


def train(cfg):
    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model_config = AutoConfig.from_pretrained(cfg.model.model_name)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.model_name, config=model_config)
    model.parameters
    model.to(device)

    ## load dataset 
    train_dataset = load_data(cfg.data.train_data)
    train_label = label_to_num(train_dataset['label'].values)

    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
    train_data, dev_data, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2, random_state=cfg.train.seed, stratify=train_label)
    train_data.reset_index(drop=True, inplace = True)
    dev_data.reset_index(drop=True, inplace = True)

    ## make dataset for pytorch
    RE_train_dataset = RE_Dataset(train_data, train_label, tokenizer)
    RE_dev_dataset = RE_Dataset(dev_data, dev_label, tokenizer)

    ## train arguments
    training_args = TrainingArguments(
        output_dir=cfg.train.checkpoint,
        save_total_limit=5,
        save_steps=cfg.train.logging_step,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        per_device_train_batch_size=cfg.train.batch_size,    # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 16
        warmup_steps=cfg.train.logging_step,               
        weight_decay=cfg.train.weight_decay,               
        logging_steps=100,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.logging_step,                 # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'micro_f1_score',
        # wandb
        report_to="wandb",
        run_name= cfg.wandb.exp_name
        )
    
    trainer = TrainerwithFocalLoss(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics  # define metrics function
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    ## train model
    trainer.train()
    
    ## save model
    model.save_pretrained(cfg.model.saved_model)