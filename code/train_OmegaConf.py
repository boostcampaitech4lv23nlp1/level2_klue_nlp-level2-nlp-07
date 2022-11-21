import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback

from omegaconf import OmegaConf
import wandb
import argparse
from load_data import *
from utils import *
import random


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    ########### setting model hyperparameter   ###########
    model_config =  AutoConfig.from_pretrained(cfg.model.model_name)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(cfg.model.model_name, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    ########### load dataset   ###########
    train_dataset = load_data(f'{cfg.path.train_path}/train.csv')
    train_label = label_to_num(train_dataset['label'].values)
    
    # Unzip_entity : subj/obj의 word, start_idx, end_idx, type 불러와서 완성된 데이터셋으로 만들기
    train_unzip = unzip_entity(train_dataset)

    # Column Selection, cols 내용을 바꿔서 원하는 columns만 뽑을 수 있다
    cols = ['id','sentence','subject_word','object_word','label']
    train_dataset = column_selection(train_unzip,cols)

    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
    train_x, dev_x, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2, random_state=cfg.train.seed, stratify=train_label)
    train_x.reset_index(drop=True,inplace = True)
    dev_x.reset_index(drop=True,inplace = True)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    # tokenizer에 넣을 sentence1, sentence2 생성, 복수의 cols 선택 가능, join으로 한 문장 만듦
    sentence1_cols = ['subject_word']
    sentence2_cols = ['object_word']
    train_sentence1, train_sentence2 = make_sentence(train_x,sentence1_cols,sentence2_cols)
    dev_sentence1, dev_sentence2 = make_sentence(dev_x,sentence1_cols,sentence2_cols)

    # tokenizing dataset
    # 맨 앞 argument True => 단일 문장 분류 / False => 두 문장 관계 분류
    tokenized_train = tokenized_dataset(True, train_sentence1, train_sentence2, tokenizer)
    tokenized_dev = tokenized_dataset(True, dev_sentence1, dev_sentence2, tokenizer)
    
    # tokenized_train, tokenized_dev, train_label, dev_label = train_test_split(tokenized_train, train_label, test_size=0.2, random_state=cfg.train.seed, stratify=train_label)
    
    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train,train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev,dev_label)

    
    training_args = TrainingArguments(
        output_dir=cfg.path.checkpoint,
        save_total_limit=5,
        save_steps=cfg.train.logging_step,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        per_device_train_batch_size=cfg.train.batch_size,  # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,   # default : 16
        warmup_steps=cfg.train.logging_step,               
        weight_decay=0.01,               
        logging_steps=100,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.logging_step,               # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'micro_f1_score',
        # wandb 설정
        report_to="wandb",
        # run_name=f"{cfg.model.saved_name}_batch_size : {cfg.train.batch_size} EPOCH : {cfg.train.epoch} lr : {cfg.train.lr}"
        run_name= cfg.model.exp_name
        )
    
    trainer = TrainerwithFocalLoss(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics  # define metrics function
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    # train model
    trainer.train()
    model.save_pretrained(f'{cfg.path.best_model}/{cfg.model.saved_name}')


# set fixed random seed
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
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    wandb.login()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    seed_everything(cfg.train.seed)
    wandb.init(project="robertaS_single", entity="klue_level2",name = cfg.model.exp_name)
    main(cfg)
    wandb.finish()