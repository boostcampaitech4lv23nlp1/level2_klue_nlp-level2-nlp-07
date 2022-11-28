import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback
import torch.optim as optim

from omegaconf import OmegaConf
import wandb
import argparse
from load_data import *
from binary_utils import *
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    ########### setting model hyperparameter   ###########
    model_config =  AutoConfig.from_pretrained(cfg.model.model_name)
    model_config.num_labels = 2

    model =  AutoModelForSequenceClassification.from_pretrained(cfg.model.model_name, config=model_config)
    print(model.config)
    optimizer = optim.AdamW(model.parameters(), lr = cfg.train.lr,eps = 1e-8 )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.T_0, T_mult=cfg.train.T_mult, eta_min=cfg.train.eta_min)
    optimizers = (optimizer,scheduler)
    model.parameters
    model.to(device)

    ########### load dataset   ###########
    train_dataset = load_data(cfg.data.train_data)
    train_label = label_to_num(train_dataset['label'].values)
    
    train_x, dev_x, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.1,                                                                         random_state=cfg.train.seed, stratify=train_label)
    train_x.reset_index(drop=True,inplace = True)
    dev_x.reset_index(drop=True,inplace = True)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.


    
    
    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(train_x,train_label, tokenizer)
    RE_dev_dataset = RE_Dataset(dev_x,dev_label, tokenizer)

    
    
    training_args = TrainingArguments(
        output_dir=cfg.train.checkpoint,
        save_total_limit=10,
        save_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        
        # label_smoothing_factor = 0.1,
        
        per_device_train_batch_size=cfg.train.batch_size,    # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 16
        warmup_steps=cfg.train.warmup_steps,               
        # weight_decay=cfg.train.weight_decay,               
        
        # for log
        logging_steps=cfg.train.logging_step,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.warmup_steps,                 # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'eval_loss',
        greater_is_better=False,                             # False : loss 기준으로 최적화 해봄 도르
        dataloader_num_workers=cfg.train.num_workers,
        fp16=True,

        # wandb
        report_to="wandb",
        run_name= cfg.wandb.exp_name
        )
    
    # trainer = Trainer(
    trainer = TrainerWithLossTuning(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics,  # define metrics function
        optimizers = optimizers
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    # train model
    trainer.train()
    model.save_pretrained(cfg.model.saved_model)
    

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
    # wandb.init(project="klue-roberta-large", entity="klue-bora",name = cfg.model.exp_name)
    wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)
    main(cfg)
    wandb.finish()