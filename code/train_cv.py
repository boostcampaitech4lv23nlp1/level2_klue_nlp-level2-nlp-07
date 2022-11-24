import os
import torch
from glob import glob
from sklearn.model_selection import train_test_split,StratifiedKFold

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

    ## load dataset 여기선 전체 데이터 셋으로 통일 해둔다.
    dataset = load_data(cfg.data.train_data)
    label = label_to_num(dataset['label'].values)

    # k-fold cross validation
    cv = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)

    for idx , (train_idx , val_idx) in enumerate(cv.split(dataset, label)):
        # prepare tokenized datasets and labels each fold
        train_dataset = dataset.iloc[train_idx]
        val_dataset = dataset.iloc[val_idx]
        
        train_y = label[train_idx]
        val_y = label[val_idx]
        
        # make dataset for pytorch
        RE_train_dataset = RE_Dataset(train_dataset, train_y,tokenizer)
        RE_valid_dataset = RE_Dataset(val_dataset, val_y,tokenizer)
        
        # instantiate pretrained language model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=42)
        model.to(device)
        
        # callbacks
        early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.00005)
        
        # set training arguemnts
        output_dir = '../result' + str(idx)
        training_args = TrainingArguments(
            output_dir=cfg.train.checkpoint,          
            logging_dir='../logs',                         
            logging_steps=100,
            save_total_limit=1,
            evaluation_strategy='steps',
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            dataloader_num_workers=cfg.train.num_workers,
            fp16=True,

            seed=cfg.train.seed,
            num_train_epochs=cfg.train.epoch,
            per_device_train_batch_size=cfg.train.batch_size,
            per_device_eval_batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.lr,
            warmup_steps=cfg.train.warmup_steps,
            weight_decay=cfg.train.weight_decay,
            # wandb
            report_to="wandb",
            run_name= cfg.wandb.exp_name
        )
        
        # traniner
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
            eval_dataset= RE_valid_dataset,             # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,         # define metrics function
            callbacks=[early_stopping]
        )

        # train model
        trainer.train()
        
        # del model
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # del cache
        path = glob(f"/opt/ml/code/result{idx}/*")[0]
        for filename in os.listdir(path):
            if filename not in ['config.json', 'pytorch_model.bin', '.ipynb_checkpoints']:
                rm_filename = os.path.join(path, filename)
                os.remove(rm_filename)

    wandb.finish()