import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback

from omegaconf import OmegaConf
from load_data import *
from utils import *
import random
from collections import Counter
from transformers import DataCollatorWithPadding

def train_single(cfg):
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

    # dev data to csv for gold label save
    dev_data['label'] = dev_label
    dev_data.to_csv(cfg.data.dev_data, index=False)

    ## make dataset for pytorch
    RE_train_dataset = RE_Dataset(train_data, train_label, tokenizer, cfg)
    RE_dev_dataset = RE_Dataset(dev_data, dev_label, tokenizer, cfg)
    model.resize_token_embeddings(len(RE_train_dataset.tokenizer))

    if cfg.train.entity_embedding:
        print('='*10, "Start", '='*10)
        insert_entity_idx_tokenized_dataset(tokenizer, RE_train_dataset.dataset, cfg)
        insert_entity_idx_tokenized_dataset(tokenizer, RE_dev_dataset.dataset, cfg)
        print('='*10, "END", '='*10)

    ## make samples_per_class (which is needed for TrainerwithLosstuning)
    train_label_counter = Counter(train_label)
    samples_per_class = [train_label_counter[i] for i in range(model_config.num_labels)] ## [7765, 1023, 339, ....]

    ## train arguments
    training_args = TrainingArguments(
        output_dir=cfg.train.checkpoint,
        save_total_limit=5,
        save_steps=cfg.train.logging_step,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        per_device_train_batch_size=cfg.train.batch_size,    # default : 32
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 32
        warmup_steps=cfg.train.logging_step,               
        weight_decay=cfg.train.weight_decay,               
        logging_steps=100,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.logging_step,                 # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'micro_f1_score',
        # wandb
        report_to="wandb",
        run_name= cfg.wandb.exp_name,
        group_by_length=cfg.train.group_by_length
        )
    ## setting data_collator
    if cfg.train.padding == "max_length":
        data_collator = DataCollatorWithPadding(tokenizer, padding = "max_length", max_length=cfg.train.max_length)
    elif cfg.train.padding == "longest":
        data_collator = DataCollatorWithPadding(tokenizer, padding = True)
    print(data_collator)
    ## setting custom trainer with default optimizer & scheduler : AdamW, LambdaLR
    trainer = TrainerwithLosstuning(
        samples_per_class=samples_per_class,
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        data_collator = data_collator,   # data collator (dynamic padding or smart batching)
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics,  # define metrics function
        callbacks = [EarlyStoppingEval(early_stopping_patience=cfg.train.patience,\
                                    early_stopping_threshold = cfg.train.threshold)]# total_step / eval_step : max_patience
    )

    ## train model
    trainer.train()
    
    ## save model
    model.save_pretrained(cfg.model.saved_model)