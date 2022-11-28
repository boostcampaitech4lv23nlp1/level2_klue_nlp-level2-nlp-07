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
from utils import *
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    ########### setting model hyperparameter   ###########
    model_config =  AutoConfig.from_pretrained(cfg.model.model_name)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(cfg.model.model_name, config=model_config)
    model.parameters
    model.to(device)

    ########### load dataset   ###########
    train_dataset = load_data(cfg.data.train_data)
    train_label = label_to_num(train_dataset['label'].values, None)
    
    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
    train_data, dev_data, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2, random_state=cfg.train.seed, stratify=train_label)
    train_data.reset_index(drop=True, inplace = True)
    dev_data.reset_index(drop=True, inplace = True)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
    
    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(train_x,train_label, tokenizer)
    RE_dev_dataset = RE_Dataset(dev_x,dev_label, tokenizer)

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
    
    # trainer = Trainer(
    trainer = TrainerWithLossTuning(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics  # define metrics function
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--feature_mode", type=str, default="ner", choices=["text", "ner", "text_ner", "ner_text"])
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    main(args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='config')
#     args, _ = parser.parse_known_args()
#     wandb.login()
#     cfg = OmegaConf.load(f'./config/{args.config}.yaml')
#     seed_everything(cfg.train.seed)
#     # wandb.init(project="klue-roberta-large", entity="klue-bora",name = cfg.model.exp_name)
#     wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)
#     main(cfg)
#     wandb.finish()