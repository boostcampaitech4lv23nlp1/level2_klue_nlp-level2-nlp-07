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
import wandb
import argparse
import pprint
import yaml


def train(cfg):

    ## wandb config 불러오기
    with open('/opt/ml/git_k/level2_klue_nlp-level2-nlp-07/my_baseline/sweep.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    ## wandb initialize 해주기
    run = wandb.init(config=sweep_config)
    ## wandb run name 지정해주기 batch_size 외에도 더 하고 싶다면 이어 붙이세요. - wandb 시각화에 표시되는 이름
    ## cfg의 project name을 받기 때문에 wandb의 yaml로 수정해야함
    wandb.run.name = '{}_{}'.format(cfg.wandb.project_name,wandb.config.batch_size)
    ## 경로 설정용 주소 저장하기
    wandb_params = '/batch-{}'.format(wandb.config.batch_size)

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
        output_dir=cfg.train.checkpoint + wandb_params,
        save_total_limit=5,
        save_steps=cfg.train.logging_step,
        num_train_epochs=wandb.config.epochs,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        per_device_train_batch_size=wandb.config.batch_size,    # default : 32
        per_device_eval_batch_size=wandb.config.batch_size,     # default : 32
        warmup_steps=cfg.train.logging_step,               
        weight_decay=cfg.train.weight_decay,               
        logging_steps=100,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.logging_step,                 # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'micro_f1_score',
        # wandb
        # report_to="wandb",
        # run_name= wandb.run.name
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
    model.save_pretrained(cfg.model.saved_model + wandb_params)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)               # 시드를 고정해도 함수를 호출할 때 다른 결과가 나오더라..?
    random.seed(seed)
    print('lock_all_seed')

## config 불러오고 train 실행

torch.cuda.empty_cache()
    ## parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config')
args, _ = parser.parse_known_args()
cfg = OmegaConf.load(f'./config/{args.config}.yaml')

seed_everything(cfg.train.seed)
train(cfg)

# wandb cli 실행 방법!!!
#1. program key가 들어간 yaml을 사용 ex)sweep.yaml / 파일 이름은 상관 없음.
#2. Sweep 실행
## wandb sweep --project 플젝이름 --entity 엔티티이름 sweep.yaml
## ex) wandb sweep --project sweep-test --entity klue-bora sweep.yaml
#3. 터미널의 Sweep id 확인
#4. 터미널의 Run sweep agent with: ~ 이후 코드 실행

## 할일!!
# config 파일의 train 관련 parameter 들을 sweep.yaml로 옮겨주기
# 