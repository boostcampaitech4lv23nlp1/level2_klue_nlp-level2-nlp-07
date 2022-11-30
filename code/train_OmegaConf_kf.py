import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts

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
    ## yaml 파일 경로 설정
    with open('/opt/ml/git_k/level2_klue_nlp-level2-nlp-07/code/sweep.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    ## wandb initialize 해주기
    cv_run = wandb.init(config=sweep_config)
    # run = wandb.init(config=sweep_config)
    ## wandb run name 지정해주기 batch_size 외에도 더 하고 싶다면 이어 붙이세요. - wandb 시각화에 표시되는 이름
    ## 경로 설정용 주소 저장하기
    run_name = wandb.run.name
    # wandb_params = '/lr_type-{}_lr-{}'.format(wandb.config.lr_type, wandb.config.lr)
    
    ## load dataset 
    dataset = load_data(cfg.data.train_data)
    label = label_to_num(dataset['label'].values)

    cv = StratifiedKFold(n_splits=3, random_state=cfg.train.seed, shuffle=True)

    for idx , (train_idx , val_idx) in enumerate(cv.split(dataset,label)):
    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
        # print(idx)
        # run_name = '{}-{}'.format(run_name,idx)
        k_run = wandb.init(config=cv_run.config)
    
        wandb.run.name = '{}_{}-{}-{}'.format(wandb.config.name, wandb.config.lr_type, wandb.config.lr,idx)

        wandb_params = '/lr_type-{}_lr-{}-{}'.format(k_run.config.lr_type, k_run.config.lr,idx)

        train_data = dataset.iloc[train_idx]
        dev_data = dataset.iloc[val_idx]
        
        train_label = label_to_num(dataset.iloc[train_idx]['label'].values)
        dev_label = label_to_num(dataset.iloc[val_idx]['label'].values)

        train_data.reset_index(drop=True, inplace = True)
        dev_data.reset_index(drop=True, inplace = True)
        
        # # dev data to csv for gold label save
        # dev_data['label'] = dev_label
        # dev_data.to_csv(cfg.data.dev_data, index=False)
        
        ## Device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
        ## Model & Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(k_run.config.model_name)
        model_config = AutoConfig.from_pretrained(k_run.config.model_name)
        model_config.num_labels = 30 

        model = AutoModelForSequenceClassification.from_pretrained(k_run.config.model_name, config=model_config)
        
        if k_run.config.lr_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = k_run.config.lr, momentum=0.9)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = 2.24e-06, max_lr=2.24e-03, step_size_up=2000, step_size_down=2000, mode='triangular')
        elif k_run.config.lr_type == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr = k_run.config.lr, eps = 1e-8)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
        
        optimizers = (optimizer,scheduler)

        model.parameters
        model.to(device)

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
            output_dir=cfg.train.checkpoint + wandb_params,
            save_total_limit=3,
            save_steps=cfg.train.logging_step,
            num_train_epochs=k_run.config.epochs,
            learning_rate= k_run.config.lr,                         # default : 5e-5
            per_device_train_batch_size=k_run.config.batch_size,    # default : 32
            per_device_eval_batch_size=k_run.config.batch_size,     # default : 32
            warmup_steps=cfg.train.logging_step,               
            weight_decay=k_run.config.weight_decay,               
            logging_steps=100,               
            evaluation_strategy='steps',     
            eval_steps = cfg.train.logging_step,                 # evaluation step.
            load_best_model_at_end = True,
            metric_for_best_model= 'micro_f1_score',
            fp16=True,
            # wandb
            report_to="wandb",
            run_name= run_name
            )
    ## setting custom trainer with default optimizer & scheduler : AdamW, LambdaLR
        trainer = TrainerwithLosstuning(
            samples_per_class=samples_per_class,
            model=model,                     # the instantiated 🤗 Transformers model to be trained
            args=training_args,              # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
            compute_metrics=compute_metrics, # define metrics function
            optimizers = optimizers  
            # callbacks = [EarlyStoppingEval(early_stopping_patience=cfg.train.patience,\
            #                             early_stopping_threshold = cfg.train.threshold)]# total_step / eval_step : max_patience
        )
        ## train model
        trainer.train()
        ## save model
        model.save_pretrained('/opt/ml/code/save_model/' + k_run.config.model_name + wandb_params)
        # wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)               # 시드를 고정해도 함수를 호출할 때 다른 결과가 나오더라..?
    random.seed(seed)
    print('lock_all_seed')

# torch.cuda.empty_cache()
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