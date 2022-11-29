import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from inference import *
from omegaconf import OmegaConf
import wandb
import argparse
from load_data import *
from utils import *
from train_OmegaConf import *
import random
import pprint
import wandb
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

def wandb_function(cfg):
    print('------------------- train start -------------------------')
    train(cfg)
    print('--------------------- test start ----------------------')
    test(cfg)
    print('----------------- Finish! ---------------------')

    ## Reset the Memory
torch.cuda.empty_cache()

## parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config')
args, _ = parser.parse_known_args()
cfg = OmegaConf.load(f'./config/{args.config}.yaml')
# swp = OmegaConf.load(f'./config/sweep.yaml')
## set seed
seed_everything(cfg.train.seed)

## sweep configuration
sweep_config = {
    'name' : 'klue-sweep',
    'method' : 'grid',
    'parameters' :{
    'batch_size': {
        'values':[32,64]
    },
}}

## wandb login
wandb.login()
sweep_id = wandb.sweep(
    sweep=  sweep_config,
    project = cfg.wandb.project_name,
    entity = cfg.wandb.entity
)
pprint.pprint(sweep_config)
    # wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)

# wandb.agent(sweep_id, function=wandb_function(cfg))

wandb.agent(sweep_id, function=train(cfg))
wandb.agent(sweep_id, function=test(cfg))
    ## wandb finish
    # wandb.finish()
