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
from inference import *

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # setting model hyperparameter #
    model_config =  AutoConfig.from_pretrained(args.model)
    model_config.num_labels = len(LABEL_TO_ID[args.type_pair_id])

    model =  AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
    # optimizer = optim.AdamW(model.parameters(), lr = cfg.train.lr,eps = 1e-8 )
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.T_0, T_mult=cfg.train.T_mult, eta_min=cfg.train.eta_min)
    # optimizers = (optimizer,scheduler)
    model.parameters
    model.to(device)

    ########### load dataset   ###########
    train_dataset = load_data(args.train_data_dir)
    subj_type, obj_type = ID_TO_TYPE_PAIR[args.type_pair_id].split('_')
    subj_data = train_dataset[train_dataset['subject_entity'].apply(lambda x: eval(x)['type']==subj_type)]
    obj_data = train_dataset[train_dataset['object_entity'].apply(lambda x: eval(x)['type']==obj_type)]
    merge_dataset = pd.merge(subj_data, obj_data, how='inner')
    train_dataset = merge_dataset[merge_dataset['label']!='no_relation']
    
    train_label = label_to_num(train_dataset['label'].values,args.type_pair_id)
    
    train_data, dev_data, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2, random_state=args.seed, stratify=train_label)
    train_data.reset_index(drop=True, inplace = True)
    dev_data.reset_index(drop=True, inplace = True)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(train_data, train_label, tokenizer)
    RE_dev_dataset = RE_Dataset(dev_data, dev_label, tokenizer)

    ## train arguments
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        save_total_limit=5,
        save_steps=args.logging_step,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,                         # default : 5e-5
        per_device_train_batch_size= args.train_batch_size,    # default : 16
        per_device_eval_batch_size= args.eval_batch_size,     # default : 16
        warmup_steps= args.logging_step,               
        weight_decay= args.weight_decay,               
        logging_steps=100,               
        evaluation_strategy='steps',     
        eval_steps = args.logging_step,                 # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model= 'micro_f1_score',
        # wandb
        report_to="wandb",
        run_name= args.wandb_name
        )
    
    # trainer = Trainer(
    trainer = TrainerwithFocalLoss(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,     # evaluation dataset use dev
        compute_metrics=multi_compute_metrics  # define metrics function
        # optimizers = optimizers
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    # train model
    trainer.train()
    model.save_pretrained(args.save_model_dir)

    ## load test datset
    test_dataset_dir = args.test_data_dir
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir,args.type_pair_id)

    Re_test_dataset = RE_Dataset(test_dataset ,test_label, tokenizer)

    ## predict answer ## 절대 바꾸지 말 것 ##
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
    pred_answer = num_to_label(pred_answer,args.type_pair_id) # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(args.output_dir, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

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
    parser.add_argument("--train_data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length")
    
    parser.add_argument('--type_pair_id', default=None, type=int, help='Type Pair Id, e.g. 0 for ORGANIZATION_PERSON.')

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--wandb_name', type=str, help="you should write your own exp name")
    parser.add_argument('--checkpoint_dir', type=str, help="checkpoint save directory")
    parser.add_argument('--save_model_dir', type=str, help="model save directory")
    parser.add_argument('--logging_step', type=int, default=1000, help="save-warmup-eval steps")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    
    args = parser.parse_args()
    wandb.login()
    seed_everything(args.seed)
    wandb.init(project=args.model.replace('/','-'), entity='klue-bora', 
            name=args.wandb_name)
    main(args)
    wandb.finish()
