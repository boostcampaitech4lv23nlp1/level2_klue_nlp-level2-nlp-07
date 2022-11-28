python klue_binary.py \
--train_data_dir /opt/ml/dataset/train/train_re.csv --test_data_dir /opt/ml/dataset/test/test_data.csv \
--model klue/roberta-small  --max_seq_length 256    --weight_decay 0.01 \
--train_batch_size 32   --eval_batch_size 32   --num_train_epochs 10  \
--learning_rate 5e-5    --seed 42   --logging_step 1000 \
--output_dir /opt/ml/code/prediction/recent/depot-recent-binary.csv \
--wandb_name 30_BaseLine-RECENT-binary \
--checkpoint_dir /opt/ml/code/results/recent/depot-recent-binary \
--save_model_dir /opt/ml/code/save_model/recent/depot-recent-binary   \

''