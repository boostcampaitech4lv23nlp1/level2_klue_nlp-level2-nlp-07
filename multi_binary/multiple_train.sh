python klue_multiple.py \
--train_data_dir /opt/ml/dataset/train/train_final.csv --test_data_dir /opt/ml/dataset/test/test_data.csv \
--model klue/roberta-small  --max_seq_length 256    --weight_decay 0.01 \
--train_batch_size 32   --eval_batch_size 32   --num_train_epochs 1  \
--learning_rate 5e-5    --seed 42   --logging_step 1000 \
--output_dir /opt/ml/code/prediction/recent_data/depot-recent-$1.csv \
--wandb_name test_BaseLine-RECENT_Data-$1 \
--checkpoint_dir /opt/ml/code/results/recent_data/depot-recent-$1 \
--save_model_dir /opt/ml/code/save_model/recent_data/depot-recent-$1    \
--type_pair_id $1

