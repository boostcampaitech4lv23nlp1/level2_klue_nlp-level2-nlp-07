mkdir saved_models
python code/klue_multiple.py \
--train_data_dir /opt/ml/dataset/train/train.csv --test_data_dir /opt/ml/dataset/test/test_data.csv \
--model klue/roberta-small  \
--train_batch_size 128   --eval_batch_size 128   --num_train_epochs 1  \
--learning_rate 5e-5    --seed 42   \
--max_seq_length 256   --output_dir /opt/ml/code/prediction/depot-recent-$1.csv \
--type_pair_id $1   --wandb_name 30_BaseLine-RECENT-$1 \
--checkpoint_dir /opt/ml/code/results/depot-recent-$1 \
--save_model_dir /opt/ml/code/save_model/depot-recent-$1 \ 
--logging_step 1000 --weight_decay 0.01
