#!/bin/bash
for i in $(seq 0  12)
do 
bash multiple_train.sh   $i 
done
bash binary_train.sh
