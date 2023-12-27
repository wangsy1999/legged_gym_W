#!/bin/bash

# 设置要运行的次数
n=10

# 设置任务名称
task_name="BITeno"

for (( i=1; i<=n; i++ ))
do
    ./legged_gym/scripts/train.py --task=$task_name --train_batch=$i
done