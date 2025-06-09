#!/bin/bash

export CUDA_VISIBLE_DEVICES="1,2,3"

# 定义基础命令部分
# 指定一个新的端口，例如 29501
base_command="accelerate launch --num_processes=3 --main_process_port 29501 /home/qiuwenqi/LLM/Fedfinetune/FCL/PILoRA-revise/PILoRA-cifar/federated_main.py --mode federated --is_peft=1 --model=bart_classification_bart-base --baseline bart_classification_bart-base_MABFedCL --dataset=fewrel --iid=0 --encoders_lr=1e-4 --epochs=3 --task_num 7 --fg_nc 10 --total_classes 80 --local_ep 5 --num_users 20 --client_local 5 --niid_type D --beta 1 --local_bs=8 --r=32 --topk_ratio=1 --lambda1=0.5 --mab_rounds=10"

# 假设任务数为 2（可根据实际情况修改）
num_tasks=8

for ((task=0; task<$num_tasks; task++)); do
    # 构建完整命令，添加 --task 参数
    full_command="$base_command --task=$task"
    echo "Running command: $full_command"
    # 执行命令
    eval $full_command
done