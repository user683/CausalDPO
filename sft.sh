#!/bin/bash

torchrun --nproc_per_node 2 sft.py \
        --model_name base_model_path  \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --dataset book \
        --prompt_path prompt_path \
        --logging_dir log_dir \
        --output_dir save_checkpoint/movie/save_path_sft \
        --learning_rate 1e-5 \
        --num_train_epochs 4 \
        --eval_step 0.2 \
        > sft.log
