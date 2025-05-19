#!/bin/bash

torchrun --nproc_per_node 1 --master_port=25642 causal_dpo.py \
            --model_name base_model_path  \
            --resume_from_checkpoint save_checkpoint/ml-10m/save_path_sft/final_checkpoint \
            --batch_size 16 \
            --gradient_accumulation_steps 4  \
            --dataset book \
            --prompt_path prompt/book.txt \
            --learning_rate 2e-5 \
            --eval_step 0.4 \
            --beta 1 \
            --neg_num 1 \
            --num_train_epochs 2 \
            --logging_dir log_dir \
            --output_dir save_checkpoint/ml-10m/save_path_cdpo_v1 \
            > causal_dpo.log