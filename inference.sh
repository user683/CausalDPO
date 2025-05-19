#!/bin/bash

python inference.py \
    --dataset /dataset \
    --batch_size 16 \
    --resume_from_checkpoint save_checkpoint/ml-10m/save_path_cdpo/final_checkpoint \
    --local_model_path /llm/Llama-3.1-8B-Instruct \
    --external_prompt_path /prompt/yelp.txt \
    > cpo_infer.log



