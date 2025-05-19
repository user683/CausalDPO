import os
import fire
import torch
import random
from Prompt import Prompt
from trl import DPOConfig
from accelerate import Accelerator
from datasets import load_dataset
from trainer.causal_dpo_trainer import DPOTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig

random.seed(2025)
def train(
    output_dir="save_checkpoint/ml-1m/save_path_cdpo ",
    local_model_path="llm/Llama-3.1-8B-Instruct",
    gradient_accumulation_steps: int = 4,
    prompt = '',
    resume_from_checkpoint: str = "save_checkpoint/ml-1m/save_path_sft/final_checkpoint",  # the checkpoint save path
    beta: float = 0.1,
    neg_num: int = 1,
    batch_size: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 3e-5,
    cutoff_len: int = 4096,
    eval_step=0.5,
):

    # dataset file path
    # data_files = {
    #     "train": "dataset/yelp2018/yelp-train.json",
    #     "validation": "dataset/yelp2018/yelp-val.json",
    # }
    data_files = {
        "train": "dataset/ml-1m/ml-train.json",  # training dataset
        "validation": "dataset/ml-1m/ml-val.json",  # validation dataset
    }
    # data_files = {
    #     "train": "dataset/exposure/exposure-train.json",
    #     "validation": "dataset/exposure/exposure-val.json",
    # }
    # data_files = {
    #     "train": "dataset/book/book-train.json",
    #     "validation": "dataset/book/book-val.json",
    # }

    def convert_dict_to_prompt(d: dict):
        t = Prompt(prompt)  #
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def process_data(examples):
        dic = {"prompt": [], "chosen": [], "rejected": []}
        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {
                "trueSelection": examples["trueSelection"][i],
                "itemList": examples["itemList"][i],
                "historyList": examples["historyList"][i],
            }
            t = convert_dict_to_prompt(data_point)
            prompt = str(t)
            chosen = data_point["trueSelection"]
            negative_items = [item for item in data_point["itemList"] if item != chosen]
            sample_negs = random.sample(negative_items, min(neg_num, len(negative_items)))  # 防止负样本不足
            for rejected in sample_negs:
                dic['prompt'].append(prompt)
                dic['chosen'].append(chosen)
                dic['rejected'].append(rejected)
        return dic

    # loading dataset
    data = load_dataset("json", data_files=data_files)
    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, batched=True, load_from_cache_file=True).shuffle(seed=42)
    print(train_data)
    val_data = data["validation"].map(process_data, remove_columns=columns, batched=True, load_from_cache_file=True).shuffle(seed=42)
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    #
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map=device_map,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    # check and load checkpoint
    if os.path.isdir(resume_from_checkpoint):
        print(f"Loading PEFT adapter from {resume_from_checkpoint}")
        base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, is_trainable=True)
        print(f"data load finished！")
    else:
        print(f"Checkpoint path {resume_from_checkpoint} not found, initializing new LoRA config.")
        peft_config = LoraConfig(
            inference_mode=False,
            r=64,
            lora_alpha=128,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, peft_config)

    base_model.print_trainable_parameters()

    # load reference model
    model_ref = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map=device_map,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    if os.path.isdir(resume_from_checkpoint):
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    else:
        reference_model = model_ref


    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


    # define the DPOConfig
    dpo_config = DPOConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        ddp_find_unused_parameters=False,
        beta=beta,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )

    # initialize DPOTrainer
    dpo_trainer = DPOTrainer(
        model=base_model,
        ref_model=reference_model,
        args=dpo_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    #  training and save
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    fire.Fire(train)
