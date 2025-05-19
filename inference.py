import os
import fire
import json
import torch
from tqdm import tqdm
from Prompt import Prompt
from peft import PeftModel
from accelerate import Accelerator
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from transformers import GenerationConfig
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig


def inference(
    dataset = "" ,
    batch_size = 16,
    resume_from_checkpoint = "save_checkpoint/ml-10m/save_path_cdpo/final_checkpoint",  #
    local_model_path = "llm/Llama-3.1-8B-Instruct",  #
    external_prompt_path = "/prompt/movie.txt" ,
    temperature = 0.3,  #
    top_k = 20 , # top-k
    top_p = 0.9,  # top-p
    max_new_tokens = 32 , #
    output_path = "/eval_result/ml-10m/cdpo_results.json",
    test_data_path = "/dataset/ml-10m/ml-ood_test.json" ,

):

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Local model path {local_model_path} does not exist, please check the path!")

    # Model setup
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    accelerator = Accelerator()
    device_index = accelerator.process_index
    device_map = {"": device_index}

    # Load model
    print(f"Loading model from local path {local_model_path}...")
    model = LlamaForCausalLM.from_pretrained(
        local_model_path,
        device_map=device_map,
        quantization_config=bnb_config,
        local_files_only=True,
    )

    # Load PEFT model
    print(f"Loading PEFT model from {resume_from_checkpoint}...")
    model = PeftModel.from_pretrained(
        model,
        resume_from_checkpoint,
        local_files_only=True
    )
    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer from local path {local_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Data processing functions
    def convert_dict_to_prompt(d: dict):
        t = Prompt('prompt/movie.txt')
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def generate_and_tokenize_prompt(data_point):
        t = convert_dict_to_prompt(data_point)
        prompt = str(t)
        dic = data_point.copy()
        dic["prompt"] = prompt.rstrip()
        return dic

    # Load and prepare dataset
    data_files = {"test": test_data_path}
    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    test_data = data["test"].map(generate_and_tokenize_prompt)

    # Generation function
    def generate_text(batch):
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=10,              # Using 10 beams to generate 10 sequences
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"generated_text": generated_texts}
    
    def extract_answer(output):
        answer_start = output.find("Answer:") + len("Answer:")
        answer_text = output[answer_start:].strip()
        cleaned_answer = ''.join(char for char in answer_text if char not in "!@#$%^&*{}")
        cleaned_answer = " ".join(cleaned_answer.split())
        return cleaned_answer

    def extract_recommendation(sentence: str) -> str:
        if not isinstance(sentence, str):
            return sentence

        if not sentence or "Based on" not in sentence:
            return sentence

        parts = sentence.split("Based on", 1)
        recommendation = parts[0].strip()
        
        return recommendation


    # Generate predictions
    test_data_with_predictions = test_data.map(
        generate_text,
        batched=True,
        batch_size=batch_size
    )

    # Process results
    results = []
    max_retries = 3

    for item in test_data_with_predictions:
        retry_count = 0
        prediction = ""
        
        while retry_count < max_retries and not prediction.strip():
            generated_text = item["generated_text"]
            # Extract prediction from generated text
            prediction = generated_text.split("### Answer:")[1].strip() if "### Answer:" in generated_text else generated_text.strip()
            
            prediction = extract_answer(prediction)
            prediction = extract_recommendation(prediction)
            prediction = prediction.split(".")[0]
            
            try:
                prediction = prediction.split(",")[0]
            except:
                pass
            
            if not prediction.strip():
                retry_count += 1

                input_text = item["prompt"]
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature * (1 + retry_count * 0.2),
                        top_k=top_k,
                        top_p=top_p,
                        num_beams=10,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                item["generated_text"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                break

        result = {
            "historyList": item["historyList"],
            "trueSelection": item["trueSelection"],
            "prediction": prediction
        }
        results.append(result)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Inference completed, results saved to {output_path}")
    return results


if __name__ == "__main__":
    fire.Fire(inference)
