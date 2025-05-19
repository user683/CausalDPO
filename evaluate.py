import os
import re
import json
import math
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="/eval_result/ml-10m/cdpo_results.json", help="result file")
parser.add_argument("--model", type=str, default="LLM3-8B", help="model name")
parser.add_argument("--exp_csv", type=str, default='/rrr.csv', help="result csv file")
parser.add_argument("--output_dir", type=str, default="/eval_result/ml-10m/cdpo_res.json", help="eval_result_llm3-8b")
parser.add_argument("--dataset", type=str, default="ml-10m", help="category")
args = parser.parse_args()

# Load id2name mapping from txt file
def load_id2name(file_path):
    id2name = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        id_part, name = line.strip().split('::', 1)
        id2name[id_part] = name
    return id2name

# Load name2id mapping from id2name
def create_name2id(id2name):
    return {name: id for id, name in id2name.items()}

# Batch function
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

# Sum of first i keys for ORRatio
def sum_of_first_i_keys(sorted_dic, i):
    keys = list(sorted_dic.values())[:i]
    return sum(keys)

# Setup category and load mappings
id2name_file = f"/dataset/ml-10m/ood_id2name.txt"
id2name = load_id2name(id2name_file)
name2id = create_name2id(id2name)
embeddings = torch.load(f"/dataset/ml-10m/item_embedding.pt")

# Load test data and model
result_json = args.input_dir
with open(result_json, 'r') as f:
    test_data = json.load(f)

total = 0
model = SentenceTransformer('/llm/paraphrase-MiniLM-L3-v2')  # Load online model

# Process predictions and generate embeddings
embeddings = torch.tensor(embeddings).cuda()
text = []
for i, data in tqdm(enumerate(test_data), desc="Extracting predictions"):
    if len(data) > 0:
        if len(data['prediction']) == 0:
            text.append("NAN")
            print("Empty prediction!")
        else:
            match = re.search(r'"([^"]*)"', data['prediction'][0])
            if match:
                name = match.group(1)
                text.append(name)
            else:
                text.append(data['prediction'][0].split('\n', 1)[0])
    else:
        print("Empty:")

predict_embeddings = []
for i, batch_input in tqdm(enumerate(batch(text, 8)), desc="Generating embeddings"):
    predict_embeddings.append(torch.tensor(model.encode(batch_input)).cuda())
predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()

# calculate the similarly
def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.t())

cosine_sim = cosine_similarity(predict_embeddings, embeddings)

# Calculate ranks
batch_size = 1
num_batches = (cosine_sim.size(0) + batch_size - 1) // batch_size
rank_list = []
for i in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, cosine_sim.size(0))
    batch_sim = cosine_sim[start_idx:end_idx]
    batch_rank = batch_sim.argsort(dim=-1, descending=True).argsort(dim=-1)  # Sort by similarity in descending order
    torch.cuda.empty_cache()
    rank_list.append(batch_rank)
rank_list = torch.cat(rank_list, dim=0)

# Metrics calculation (HR, NDCG, ORRatio)
metrics = {"HR@10": 0, "NDCG@10": 0, "ORRatio@10": 0, "HR@20": 0, "NDCG@20": 0, "ORRatio@20": 0}
topk_list = [10, 20]  #  calculate the Top-10 and Top-20

for topk in topk_list:
    S_ndcg = 0
    S_hr = 0
    diversity_dic = {}  # For ORRatio calculation
    for i in tqdm(range(len(test_data)), desc=f"Calculating Metrics for Top-{topk}"):
        rank = rank_list[i]
        target_name = test_data[i]["trueSelection"]
        target_name = target_name.strip().strip('"')
        if target_name in name2id:
            target_id = name2id[target_name]
            total += 1
        else:
            continue

        rankId = rank[int(target_id)]  # Convert target_id to int

        # NDCG & HR
        if rankId < topk:
            S_ndcg += (1 / math.log(rankId + 2))
            S_hr += 1

        # For ORRatio: track frequency of top-k items
        for j in range(topk):
            item_id = torch.argwhere(rank == j).item()
            diversity_dic[item_id] = diversity_dic.get(item_id, 0) + 1

    ndcg_value = S_ndcg / len(test_data) / (1 / math.log(2))
    hr_value = S_hr / len(test_data)
    sorted_dic = dict(sorted(diversity_dic.items(), key=lambda item: item[1], reverse=True))
    orratio_value = sum_of_first_i_keys(sorted_dic, 3) / (topk * total)

    if topk == 10:
        metrics["HR@10"] = hr_value
        metrics["NDCG@10"] = ndcg_value
        metrics["ORRatio@10"] = orratio_value
    elif topk == 20:
        metrics["HR@20"] = hr_value
        metrics["NDCG@20"] = ndcg_value
        metrics["ORRatio@20"] = orratio_value

# Evaluation dictionary
eval_dic = {
    "model": args.input_dir,
    "HR@10": metrics["HR@10"],
    "NDCG@10": metrics["NDCG@10"],
    "ORRatio@10": metrics["ORRatio@10"],
    "HR@20": metrics["HR@20"],
    "NDCG@20": metrics["NDCG@20"],
    "ORRatio@20": metrics["ORRatio@20"]
}

# Save to JSON
file_path = args.output_dir
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []
else:
    data = []

data.append(eval_dic)
with open(args.output_dir, 'w') as file:
    json.dump(data, file, separators=(',', ': '), indent=2)


# Print metrics
print(f"HR@10: {metrics['HR@10']}")
print(f"NDCG@10: {metrics['NDCG@10']}")
print(f"ORRatio@10: {metrics['ORRatio@10']}")
print(f"HR@20: {metrics['HR@20']}")
print(f"NDCG@20: {metrics['NDCG@20']}")
print(f"ORRatio@20: {metrics['ORRatio@20']}")
