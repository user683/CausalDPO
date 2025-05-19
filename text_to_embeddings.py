import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Load base model
base_model = SentenceTransformer('/llm/paraphrase-MiniLM-L3-v2')

# Read item names from file with id::name format
def load_item_names(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Process each line: split by :: and take the name (second part)
    return [line.strip().split('::')[1] for line in lines]

# Batch processing function
def batch_iter(items, batch_size=16):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# Main processing
def generate_embeddings(text_list, model, output_file='/dataset/ml-10m/item_embedding.pt'):
    item_embeddings = []
    
    # Process in batches with progress bar
    for batch_input in tqdm(batch_iter(text_list), total=(len(text_list) + 15) // 16):
        # Generate embeddings directly using SentenceTransformer
        batch_embedding = model.encode(
            batch_input,
            convert_to_tensor=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            show_progress_bar=False
        ).detach().cpu()
        item_embeddings.append(batch_embedding)
    
    # Concatenate all embeddings and save
    item_embedding = torch.cat(item_embeddings, dim=0)
    torch.save(item_embedding, output_file)
    return item_embedding

if __name__ == "__main__":
    # Load items and generate embeddings
    item_name_file = "/dataset/ml-10m/ood_id2name.txt"
    item_names = load_item_names(item_name_file)
    embeddings = generate_embeddings(item_names, base_model)
    print(f"Generated embeddings shape: {embeddings.shape}")