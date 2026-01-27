import gzip
import json
import random
import os

BASE_DIR = "data/a5-alignment/hh"
SUBSETS = [
    "harmless-base", 
    "helpful-base", 
    "helpful-online", 
    "helpful-rejection-sampled"
]

def load_hh_dataset():
    data = []
    print(f"Reading from {BASE_DIR}...")
    
    for subset in SUBSETS:
        path = os.path.join(BASE_DIR, subset, "train.jsonl.gz")
        if not os.path.exists(path):
            print(f"Skipping {subset} (file not found)")
            continue
            
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                chosen = item["chosen"]
                rejected = item["rejected"]
                
                # Filter multi-turn: Human speaks more than once
                if chosen.count("\n\nHuman:") > 1:
                    continue
                
                # Parse Instruction (everything before the first Assistant response)
                try:
                    # Split on the first Assistant marker
                    parts = chosen.split("\n\nAssistant:", 1)
                    instruction = parts[0].replace("\n\nHuman:", "").strip()
                    chosen_response = parts[1].strip()
                    
                    # Parse rejected similarly
                    rejected_response = item["rejected"].split("\n\nAssistant:", 1)[1].strip()
                    
                    data.append({
                        "source": subset,
                        "instruction": instruction,
                        "chosen_response": chosen_response,
                        "rejected_response": rejected_response
                    })
                except IndexError:
                    continue # Skip malformed lines

    return data

def main():
    dataset = load_hh_dataset()
    print(f"Total single-turn examples: {len(dataset)}")

    # Helper to print samples
    def show_samples(category_name, keyword):
        subset_data = [d for d in dataset if keyword in d['source']]
        if not subset_data: return

        print(f"\n{'='*30} 3 Random {category_name} Examples {'='*30}")
        samples = random.sample(subset_data, 3)
        for i, ex in enumerate(samples):
            print(f"\n[Example {i+1}] Source: {ex['source']}")
            print(f"INSTRUCTION:\n{ex['instruction']}")
            print(f"-" * 20)
            print(f"CHOSEN:\n{ex['chosen_response'][:400]}...") 
            print(f"-" * 20)
            print(f"REJECTED:\n{ex['rejected_response'][:400]}...")
            print("*" * 80)

    show_samples("HELPFUL", "helpful")
    show_samples("HARMLESS", "harmless")

if __name__ == "__main__":
    main()