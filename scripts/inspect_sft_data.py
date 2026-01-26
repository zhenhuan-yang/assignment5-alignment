import gzip
import json
import random

DATA_PATH = "data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"

def main():
    print(f"Reading from {DATA_PATH}...")
    examples = []
    with gzip.open(DATA_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Total examples: {len(examples)}")
    
    # Sample 10 random examples
    samples = random.sample(examples, 10)
    
    print("\n" + "="*80)
    for i, ex in enumerate(samples):
        print(f"Example {i+1}:")
        print(f"PROMPT: {ex.get('prompt', '')[:300]}...") # Truncate for readability
        print(f"RESPONSE: {ex.get('response', '')[:300]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()