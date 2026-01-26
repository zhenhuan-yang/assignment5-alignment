import os
import glob
import json
import time
import argparse
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
from cs336_alignment.parsing import parse_mmlu_response

# ----------------- Configuration -----------------
# Default paths based on your setup
DEFAULT_MODEL_PATH = "data/a5-alignment/models/Llama-3.1-8B"
DEFAULT_DATA_DIR = "data/mmlu/test"
OUTPUT_DIR = "eval_results/mmlu_baseline"

# ----------------- Prompts -----------------
SYSTEM_PROMPT = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```
# Answer:
"""

MMLU_PROMPT_TEMPLATE = """Answer the following multiple choice question about {subject}. Respond with a single
sentence of the form "The correct answer is _", filling the blank with the letter
corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer:
"""

def get_subject_from_filename(filename):
    """
    Extracts subject from filename. 
    e.g., 'high_school_biology_test.csv' -> 'high school biology'
    """
    base = os.path.basename(filename)
    if base.endswith("_test.csv"):
        subject = base.replace("_test.csv", "")
    else:
        subject = base.replace(".csv", "")
    return subject.replace("_", " ")

def format_prompt(subject: str, question: str, options: List[str]) -> str:
    user_instruction = MMLU_PROMPT_TEMPLATE.format(
        subject=subject,
        question=question,
        options=options
    )
    return SYSTEM_PROMPT.format(instruction=user_instruction)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_file", type=str, default="mmlu_generations.jsonl")
    args = parser.parse_args()

    # 1. Load Data
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {args.data_dir}")

    all_examples = []
    
    for file_path in csv_files:
        subject = get_subject_from_filename(file_path)
        # Load CSV without header. 
        # Columns: 0=Question, 1=A, 2=B, 3=C, 4=D, 5=Answer
        df = pd.read_csv(file_path, header=None)
        
        for _, row in df.iterrows():
            example = {
                "subject": subject,
                "question": row[0],
                "options": [row[1], row[2], row[3], row[4]],
                "answer": row[5],
                "source_file": os.path.basename(file_path)
            }
            all_examples.append(example)

    print(f"Loaded {len(all_examples)} total examples.")

    # 2. Prepare Prompts
    prompts = [
        format_prompt(ex["subject"], ex["question"], ex["options"]) 
        for ex in all_examples
    ]

    # 3. Load Model
    print(f"Loading model from {args.model_path}...")
    llm = LLM(model=args.model_path, tensor_parallel_size=1)
    
    # 4. Generate (Greedy decoding as specified)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)
    
    print("Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    total_duration = end_time - start_time
    throughput = len(all_examples) / total_duration

    # 5. Evaluate and Save
    results = []
    correct_count = 0
    parse_fail_count = 0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.output_file)

    for example, output in zip(all_examples, outputs):
        generated_text = output.outputs[0].text
        predicted_label = parse_mmlu_response(example, generated_text)
        
        is_correct = False
        if predicted_label is None:
            parse_fail_count += 1
        elif predicted_label == example["answer"]:
            is_correct = True
            correct_count += 1
            
        results.append({
            **example,
            "generated_text": generated_text,
            "predicted_label": predicted_label,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(all_examples) if all_examples else 0
    parse_fail_rate = parse_fail_count / len(all_examples) if all_examples else 0

    print("\n" + "="*30)
    print(f"MMLU Evaluation Results")
    print(f"="*30)
    print(f"Total Examples:  {len(all_examples)}")
    print(f"Accuracy:        {accuracy:.2%}")
    print(f"Parse Failures:  {parse_fail_count} ({parse_fail_rate:.2%})")
    print(f"Time Taken:      {total_duration:.2f}s")
    print(f"Throughput:      {throughput:.2f} examples/s")
    print("="*30)

    with open(out_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Detailed results saved to {out_path}")

if __name__ == "__main__":
    main()