from datasets import load_dataset
import json
import os

OUT = "data/eval.jsonl"
os.makedirs("data", exist_ok=True)

ds = load_dataset("gsm8k", "main", split="test[:200]") 

with open(OUT, "w") as f:
    for ex in ds:
        # We'll evaluate by extracting final numeric answer later.
        record = {
            "id": ex["question"][:60].replace("\n"," "),
            "prompt": f"Solve step by step. Give final answer as: #### <number>\n\nQuestion: {ex['question']}\nAnswer:",
            "gold": ex["answer"]
        }
        f.write(json.dumps(record) + "\n")

print(f"Wrote {OUT} with {len(ds)} examples")