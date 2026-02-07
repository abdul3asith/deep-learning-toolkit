from datasets import load_dataset
import json
import os

OUT = "data/sft.jsonl"
os.makedirs("./data", exist_ok=True)

ds = load_dataset("gsm8k", "main", split="train[:2000]")  # small & fast

with open(OUT, "w") as f:
    for ex in ds:
        record = {
            "prompt": f"Solve step by step. Give final answer as: #### <number>\n\nQuestion: {ex['question']}\nAnswer:",
            "response": ex["answer"]  # includes rationale and final ####
        }
        f.write(json.dumps(record) + "\n")

print(f"Wrote {OUT} with {len(ds)} examples")
