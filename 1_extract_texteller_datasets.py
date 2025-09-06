from datasets import load_dataset
import json

ds_online = load_dataset("OleehyO/latex-formulas-80M", "handwritten_online")
ds_nature = load_dataset("OleehyO/latex-formulas-80M", "handwritten_nature")


ds_online = ds_online["train"]
ds_nature  = ds_nature ["train"]
# ds = ds.select_columns(["latex_formula"])

import os
from tqdm import tqdm


all_expressions = [item["latex_formula"] for item in tqdm(ds_nature, desc="Extracting expressions")]
# os.makedirs("./data", exist_ok=True)
with open("./handwritten_nature.json", "w") as f:
    json.dump(all_expressions, f, indent=4, ensure_ascii=False)
    
all_expressions = [item["latex_formula"] for item in tqdm(ds_online, desc="Extracting expressions")]
# os.makedirs("./data", exist_ok=True)
with open("./handwritten_online.json", "w") as f:
    json.dump(all_expressions, f, indent=4, ensure_ascii=False)