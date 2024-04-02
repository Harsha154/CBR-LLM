import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.set_device(4)
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct",
                                          trust_remote_code=True, torch_dtype=torch.float16).to(device)

cases_data = pd.read_csv("case_base_3-31.csv")
cases = cases_data.to_dict(orient='records')

for case_info in cases:
    text = case_info["Text passage"]
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    # Extract embeddings
    with torch.no_grad():  # No need to compute gradients
        embeddings = model.get_input_embeddings()(input_ids)
    case_info["embedding"] = embeddings
    case_info["Passage length"] = len(text)
    

updated_cases_data = pd.DataFrame(cases)
updated_cases_data.to_csv("updated_case_base_3-31.csv", index=False)

